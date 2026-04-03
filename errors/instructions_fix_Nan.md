The session has done good work narrowing it down. Layers 0 and 1 work, layer 2 fails — that's a highly constrained problem. There are three clear suspects and you should eliminate them in order of investigation cost, cheapest first.

**Step 1 — Dump weight statistics for all three GDN layers**

This is the fastest thing to do and will immediately tell you if the problem is a corrupted tensor. In your Claude Code session, add a diagnostic block that runs at model load time (not inference time) and prints min/max/has_nan/has_inf for every weight tensor in GDN layers 0, 1, and 2:

```rust
for layer_idx in 0..3 {
    let w = &self.gdn_layers[layer_idx];
    for (name, tensor) in &[
        ("conv_weight", &w.conv_weight),
        ("out_proj",    &w.out_proj),
        ("q_proj",      &w.q_proj),
        // add all weight fields
    ] {
        let t = tensor.dequantize(&Device::Cpu)?
            .to_dtype(DType::F32)?;
        let flat = t.flatten_all()?;
        let min = flat.min(0)?.to_scalar::<f32>()?;
        let max = flat.max(0)?.to_scalar::<f32>()?;
        tracing::info!(
            layer = layer_idx, weight = name,
            min, max,
            has_nan = min.is_nan() || max.is_nan(),
            has_inf = min.is_infinite() || max.is_infinite(),
            "weight stats"
        );
    }
}
```

If any tensor in layer 2 shows NaN/Inf at load time — the problem is the GGUF read and you move to step 2. If all weights are clean — the problem is the forward pass and you move to step 3.

**Step 2 (if weights are corrupt) — Verify GGUF tensor offsets**

Even though `ct.tensor()` reads by name, GGUF Q4_K_M blocks have a specific structure where the scale and min tensors are stored adjacent to the data tensor. If the layer 2 weight name lookup resolves to the right offset but the adjacent scale block is actually from a different layer due to file alignment, you get silent garbage on dequant. 

Tell Claude Code to print the raw GGUF metadata for the layer 2 tensors:

```rust
// Before loading, print the tensor info from the reader
for tensor_info in reader.tensors() {
    if tensor_info.name.contains("layers.2.") {
        tracing::info!(
            name = tensor_info.name,
            offset = tensor_info.offset,
            n_elements = tensor_info.tensor_type,
            "gguf tensor info layer 2"
        );
    }
}
```

Compare the offsets against layers 0 and 1. If they look wrong relative to their neighbours, you have an offset bug in the loader.

**Step 3 (if weights are clean) — Instrument the GDN layer 2 forward op-by-op**

This means the numerical instability is emergent during the forward pass, not present in the weights. Add NaN checks after every operation inside the GDN forward, gated behind `if self.layer_idx == 2`:

```rust
macro_rules! nancheck {
    ($tensor:expr, $op:literal) => {
        if self.gdn_layer_idx == 2 {
            let f = $tensor.flatten_all()?.to_dtype(DType::F32)?;
            let mx = f.max(0)?.to_scalar::<f32>()?;
            if mx.is_nan() || mx.is_infinite() {
                tracing::error!(op = $op, "first NaN at this op in GDN layer 2");
            }
        }
    }
}
```

Apply it after: conv state update, the SSM recurrence step, the gate multiply, and the out_proj matmul. The first one that fires is your root cause.

**Most likely culprit based on the pattern**

Given that layers 0 and 1 work and the code path is identical, the single most likely cause is that `conv_state` or `recurrent_state` for `gdn_idx=2` is being initialized correctly to zeros but after the *first forward pass* the state update accumulates a large value that overflows BF16 range on the next token, producing Inf which then becomes NaN in subsequent ops. This would only manifest at layer 2 if that layer's weight magnitudes happen to be larger than layers 0 and 1.

Check the weight norms: if layer 2's `conv_weight` or `A` matrix has significantly larger values than layers 0 and 1, clamp the state update or cast to F32 for the accumulation step.

Start with step 1 — it will either confirm or rule out a corrupt tensor in under five minutes.

# NEW ISSUES TO CHECK

This is a rich area with several real issues you need to be aware of. Let me break it down into architecture concerns and tool calling concerns separately.

---

## Architecture things to audit in your Qwen3.5 implementation

**GDN block layout for the 9B model**

The block structure for the 0.8B model is explicitly documented as: 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)). The pattern for the 9B is the same ratio — 3 GDN layers per 1 full attention layer, repeating. Your current bug with layer 2 failing while 0 and 1 pass is consistent with this structure, since layers 0/1/2 are the first GDN trio and layer 3 is the first full attention layer. Confirm your block index mapping matches this exactly — if you're off by one in how you assign `gdn_layer_idx` vs `attn_layer_idx`, the weights for the full attention layer could be getting loaded into the third GDN slot.

**GDN recurrent state across decode steps**

Each GDN layer keeps a fixed-size state matrix with dimensions proportional to the head dimension squared (e.g., 128 × 128), independent of sequence length. New tokens update this state incrementally. This means the MambaCache/recurrent state must persist correctly across decode steps and be properly reset between requests. Check that your cache reset logic distinguishes between GDN state (recurrent, must persist within a sequence) and KV cache (paged attention, different lifecycle).

**Chunk-wise parallel training vs your prefill path**

GDN layers train efficiently through a chunk-wise parallel algorithm: the sequence is split into fixed-size chunks, a small local attention-like computation runs in parallel within each chunk, and the recurrent state propagates across chunk boundaries. Your prefill implementation needs to handle chunk boundaries correctly — if you're processing the full prefill sequence as one shot without chunking, you may get numerically different results from what the model was trained to expect, which could surface as instability at longer contexts even after the layer 2 NaN is fixed.

**YaRN RoPE scaling**

Context: 262,144 tokens native; up to 1,010,000 tokens using RoPE scaling (e.g., YaRN). Qwen3.5 uses a large RoPE base frequency for long context. Candle-vllm has a `--yarn-scaling-factor` flag — make sure your Qwen3.5 config is loading the correct base theta and not inheriting Qwen2/Qwen3's value. A wrong theta won't cause NaN at short sequences but will degrade quality at longer ones.

**Sampling parameters**

The official recommended parameters are notably different from typical defaults. Thinking mode for general tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`. The `top_k=20` and `presence_penalty=1.5` combination is important — without it the model tends to repeat and degrade. Make sure your candle-vllm serving defaults for this model reflect these values.

---

## Tool calling issues — these are serious and well-documented

This is the area with the most active bugs. There is a known community-documented set of issues specifically with Qwen3.5 tool calling.

**The chat template bug (highest priority)**

The bugs fixed include: a tool calling crash from `arguments | items` (should use `.items()` with a mapping check), KV-cache reuse breaking with `enable_thinking=false`, inability to close thinking mode, missing `reasoning_content` in tool calling, and parallel tool calls interleaving. These are template-level bugs that affect how tool call arguments are serialized into the prompt. Your candle-vllm implementation will have its own tool call prompt formatter — you need to audit it against all of these issues.

**Unsloth confirmed a universal template fix was required**

Tool-calling improved following chat template fixes. The fix is universal and applies to any Qwen3.5 format and any uploader. This means the GGUF files you downloaded before this fix was published may have had the old template baked in. Verify what template your code is actually using — if it's pulled from the GGUF's embedded metadata, it may be stale.

**The `qwen3_coder` parser**

Qwen 3.5 supports reasoning and tool calling via the Qwen3 parsers: `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. The `qwen3_coder` parser is distinct from the generic `qwen` parser. In candle-vllm you have `--enforce-parser` — make sure you're using `qwen_coder` not `qwen` or `json` for this model, since the output format for tool calls is different.

**Thinking mode bleeds into tool call output**

This is the subtle one. When `enable_thinking=true` (default for 9B), the model emits `<think>...</think>` blocks before its response. If your tool call parser is scanning the output for JSON tool call markers without first stripping or skipping the thinking block, it will either miss the tool call entirely or corrupt the arguments. You need to confirm your parser handles the thinking block delimiter before looking for tool call JSON.

**Parallel tool calls**

Parallel tool call interleaving requires a `\n\n` delimiter between blocks. If a user sends a request that triggers multiple tool calls simultaneously, your parser needs to split on `\n\n` between tool call blocks, not just parse the first one it finds.

---

**Concrete things to check in your codebase at `/home/gqadonis/Projects/prometheus/candle-vllm`:**

Search for wherever you format the tool call system prompt and compare it against the 21-fix template at `https://huggingface.co/barubary/qwen3.5-barubary-attuned-chat-template`. Check your `--enforce-parser` default for Qwen3.5 is `qwen_coder`. Find where you parse tool call output and verify thinking block stripping happens before argument extraction. And confirm you're resetting GDN recurrent state between requests but not within a single request's decode loop.


# FINAL CONSIDERATIONS

Good position to be in. The hard bugs are gone — NaN fixed, RoPE corrected, sampling dialed in. What remains is a configuration chain problem, not a code correctness problem. Here's how to think about priority:

**The 8-token problem is the only thing that matters right now.**

Everything else — garbled output, mixed Chinese/English, strange generations — is a symptom of that one root cause. A properly prompted Qwen3.5 with ChatML should produce at minimum ~28-32 tokens for `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\ntell me something good.<|im_end|>\n<|im_start|>assistant\n`. Getting 8 means the tokenizer is seeing only the raw user string with no template wrapping at all.

**Three things to check in order:**

First, verify the special tokens are actually in the loaded vocabulary. For Qwen3.5, `<|im_start|>` should be token ID 151644 and `<|im_end|>` should be 151645. Add a quick diagnostic that dumps the IDs for those two strings specifically at tokenizer load time. If they come back as unknown or produce multi-token splits, the GGUF tokenizer metadata is not loading those special tokens correctly — they need to be added to the `added_tokens` list, not the standard BPE vocab.

Second, check whether your chat template renderer is actually being invoked. The template is loading from the GGUF (you confirmed that), but loading and invoking are separate steps. Search for where `apply_chat_template` or equivalent is called in your request preprocessing path and add a single log line printing the rendered string length before tokenization. If it's printing the raw user content length rather than a full ChatML-wrapped string, the template call is either being skipped or silently falling through to a default passthrough.

Third, if the template IS being called but producing wrong output, the Qwen3.5 GGUF template is a Jinja2 template with thinking mode logic, multimodal conditionals, and the 21 known bugs from issue #1831. Your Jinja2 renderer may be choking on one of those constructs and silently returning empty or partial output. The simplest test: hardcode a ChatML string directly, bypassing the template renderer entirely:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
tell me something good.<|im_end|>
<|im_start|>assistant
```

Tokenize that string directly and count the tokens. If you get ~28, the template renderer is the problem. If you still get ~8, the special tokens aren't in the vocabulary.

**On the tool calling audit:**

Let it finish before acting on it. The three structural issues to expect from the audit are: whether thinking block stripping exists before the parser runs, whether the parser is `qwen_coder` or a generic JSON parser, and whether parallel call splitting uses `\n\n` as the delimiter. Those are the three most likely failures based on issue #1831. But none of those matter until the template issue is resolved — you can't test tool calling with a model that isn't receiving properly formatted prompts.

**The mental model for where you are:**

You've solved the infrastructure layer (NaN, GDN weights, RoPE). You're now in the configuration layer (template, special tokens). Once that's clean and the model produces coherent English responses, tool calling validation is a single test call away. You're probably one to two hours from a fully functional model at this point.