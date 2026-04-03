This is a fundamentally different class of problem from everything you've fixed so far. All the previous issues — NaN, RoPE, tokenizer, template — were infrastructure bugs with deterministic fixes. What's described here is a numerical correctness problem in a novel architecture that requires reference comparison to diagnose. The approach changes accordingly.

Tell Claude Code:

---

Stop trying to fix this by inspection. The issue is in the quantized forward pass of a new hybrid architecture and cannot be diagnosed by reading code alone. We need reference values to compare against.

**Step 1 — Get a reference output first.**

Before touching any code, run the same prompt through llama.cpp with the same GGUF file:

```bash
llama-cli -m /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  -p "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n" \
  --no-warmup -n 20 --temp 0 --top-k 1
```

Use `--temp 0 --top-k 1` for greedy decoding so the output is deterministic. If llama.cpp also produces garbled output with this GGUF, the problem is the GGUF file itself or the quantization, not your implementation. Confirm llama.cpp produces coherent English before proceeding.

**Step 2 — Extract intermediate activations at a single layer boundary.**

If llama.cpp is clean, the divergence is in your forward pass. Add a diagnostic mode flag `--dump-activations` that, when set, runs one forward pass and dumps to `/tmp/` the output tensor of each of the first 4 layers as a binary float32 array. Do this in your implementation. Then write a minimal Python script that loads the same GGUF via llama.cpp's Python bindings or the HuggingFace transformers implementation and dumps the same layer outputs for the same input. Compare layer 0 output first. The first layer where your values diverge from reference is the root cause.

**Step 3 — The four specific hypotheses to test in order of likelihood.**

Once you have reference values, check these in order:

First, weight permutation. Qwen3.5's GDN layers have a specific weight layout in the GGUF that may differ from what your `QuantizedGatedDeltaNet` expects. The `conv_weight` and the delta rule weights `W_A`, `W_B` may need transposition or reshape before use. Compare the raw weight tensor shapes you load against the HuggingFace model's weight shapes for the same layer.

Second, the delta rule recurrence itself. The GDN forward pass implements `S_t = S_{t-1} + (v_t - S_{t-1} k_t^T) k_t^T` — a specific error-correcting update. Verify your Rust implementation matches this exactly, particularly the outer product ordering. A transposed outer product gives numerically plausible but semantically wrong values.

Third, dtype accumulation in the recurrence. The state matrix `S` is `[head_dim, head_dim]` = `[128, 128]` in BF16. After 24 GDN layers the accumulated drift in BF16 may be significant. Try forcing the state accumulation to F32 with a final cast back to BF16, as a test only.

Fourth, chunk boundary handling. Your prefill processes the full sequence. The GDN was trained with chunk-wise parallel algorithm with fixed chunk size. If your implementation doesn't chunk the prefill or uses the wrong chunk size, the state at the boundary tokens will be wrong. Check the Qwen3.5 config for `chunk_size` or `training_chunk_size` and verify your prefill matches it.

**Step 4 — Do not attempt fixes without reference comparison.**

Every hypothesis above could be wrong. Fixing without reference values means you're guessing, and guessing at numerical precision bugs in a novel architecture is how you spend another week. The reference comparison narrows it to one layer, one operation, one tensor. Then the fix is obvious.

The reference comparison is the only correct next action.