The framing of "fixing language output" is important here — the garbled Chinese/mixed output is not a language selection bug, it's a direct consequence of the tokenizer special tokens issue that was just identified. The model is receiving wrong token IDs for the prompt, so it's pattern-matching against its training distribution incorrectly. Fix the root cause first, not the symptom.

Tell Claude Code:

---

Do not add any language forcing, language detection, or output filtering. The mixed Chinese/English output is caused entirely by the tokenizer not recognizing `<|im_start|>` and `<|im_end|>` as single special tokens — the model is receiving garbled input and responding accordingly. Once the tokenizer is fixed, the model will naturally respond in the language of the prompt.

The fix is in `get_gguf_info()` where the tokenizer is built from GGUF metadata. Do the following:

**1. Audit how special tokens are currently loaded from the GGUF.**

The GGUF file contains `tokenizer.ggml.added_tokens` — a list of special token strings that must be registered as atomic tokens. Check whether `get_gguf_info()` is reading this field and passing it to the tokenizer's `added_tokens` list. If it's only loading the base BPE vocabulary without the added tokens, then `<|im_start|>` (which should be token ID 151644) gets BPE-split into subword pieces instead.

**2. Load the `tokenizer.ggml.added_tokens` array from GGUF metadata and register every entry as a single-token special token.**

In HuggingFace tokenizers terminology, these go into `added_tokens_decoder` with `special: true`. In your Rust tokenizer construction, they need to be added via `add_special_tokens()` or the equivalent before any encoding happens. All 276 of them, not just `<|im_start|>` and `<|im_end|>`.

**3. Verify the fix with a direct tokenizer encode test.**

After the fix, add a one-time startup log that encodes the string `<|im_start|>` and prints its token IDs. It must produce a single token ID `151644`. If it produces multiple IDs, the special tokens still aren't registered correctly. Kill the server after seeing that log — don't run full inference until this check passes.

**4. Verify prompt token count.**

Once the special tokens are correctly registered, a request with `"hello"` should produce `prompt_tokens` in the range of 14-18, not 9. The exact count depends on the system prompt. If you still see 9 tokens, the special tokens are still not being treated as atomic.

Do not touch sampling parameters, language models, output filtering, or any other part of the stack until this tokenizer encode test passes. Everything else is working correctly — this is the last infrastructure issue standing between you and coherent output.