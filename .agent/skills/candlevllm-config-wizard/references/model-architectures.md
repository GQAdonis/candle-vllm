# Supported Model Architectures

Reference table of all model architectures supported by candle-vllm.

| Architecture | Example Models | Context Length | Special Features |
|---|---|---|---|
| LLAMA | Llama-3.x, Llama-3.2-1B/3B | 8K-128K | General purpose, tool calling |
| Mistral | Mistral-7B, Ministral-3B, Mistral-Small | 8K-32K | Sliding window attention, tool calling |
| Phi | Phi-3, Phi-3.5 | 4K-128K | Compact, vision variants |
| Qwen2/Qwen3 | Qwen2.5-0.5B to 72B, Qwen3-8B | 32K-128K | Multilingual, MoE variants |
| Yi | Yi-1.5 | 4K-200K | Long context |
| StableLM | StableLM-2 | 4K-16K | Efficient |
| Gemma | Gemma-2, Gemma-3 | 8K | Vision variants (Gemma-3-VL) |
| DeepSeek | DeepSeek-R1, V2/V3 | 128K | Reasoning, MoE, up to 671B params |
| GLM4 | GLM-4 | 128K | Multilingual |
| QwQ | QwQ-32B | 32K | Reasoning |

## Notes

- **LLAMA** variants include Llama-3, Llama-3.1, and Llama-3.2 families. All support tool calling via `<function=...>` syntax.
- **Mistral/Ministral**: Use BF16/FP16 variants only; FP8 models are not supported. Tool calling uses `[TOOL_CALLS]` prefix.
- **Qwen2/Qwen3**: Includes both dense and MoE variants (Qwen2-MoE, Qwen3-MoE). Tool calling uses `<tool_call>` tags. Qwen3.5 and Qwen3.5-MoE are also supported.
- **DeepSeek**: R1 series supports multi-node deployment for the 671B MoE variant. V2/V3 use MLA (Multi-head Latent Attention).
- **Gemma-3**: Supports native vision via Gemma3-VL architecture.
- **Phi**: Phi-3 models support long context (128K) with sliding window.
- Multi-GPU deployment requires GPU count aligned to powers of 2 (2, 4, 8).
