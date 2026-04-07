# Qwen 3.6 in candle-vllm

This document describes how **Qwen 3.6** family checkpoints map onto the existing **Qwen 3.5 hybrid** implementation (full attention + optional linear attention / GatedDeltaNet), **GGUF** loading, **TurboQuant** KV-cache compression, and how to validate output quality.

## Hugging Face `config.json` architectures

When Alibaba publishes Qwen 3.6 weights, `architectures[0]` may use the following class names. Each is routed to the same loaders and tensor paths as the corresponding **Qwen 3.5** architecture:

| `architectures[0]` | Loader / model |
|--------------------|----------------|
| `Qwen3_6ForCausalLM` | `Qwen3_5` (dense hybrid text) |
| `Qwen3_6ForConditionalGeneration` | `Qwen3_5` (config) / `Qwen3VLForConditionalGeneration` when multimodal |
| `Qwen3_6MoeForCausalLM` | `Qwen3_5MoE` |
| `Qwen3_6MoeForConditionalGeneration` | `Qwen3_5MoE` (VL bundle when applicable) |
| `Qwen3_6NextForCausalLM` | `Qwen3_5MoE` (same as `Qwen3Next*`) |
| `Qwen3_6NextForConditionalGeneration` | `Qwen3_5` or `Qwen3_5MoE` per MoE flags (same as `Qwen3Next*`) |

**Important:** These names must appear in `is_qwen3_hybrid_arch_name` in code so hybrid fields (`layer_types` / `layers_block_type`, etc.) in `config.json` are parsed. If the arch string is missing from that whitelist, the stack can fall back to an incorrect attention pattern and **TurboQuant** may allocate the wrong number of KV layers (`kv_cache_num_layers()`). At startup, look for structured log event **`qwen_hybrid_layout`** with `kv_cache_layers` and per-layer counts.

Some community checkpoints may still report **`Qwen3ForCausalLM`** (dense Qwen3 path) even when the name includes “3.6”; use the checkpoint’s actual `architectures` field to choose the code path.

## GGUF (`--f`)

GGUF metadata `general.architecture` may be **`qwen36`** (in addition to **`qwen35`**). Both are loaded with the same **Qwen 3.5 hybrid** GGUF implementation (`GGUFQWen3_5`).

## ~12 GB VRAM (e.g. RTX 4070 Ti)

- Prefer a **quantized** weight format that fits **weights + activations + KV cache** at batch 1: community **GGUF** (e.g. Q4_K_M) or safetensors with **`--isq q4k`** (see CLI / `docs/CONFIGURATION.md`).
- Set **`--mem`** (KV budget, MB) conservatively; increase only after smoke tests.
- **TurboQuant** (see [`TURBOQUANT_KV_COMPRESSION.md`](TURBOQUANT_KV_COMPRESSION.md)) reduces KV footprint; typical default is **3 bits** with an optional **threshold** policy so short prompts stay in higher fidelity. CLI flags are documented under `candle-vllm` help (`--kvcache-compression-bits`, `--kvcache-compression-policy`, …) and in `models.yaml` as `kvcache_compression`.

Example sketch (adjust model ID, paths, and memory to your card):

```bash
cargo run --release --features cuda,nccl -- \
  --m <org>/<Qwen3.6-model> \
  --d 0 \
  --isq q4k \
  --mem 4096 \
  --kvcache-compression-bits 3 \
  --kvcache-compression-policy threshold_tokens \
  --kvcache-compression-threshold-tokens 4096 \
  --p 8000
```

## Quality checks (avoid “loads but garbage output”)

1. Confirm **`qwen_hybrid_layout`** logs show expected **`full_attention_layers`** / **`linear_attention_layers`** and **`kv_cache_layers`**.
2. Run a short **English** prompt; then repeat with **TurboQuant off** vs **on** to isolate KV compression issues.
3. Optionally compare logits on a fixed small input against another framework.

## See also

- [`TURBOQUANT_KV_COMPRESSION.md`](TURBOQUANT_KV_COMPRESSION.md) — TurboQuant policies and 12 GB tier notes  
- [`docs/opencode.md`](opencode.md) — `--enforce-parser qwen_coder` for tool-heavy Qwen coder workflows  
