# Memory Estimation Formulas

## Model Memory

| Precision | Formula | Example (7B) | Example (13B) | Example (70B) |
|---|---|---|---|---|
| fp16/bf16 | params_B * 2 GB | ~14 GB | ~26 GB | ~140 GB |
| q8_0 | params_B * 1.0 GB | ~7 GB | ~13 GB | ~70 GB |
| q6k | params_B * 0.75 GB | ~5.25 GB | ~9.75 GB | ~52.5 GB |
| q4k | params_B * 0.5 GB | ~3.5 GB | ~6.5 GB | ~35 GB |
| q3k | params_B * 0.375 GB | ~2.6 GB | ~4.9 GB | ~26.25 GB |
| q2k | params_B * 0.25 GB | ~1.75 GB | ~3.25 GB | ~17.5 GB |

## KV Cache Memory

### Per-Token KV Cache Size (fp16)

```
per_token_kv_bytes = 2 * num_layers * num_kv_heads * head_dim * 2
```

Where:
- Factor of 2: one for K, one for V
- num_layers: number of transformer layers
- num_kv_heads: number of key-value heads (may differ from query heads in GQA)
- head_dim: dimension per head (typically 64 or 128)
- Final * 2: bytes per fp16 element

### Common Model KV Cache Sizes

| Model | Layers | KV Heads | Head Dim | Per-Token KV (fp16) |
|---|---|---|---|---|
| Llama-3-8B | 32 | 8 | 128 | 128 KB |
| Llama-3-70B | 80 | 8 | 128 | 320 KB |
| Mistral-7B | 32 | 8 | 128 | 128 KB |
| Qwen2.5-7B | 28 | 4 | 128 | 56 KB |
| Qwen2.5-72B | 80 | 8 | 128 | 320 KB |

### KV Cache Budget

```
kv_budget = total_vram - model_size - 500 MB (overhead)
```

### Maximum Concurrent Sequences Estimate

```
max_num_seqs = kv_budget / (avg_context_len * per_token_kv_size)
```

Example: 24 GB VRAM, 7B model at q4k (3.5 GB), Llama-3-8B architecture:
- kv_budget = 24 - 3.5 - 0.5 = 20 GB
- At 4K context: max_seqs = 20 GB / (4096 * 128 KB) = ~39 sequences

## TurboQuant KV Cache Compression

TurboQuant compresses KV cache entries to reduce memory usage:

| TurboQuant Bits | Compression vs fp16 | Effective Per-Token Size |
|---|---|---|
| 2-bit | ~6-10x smaller | ~3/16 of fp16 |
| 3-bit (recommended) | ~5-6x smaller | ~3/16 of fp16 |
| 4-bit | ~3-4x smaller | ~4/16 of fp16 |

### TurboQuant Savings Example

70B model at 16K context with 3-bit TurboQuant:
- Standard fp16 KV cache: ~80 GB
- With 3-bit TurboQuant: ~15 GB

## Block Size Impact

| Block Size | Overhead | Granularity | Recommendation |
|---|---|---|---|
| 16 | Higher | Finer | Memory-constrained, short sequences |
| 32 | Moderate | Moderate | General purpose |
| 64 | Lower | Coarser | High throughput, long sequences |

## --mem Parameter Guidelines

The `--mem` parameter sets the KV cache memory budget in MB:

| Use Case | Recommended --mem |
|---|---|
| Single user, short context | 1024 |
| Single user, long context | 4096 |
| Multi-user chat (5-10 users) | 8192 |
| High-throughput batch | 16384+ |
| Default (if unspecified) | Varies by available VRAM |
