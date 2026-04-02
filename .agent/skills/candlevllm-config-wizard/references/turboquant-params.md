# TurboQuant Parameter Reference

TurboQuant provides KV-cache compression to reduce memory usage during inference, allowing longer contexts and more concurrent sequences.

## Bit Width Options

| Bits | Compression Ratio | Quality Impact | Recommendation |
|---|---|---|---|
| 2 | 6-10x vs fp16 | Noticeable quality loss | Use only when memory is critical |
| 3 | 5-6x vs fp16 | Minimal quality loss | Recommended default |
| 4 | 3-4x vs fp16 | Near-lossless | Use when quality is priority |

## Compression Policies

| Policy | Description | Use Case |
|---|---|---|
| always | Compress all KV cache entries | Maximum memory savings |
| disabled | No compression | Baseline, quality-critical |
| threshold_tokens: N | Compress after N tokens in sequence | Balance quality for short vs long sequences |
| memory_pressure: {free_block_pct: F} | Compress when free blocks drop below F% | Adaptive, responds to actual memory state |

### Policy Examples

```yaml
# Always compress
turboquant:
  bits: 3
  policy: always

# Compress only after 2048 tokens
turboquant:
  bits: 3
  policy:
    threshold_tokens: 2048

# Compress when less than 20% of blocks are free
turboquant:
  bits: 3
  policy:
    memory_pressure:
      free_block_pct: 0.2
```

## Requirements

- **head_dim** must be a power of 2: 64, 128, or 256
- All currently supported architectures meet this requirement
- Works with both ISQ-quantized and unquantized models
- Compatible with PagedAttention

## Architecture Compatibility

TurboQuant works with all supported model architectures:
- LLAMA, Mistral, Phi, Qwen2/Qwen3, Yi, StableLM, Gemma, DeepSeek, GLM4, QwQ

## Memory Savings Examples

| Model | Context | Standard KV (fp16) | 3-bit TurboQuant | Savings |
|---|---|---|---|---|
| Llama-3-8B | 8K | ~1 GB | ~190 MB | ~81% |
| Llama-3-8B | 32K | ~4 GB | ~750 MB | ~81% |
| Llama-3-70B | 16K | ~5 GB | ~940 MB | ~81% |
| Llama-3-70B | 128K | ~40 GB | ~7.5 GB | ~81% |
| DeepSeek-R1-671B | 16K | ~80 GB | ~15 GB | ~81% |

## Tuning Recommendations

1. **Start with 3-bit**: Best balance of compression and quality for most models
2. **Use threshold_tokens for chat**: Short exchanges stay at full precision; only long conversations get compressed
3. **Use memory_pressure for dynamic workloads**: Automatically adapts to actual usage patterns
4. **Monitor quality**: Run evaluation benchmarks with and without TurboQuant to measure impact for your specific model
