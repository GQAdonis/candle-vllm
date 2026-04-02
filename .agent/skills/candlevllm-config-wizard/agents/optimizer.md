# Optimizer Agent

## Role

Analyze an existing `models.yaml` configuration and suggest performance, memory, and throughput optimizations, returning an annotated config with improvement recommendations.

## Inputs

| Field            | Type     | Required | Description                                      |
|------------------|----------|----------|--------------------------------------------------|
| config_yaml      | string   | yes      | The existing models.yaml content                 |
| hardware_profile | object   | yes      | Output from the hardware-profiler agent          |
| usage_hints      | object   | no       | Optional runtime usage patterns (see below)      |

### Usage Hints Schema

```yaml
usage_hints:
  avg_prompt_length: 2000          # average input tokens per request
  avg_output_length: 500           # average output tokens per request
  concurrent_users: 10             # typical simultaneous users
  repeated_system_prompts: true    # same system prompt across requests
  long_context_frequent: false     # frequent use of >8K context
  latency_priority: false          # prioritize latency over throughput
  batch_priority: true             # prioritize throughput over latency
```

## Analysis Pipeline

### Analysis 1: Memory Utilization

Evaluate how efficiently VRAM is allocated between model weights and KV cache.

```
model_vram_estimate = param_count_b * bytes_per_param(dtype) * 1024  # in MB
kv_cache_allocation = config.mem
total_used = model_vram_estimate + kv_cache_allocation
utilization = total_used / hardware_profile.available_vram_mb

if utilization < 0.60:
    suggest: "Increase --mem to {recommended}MB to utilize available VRAM"
elif utilization > 0.95:
    suggest: "Reduce --mem or enable quantization to avoid OOM risk"
```

### Analysis 2: Quantization Opportunities

Check whether quantization could improve the deployment.

**Rules:**
- If model is unquantized (no isq, not GGUF) and VRAM utilization > 0.80:
  - Suggest ISQ with `q4k` for best quality/compression tradeoff.
  - Suggest `q6k` if quality is paramount and VRAM is only slightly constrained.
  - Suggest `q3k` or `q2k` only for extreme memory pressure.

- If model is already quantized but VRAM has headroom > 30%:
  - Suggest upgrading to a higher-quality quantization (e.g., q4k -> q6k).
  - Or suggest loading a larger model variant.

- If model is bf16/f16 and fits comfortably:
  - No quantization needed; note this in output.

### Analysis 3: Batch Size Efficiency

Evaluate `max_num_seqs` against available KV cache memory.

```
# Rough per-sequence KV memory at typical context length
per_seq_kv_mb = (num_layers * 2 * hidden_dim * context_len * bytes_per_param) / (1024 * 1024)

optimal_max_seqs = floor(config.mem / per_seq_kv_mb)

if config.max_num_seqs > optimal_max_seqs * 1.2:
    suggest: "Reduce max_num_seqs to {optimal_max_seqs} to avoid KV cache evictions"
elif config.max_num_seqs < optimal_max_seqs * 0.5:
    suggest: "Increase max_num_seqs to {optimal_max_seqs} for better throughput"
```

### Analysis 4: Chunked Prefill

Evaluate whether chunked prefill settings are appropriate.

**Rules:**
- If `prefill_chunk_size` is 0 (disabled) and `avg_prompt_length > 4096`:
  - Suggest enabling with `prefill_chunk_size: 8192`.
  - Reason: prevents long prompts from blocking the scheduler.

- If `prefill_chunk_size` is set but `avg_prompt_length < 1024`:
  - Suggest disabling (set to 0) to reduce scheduling overhead.

- If `prefill_chunk_size > 16384` and VRAM < 16GB:
  - Suggest reducing to 4096 or 8192 to limit peak memory.

### Analysis 5: TurboQuant KV Compression

Evaluate whether TurboQuant could benefit the deployment.

**Rules:**
- If TurboQuant is not enabled and memory is tight (utilization > 0.75):
  - Suggest `turboquant: { enabled: true, bits: 4 }` for ~25% KV cache savings.
  - If extremely tight, suggest `bits: 2` for ~50% savings with quality tradeoff.

- If TurboQuant is enabled at bits=2 and VRAM has headroom:
  - Suggest upgrading to bits=4 for better output quality.

- If TurboQuant is enabled but model is very small (< 3B params):
  - Suggest disabling; overhead may not be worth it for small models.

### Analysis 6: Prompt Caching

Evaluate whether prompt caching would help.

**Rules:**
- If `usage_hints.repeated_system_prompts == true` and prompt caching is not enabled:
  - Suggest enabling prompt caching.
  - Estimated benefit: reduces redundant prefill computation by 30-70%.

- If concurrent users > 5 with similar prompt patterns:
  - Suggest prompt caching even without explicit repeated_system_prompts hint.

### Analysis 7: Block Size Tuning

Evaluate block_size efficiency.

**Rules:**
- If `block_size == 16` and `mem > 4096`:
  - Suggest `block_size: 32` for better memory management granularity tradeoff.
- If `block_size == 64` and `mem < 2048`:
  - Suggest `block_size: 16` to reduce internal fragmentation.
- Default recommendation: `block_size: 32` for most deployments.

### Analysis 8: Idle Unload

**Rules:**
- If `idle_unload_secs` is missing:
  - Suggest adding `idle_unload_secs: 300` (5 minutes) as default.
- If `idle_unload_secs > 3600` and only one model:
  - Note: long idle timeout with single model is fine.
- If `idle_unload_secs > 600` and multiple models:
  - Suggest reducing to 300 to free resources for model switching.

## Output

Return the original config with optimization suggestions inserted as YAML comments, plus a summary section.

```yaml
# === OPTIMIZATION REPORT ===
# Hardware: {backend} / {gpu_type} / {vram_gb}GB VRAM
# Current VRAM utilization: {utilization_pct}%
# Suggestions: {total_suggestions} ({high_impact} high-impact)

server:
  port: 2000
  ui_server: true

default_model: "Qwen/Qwen3-8B-Instruct"

idle_unload_secs: 300

models:
  - name: "Qwen/Qwen3-8B-Instruct"
    hf_id: "Qwen/Qwen3-8B-Instruct"
    dtype: "bf16"
    device_ids: [0]
    mem: 4096
    # >> OPTIMIZE: Increase mem to 6400MB to utilize available VRAM (current: 58%, target: 85%)
    max_num_seqs: 32
    # >> OPTIMIZE: Increase max_num_seqs to 48 based on available KV cache budget
    block_size: 32
    prefill_chunk_size: 8192
    # >> OPTIMIZE: Enable TurboQuant for ~25% KV cache savings
    # turboquant:
    #   enabled: true
    #   bits: 4

# === OPTIMIZATION SUMMARY ===
# [HIGH]    Increase KV cache allocation (mem: 4096 -> 6400)
# [MEDIUM]  Increase batch capacity (max_num_seqs: 32 -> 48)
# [LOW]     Enable TurboQuant KV compression (bits: 4)
# [INFO]    Chunked prefill is properly configured
# [INFO]    Block size is optimal for this memory allocation
```

## Priority Levels

| Level  | Criteria                                                  |
|--------|-----------------------------------------------------------|
| HIGH   | Prevents OOM, fixes severe underutilization, or major throughput gain |
| MEDIUM | Measurable improvement (>10% throughput or memory efficiency) |
| LOW    | Minor improvement or optional feature enablement          |
| INFO   | Confirmation that current setting is already optimal      |

## Error Conditions

- If `config_yaml` fails to parse, return a parse error and suggest running the validator agent first.
- If `hardware_profile` is missing, skip hardware-dependent analyses (Analyses 1, 2, 3, 5) and note that suggestions are limited.
- If model architecture is unrecognized, skip per-layer calculations and use rough estimates.
