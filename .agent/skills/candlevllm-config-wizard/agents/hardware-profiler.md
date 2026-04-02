# Hardware Profiler Agent

## Role

Profile the target hardware and compute resource budgets for candle-vllm deployment, producing feature flags, dtype recommendations, and memory allocations.

## Inputs

| Field       | Type     | Required | Description                                      |
|-------------|----------|----------|--------------------------------------------------|
| gpu_type    | string   | yes      | GPU model name (e.g., "RTX 4090", "M2 Ultra", "A100", "CPU") |
| vram_gb     | float    | yes      | Total GPU VRAM in GB (0 for CPU-only)            |
| system_ram  | float    | no       | Total system RAM in GB (default: 16)             |
| os          | string   | yes      | Operating system: "macos", "linux", "windows"    |
| gpu_count   | integer  | no       | Number of GPUs (default: 1)                      |
| cuda_arch   | integer  | no       | CUDA compute capability x100 (e.g., 800 for sm_80). Omit for non-CUDA. |

## Processing Pipeline

### Step 1: Determine Backend

```
if os == "macos":
    backend = "metal"
    multi_gpu = false          # Metal does not support multi-GPU
elif gpu_type == "CPU" or vram_gb == 0:
    backend = "cpu"
    multi_gpu = false
else:
    backend = "cuda"
    multi_gpu = (gpu_count > 1)
```

### Step 2: Select Feature Flags

```
features = []

match backend:
    "metal"  -> features.append("metal")
    "cpu"    -> []  # no GPU features
    "cuda"   ->
        features.append("cuda")
        if multi_gpu:
            features.append("nccl")
        if cuda_arch >= 700:
            features.append("graph")
        if cuda_arch >= 800:
            features.append("flash-attn")
```

### Step 3: Recommend dtype

| Backend | GPU Generation      | Recommended dtype | Fallback  |
|---------|---------------------|-------------------|-----------|
| metal   | any Apple Silicon   | bf16              | f16       |
| cuda    | Ampere+ (sm_80+)   | bf16              | f16       |
| cuda    | Turing (sm_75)     | f16               | f16       |
| cuda    | Volta (sm_70)      | f16               | f32       |
| cuda    | older              | f32               | f32       |
| cpu     | any                | f32               | f16       |

### Step 4: Compute Memory Budget

```
overhead_mb = 500                       # driver, runtime, framework overhead
per_gpu_vram_mb = vram_gb * 1024

available_vram_mb = (per_gpu_vram_mb * 0.90) - overhead_mb

# For multi-GPU, total is sum across devices (tensor parallel)
total_available_mb = available_vram_mb * gpu_count
```

### Step 5: Estimate Capacity

```
# Model size estimate based on dtype
bytes_per_param = { "f32": 4.0, "bf16": 2.0, "f16": 2.0 }
max_model_params_b = total_available_mb / (bytes_per_param[dtype] * 1024)

# Reserve for KV cache: at least 20% of available VRAM
kv_cache_budget_mb = available_vram_mb * 0.20
model_budget_mb = total_available_mb - kv_cache_budget_mb

# Estimate max concurrent sequences
# ~2MB per sequence per layer (rough average for 32-layer model at 4K context)
estimated_max_num_seqs = max(1, int(kv_cache_budget_mb / 64))
```

### Step 6: Build Device IDs

```
if backend == "cpu":
    device_ids = []
elif backend == "metal":
    device_ids = [0]
else:
    device_ids = list(range(gpu_count))
    # Validate: gpu_count should align to power of 2 for tensor parallel
    if gpu_count > 1 and not is_power_of_two(gpu_count):
        warning = "GPU count should be a power of 2 (2, 4, 8) for tensor parallelism"
```

## Output

```yaml
hardware_profile:
  backend: "metal"                    # metal | cuda | cpu
  features:
    - "metal"
  device_ids: [0]
  dtype: "bf16"
  total_vram_mb: 36864                # raw total
  available_vram_mb: 32268            # after overhead and safety margin
  model_budget_mb: 25814              # VRAM for model weights
  kv_cache_budget_mb: 6454            # VRAM for KV cache
  recommended_mem_flag_mb: 6400       # value for --mem CLI flag
  estimated_max_num_seqs: 100
  max_model_params_b: 12.6            # approx max model size in billions
  multi_gpu: false
  warnings: []                        # any advisory messages
  build_command: "cargo build --release --features metal"
  run_flags: "--p 2000 --d 0 --dtype bf16"
```

## Validation Rules

- `vram_gb` must be >= 0. If 0 and `gpu_type != "CPU"`, emit a warning and treat as CPU.
- `gpu_count` must be >= 1. If multi-GPU on macOS, emit an error (not supported).
- If `cuda_arch` is provided but backend is not CUDA, ignore it with a warning.
- If `system_ram < 8`, warn that CPU offloading may be constrained.

## Known GPU Profiles

Pre-populated data for common GPUs (used when user provides only a name):

| GPU             | VRAM (GB) | cuda_arch | Notes                   |
|-----------------|-----------|-----------|-------------------------|
| RTX 4090        | 24        | 890       | Ada Lovelace            |
| RTX 3090        | 24        | 860       | Ampere                  |
| RTX 3080        | 10        | 860       | Ampere                  |
| A100 40GB       | 40        | 800       | Ampere datacenter       |
| A100 80GB       | 80        | 800       | Ampere datacenter       |
| H100            | 80        | 900       | Hopper datacenter       |
| M1 8GB          | 8         | n/a       | Shared memory           |
| M1 Pro 16GB     | 16        | n/a       | Shared memory           |
| M1 Max 32GB     | 32        | n/a       | Shared memory           |
| M2 Ultra 192GB  | 192       | n/a       | Shared memory           |
| M3 Max 48GB     | 48        | n/a       | Shared memory           |
| M4 Max 128GB    | 128       | n/a       | Shared memory           |

For Apple Silicon, `vram_gb` equals the unified memory allocation available to the GPU (typically total system RAM minus ~2-4GB for OS).
