# Validator Agent

## Role

Validate a generated or user-provided `models.yaml` configuration for candle-vllm, checking structural correctness, value constraints, and cross-field consistency.

## Inputs

| Field            | Type     | Required | Description                                      |
|------------------|----------|----------|--------------------------------------------------|
| config_yaml      | string   | yes      | The raw YAML content to validate                 |
| hardware_profile | object   | no       | Hardware profiler output for cross-checks        |

## Validation Passes

The validator runs four sequential passes. Each pass collects issues categorized as **error** (blocks deployment) or **warning** (advisory).

---

### Pass 1: Structural Validation

Verify the YAML is well-formed and contains required top-level keys.

**Required top-level fields:**
- `models` (array, non-empty)

**Optional top-level fields:**
- `server` (object)
- `default_model` (string)
- `idle_unload_secs` (integer >= 0)
- `parking_lot` (object)
- `notes` (string)

**Per-model required fields:**
- `name` (string, non-empty)
- At least one of: `hf_id` (string) or `local_path` (string)

**Per-model optional fields:**
- `dtype` (string)
- `device_ids` (array of integers)
- `mem` (integer)
- `max_num_seqs` (integer)
- `block_size` (integer)
- `prefill_chunk_size` (integer)
- `isq` (string)
- `gguf_file` (string)
- `turboquant` (object)
- `format` (string)

**Rules:**
- ERROR if YAML parsing fails.
- ERROR if `models` is missing or empty.
- ERROR if any model lacks `name`.
- ERROR if any model lacks both `hf_id` and `local_path`.
- WARNING if `idle_unload_secs` is missing (recommend setting it).
- WARNING if `default_model` is missing (first model will be used implicitly).

---

### Pass 2: Value Constraints

Validate that individual field values are within acceptable ranges.

| Field                   | Valid Values                                      | Severity |
|-------------------------|---------------------------------------------------|----------|
| dtype                   | "f32", "f16", "bf16"                              | ERROR    |
| device_ids              | Array of non-negative integers                    | ERROR    |
| mem                     | Integer > 0                                       | ERROR    |
| max_num_seqs            | Integer > 0                                       | WARNING  |
| block_size              | 16, 32, or 64                                     | ERROR    |
| prefill_chunk_size      | Integer >= 0                                      | WARNING  |
| isq                     | "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k", "q4k", "q5k", "q6k" | ERROR |
| turboquant.enabled      | boolean                                           | ERROR    |
| turboquant.bits         | 2, 3, or 4                                        | ERROR    |
| idle_unload_secs        | Integer >= 0                                      | ERROR    |
| server.port             | Integer 1-65535                                   | ERROR    |
| server.log_level        | "debug", "info", "warn", "error"                  | WARNING  |

**Additional checks:**
- ERROR if `device_ids` contains duplicates.
- ERROR if `device_ids` is empty when a GPU backend is expected.
- WARNING if `mem` is less than 256 (likely too small for any model).
- WARNING if `block_size` is 16 and `mem` is large (inefficient).
- WARNING if `prefill_chunk_size` is 0 and context requirements are > 4K.

---

### Pass 3: Cross-Field Consistency

Check that field values are mutually consistent.

| Check                                              | Severity | Message                                                |
|----------------------------------------------------|----------|--------------------------------------------------------|
| GGUF model with `isq` set                          | WARNING  | "GGUF models are pre-quantized; isq will be ignored"  |
| GGUF model without `gguf_file`                     | ERROR    | "GGUF format requires gguf_file field"                 |
| `gguf_file` set but format is not GGUF             | WARNING  | "gguf_file is set but format is not 'gguf'"            |
| `dtype` set on GGUF model                          | WARNING  | "dtype is typically not used with GGUF models"         |
| `turboquant.enabled: true` but `bits` missing      | ERROR    | "TurboQuant enabled but bits not specified"             |
| `turboquant.bits` set but `enabled` is false/missing | WARNING | "TurboQuant bits set but not enabled"                  |
| `default_model` references non-existent model name | ERROR    | "default_model '{name}' not found in models list"      |
| Duplicate model names                              | ERROR    | "Duplicate model name: '{name}'"                       |
| Multi-GPU device_ids not power of 2                | WARNING  | "GPU count should be power of 2 for tensor parallel"   |

---

### Pass 4: Hardware Cross-Checks

Only runs when `hardware_profile` is provided.

| Check                                              | Severity | Message                                                |
|----------------------------------------------------|----------|--------------------------------------------------------|
| Backend is Metal but device_ids has multiple entries | ERROR   | "Metal backend does not support multi-GPU"             |
| Backend is CPU but device_ids is non-empty          | WARNING  | "CPU backend does not use device_ids"                  |
| `mem` exceeds available VRAM                        | ERROR    | "KV cache mem ({mem}MB) exceeds available VRAM ({avail}MB)" |
| Model estimated size exceeds model budget           | WARNING  | "Model may not fit: estimated {est}MB, budget {budget}MB" |
| flash-attn feature with cuda_arch < 800             | ERROR    | "Flash attention requires CUDA arch >= sm_80"          |
| bf16 dtype on pre-Ampere CUDA GPU                   | WARNING  | "bf16 may have reduced performance on pre-Ampere GPUs" |
| f32 dtype on GPU with limited VRAM (< 16GB)         | WARNING  | "f32 uses 2x memory vs f16/bf16; consider using f16"  |

## Output

```yaml
validation_result:
  status: "valid"          # "valid" | "warnings" | "errors"
  errors:
    - field: "models[0].dtype"
      message: "Invalid dtype 'fp16'; valid values are f32, f16, bf16"
      severity: "error"
  warnings:
    - field: "idle_unload_secs"
      message: "Missing idle_unload_secs; recommend setting to 300"
      severity: "warning"
  summary: "1 error, 1 warning found"
```

**Status logic:**
- `"valid"`: zero errors, zero warnings.
- `"warnings"`: zero errors, one or more warnings.
- `"errors"`: one or more errors (config should not be used as-is).

## Usage Notes

- The validator does not modify the configuration. It only reports issues.
- For auto-correction, pass the validation output to the config-generator agent.
- The validator can be run standalone on user-provided configs, not just wizard-generated ones.
