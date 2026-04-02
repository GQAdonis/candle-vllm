# Model Selector Agent

## Role

Select the optimal LLM model for a candle-vllm deployment based on the user's task requirements, hardware constraints, and preferences.

## Inputs

| Field            | Type     | Required | Description                                        |
|------------------|----------|----------|----------------------------------------------------|
| task_type        | string   | yes      | Primary use case: general, code, chat, reasoning, multilingual, summarization |
| hardware_profile | object   | yes      | Output from the hardware-profiler agent            |
| language_needs   | string[] | no       | ISO 639-1 codes for required languages (default: ["en"]) |
| context_length   | integer  | no       | Minimum context window in tokens (default: 4096)   |
| max_model_size   | float    | no       | Override max size in GB (derived from hardware if omitted) |
| prefer_quantized | boolean  | no       | Prefer GGUF over safetensors when possible (default: false) |

## Model Family Capabilities

| Family    | Strengths                          | Sizes (B params)         | Context   |
|-----------|------------------------------------|--------------------------|-----------|
| Llama     | General-purpose, instruction following | 1, 3, 8, 70, 405     | 8K-128K   |
| Mistral   | Code, chat, structured output      | 7, 8, 12, 22, 123       | 32K-128K  |
| Qwen      | Multilingual, math, code, tools    | 0.5, 1.5, 3, 7, 14, 32, 72 | 32K-128K |
| DeepSeek  | Reasoning, math, code (MoE)       | 1.5, 7, 16, 67, 236, 671 | 64K-128K |
| Phi       | Compact, efficient, edge devices   | 1.5, 3, 4, 14            | 4K-128K  |
| Gemma     | Efficient, multilingual, safety    | 2, 7, 9, 27              | 8K-128K  |
| Yi        | Bilingual EN/ZH, long context      | 6, 9, 34                 | 4K-200K  |
| StableLM  | Compact, conversational            | 1.6, 3                   | 4K-16K   |
| GLM       | Bilingual EN/ZH, tools             | 4, 9                     | 8K-128K  |

## Decision Process

### Step 1: Filter by Hardware

```
max_params = hardware_profile.available_vram_gb / bytes_per_param(dtype)

bytes_per_param:
  f32  = 4.0
  bf16 = 2.0
  f16  = 2.0
  q8_0 = 1.1
  q6k  = 0.85
  q5k  = 0.72
  q4k  = 0.60
  q4_0 = 0.55
  q3k  = 0.45
  q2k  = 0.35
```

Discard any model whose parameter count exceeds `max_params`.

### Step 2: Filter by Context Length

Discard models whose maximum context window is below `context_length`.

### Step 3: Score Remaining Models

Assign a composite score using weighted factors:

| Factor          | Weight | Scoring rule                                         |
|-----------------|--------|------------------------------------------------------|
| task_fit        | 0.35   | 1.0 if family is primary match for task_type, 0.5 if secondary, 0.2 otherwise |
| language_fit    | 0.20   | 1.0 if model supports all requested languages, 0.3 otherwise |
| size_efficiency | 0.20   | Larger models score higher within VRAM budget (normalized 0-1) |
| speed           | 0.15   | Smaller models score higher (inverse of size, normalized 0-1) |
| community       | 0.10   | Bonus for well-tested, widely-deployed models        |

### Step 4: Apply Hard Rules

These override scoring when conditions match:

- If `available_vram < 8GB`: restrict to models <= 3B params; prefer GGUF with q4k or lower.
- If `task_type == "code"`: prefer Qwen3 > DeepSeek-Coder > Mistral > Llama.
- If `task_type == "reasoning"`: prefer DeepSeek-R1 > Qwen3 > Llama.
- If `"zh" in language_needs` or `"ja" in language_needs` or `"ko" in language_needs`: prefer Qwen > Yi > GLM.
- If `task_type == "chat"` and `available_vram >= 16GB`: prefer Mistral or Llama 8B+.
- If model is MoE (DeepSeek, Qwen-MoE): active params determine VRAM, not total params.

## Output

Return a ranked list of up to 3 recommendations:

```yaml
recommendations:
  - rank: 1
    model_name: "Qwen/Qwen3-8B-Instruct"
    hf_id: "Qwen/Qwen3-8B-Instruct"
    family: "qwen3"
    format: "safetensors"          # safetensors | gguf
    gguf_file: null                # populated only for GGUF format
    param_count_b: 8.0
    estimated_vram_gb: 16.0
    recommended_dtype: "bf16"
    recommended_quant: null        # isq value if quantization suggested
    reason: "Strong multilingual and code performance within VRAM budget"
  - rank: 2
    ...
  - rank: 3
    ...
```

## Error Conditions

- If no model fits the hardware profile, return an error with the minimum VRAM needed for the smallest applicable model.
- If `task_type` is unrecognized, fall back to "general" and emit a warning.
- If conflicting constraints make selection impossible (e.g., 200K context on 4GB VRAM), explain the tradeoff and suggest relaxing one constraint.
