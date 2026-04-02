# candle-vllm Configuration Wizard Skill

## Purpose

Interactive configuration wizard that guides users through generating optimized `models.yaml` configuration files for the candle-vllm inference server. The wizard assesses hardware capabilities, recommends model selections, configures quantization strategies, and produces production-ready configuration files.

## Trigger Phrases

- "configure candle-vllm"
- "generate models.yaml"
- "setup model config"
- "config wizard"
- "help me configure"
- "create a model configuration"
- "optimize my candle-vllm setup"

## Workflow

The wizard follows a five-stage pipeline:

### Stage 1: Hardware Assessment

Determine the user's available hardware and constraints:
- GPU type and VRAM (CUDA, Metal/Apple Silicon, CPU-only)
- Number of GPUs and multi-GPU topology
- Available system RAM
- Target deployment environment (development, production, edge)

### Stage 2: Model Selection

Guide the user to an appropriate model based on:
- Task requirements (chat, code generation, reasoning, vision, general purpose)
- Language requirements
- Context window needs
- Latency and throughput targets
- Hardware constraints from Stage 1

### Stage 3: Quantization Strategy

Choose the right quantization approach:
- No quantization (FP16/BF16) for maximum quality
- In-situ quantization (ISQ): q4_0, q4_1, q5_0, q5_1, q8_0, q2k, q3k, q4k, q5k, q6k
- GGUF pre-quantized models
- GPTQ/Marlin 4-bit (CUDA only)
- TurboQuant KV-cache compression for memory savings

### Stage 4: Scheduler and Runtime Configuration

Configure operational parameters:
- Parking lot scheduler settings (batch size, priorities)
- Prompt caching strategy (memory, sled, redis)
- KV cache memory allocation
- Chunked prefill settings
- Vision proxy mode (if applicable)

### Stage 5: YAML Generation

Produce the final configuration:
- Generate validated `models.yaml`
- Generate companion `.env` entries if needed
- Provide CLI invocation examples
- Summarize configuration choices and tradeoffs

## Capabilities

| Capability | Description |
|---|---|
| Model Selection | Recommend models from supported architectures based on task and hardware |
| Hardware-Aware Defaults | Automatically tune memory, batch size, and quantization for detected hardware |
| Multi-Model Configs | Configure multiple models in a single `models.yaml` for A/B testing or multi-purpose serving |
| TurboQuant Integration | Configure KV-cache compression with appropriate precision levels |
| Vision Mode | Set up vision proxy mode for multimodal models (Gemma-3, Mistral-3, Qwen-VL) |
| Prompt Caching | Configure memory, sled, or redis-backed prompt caching |
| Multi-GPU | Configure tensor parallelism across multiple GPUs |
| ISQ Selection | Guide users through in-situ quantization level tradeoffs |
| Validation | Verify generated configurations for correctness before output |

## Supported Model Architectures

- Llama (including Llama 3.x variants)
- Mistral / Ministral (including Mistral 3)
- Phi (Phi-3, Phi-4)
- Qwen2 / Qwen3 (dense and MoE)
- Yi
- StableLM
- Gemma-2 / Gemma-3
- DeepSeek (R1, V2/V3)
- GLM4

## Directory Structure

```
candlevllm-config-wizard/
├── SKILL.md              # This file - skill definition
├── README.md             # User-facing documentation
├── CLAUDE.md             # AI-specific behavioral instructions
├── AGENTS.md             # Agent role definitions
├── plugin.json           # Skill metadata
├── hooks/
│   └── post-generate.sh  # Post-generation validation hook
└── prompts/
    └── model-selection.md # Structured model selection prompt
```

## References

- `example.models.yaml` - Example configuration file in the project root
- `docs/CONFIGURATION.md` - Full configuration documentation
- `docs/coding-standards/README.md` - Coding standards for any generated code
- `.example.env` - Environment variable reference
