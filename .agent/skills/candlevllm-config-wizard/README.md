# candle-vllm Configuration Wizard

An interactive wizard that helps you generate optimized `models.yaml` configuration files for the candle-vllm inference server.

## What It Does

The configuration wizard walks you through a structured process to produce a complete, validated `models.yaml` file tailored to your hardware and use case. It handles:

- Detecting or assessing your hardware (GPU type, VRAM, system RAM)
- Recommending models that fit your task and hardware constraints
- Selecting the right quantization strategy for your quality/performance tradeoff
- Configuring scheduler, caching, and runtime parameters
- Generating valid YAML output ready for deployment

## How to Invoke

Use any of the following phrases in your conversation:

```
config wizard
configure candle-vllm
generate models.yaml
setup model config
```

The wizard will begin by asking about your hardware environment and guide you through each step.

## Example Conversation Flows

### Quick Setup (Single Model, Known Hardware)

```
User: config wizard
AI:   [asks about hardware]
User: I have an M3 Max with 64GB RAM
AI:   [asks about task]
User: I need a general-purpose chat model
AI:   [recommends models, asks about quantization preference]
User: Go with Qwen3-8B and q4k quantization
AI:   [generates models.yaml]
```

### Multi-Model Configuration

```
User: generate models.yaml
AI:   [asks about hardware]
User: 2x A100 80GB GPUs
AI:   [asks about task]
User: I want a code model and a chat model running together
AI:   [recommends model pairs, configures both]
User: DeepSeek-R1 for reasoning and Llama-3 for chat
AI:   [generates multi-model models.yaml with GPU assignments]
```

### Optimization of Existing Config

```
User: config wizard - I already have a models.yaml but want to optimize it
AI:   [asks to see current config]
User: [pastes config]
AI:   [analyzes and suggests improvements]
```

## Output Format

The wizard produces a `models.yaml` file with the following structure:

```yaml
models:
  - model_id: "organization/model-name"
    # Model source configuration
    source:
      type: "huggingface"  # or "local"
      path: "org/model-name"
    # Quantization settings
    quantization:
      method: "isq"
      level: "q4k"
    # Hardware assignment
    device:
      type: "metal"  # or "cuda", "cpu"
      ids: [0]
    # Runtime settings
    runtime:
      kv_cache_mem_mb: 4096
      dtype: "bf16"
      prefill_chunk_size: 8192
```

When TurboQuant or prompt caching is configured, additional sections are included.

## Tips for Customization

1. **Start conservative**: The wizard defaults to safe settings. You can always tune up after verifying stability.

2. **VRAM is the bottleneck**: For GPU deployments, available VRAM determines your maximum model size and batch capacity. Be accurate about your VRAM when asked.

3. **Quantization tradeoffs**: Lower quantization (q2k, q3k) saves memory but reduces quality. For production use, q4k or q5k offer a good balance. q8_0 is nearly lossless.

4. **Multi-GPU alignment**: When using multiple GPUs, the count must be a power of 2 (2, 4, or 8).

5. **GGUF vs ISQ**: Pre-quantized GGUF files load faster but offer less flexibility. ISQ quantizes at load time, letting you use the same model weights at different precision levels.

6. **Metal limitations**: On Apple Silicon, GPTQ/Marlin quantization is not available. Use ISQ or GGUF instead.

7. **Prompt caching**: For development, memory-backed caching is simplest. For production with multiple server instances, use redis.
