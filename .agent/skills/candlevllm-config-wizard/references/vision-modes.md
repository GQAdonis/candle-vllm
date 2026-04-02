# Vision Mode Reference

candle-vllm supports image understanding through two modes: proxy vision and native multi-modal processing.

## Vision Modes

| Mode | Description | Latency | Quality | Setup Complexity |
|---|---|---|---|---|
| disabled | No vision support (default) | N/A | N/A | None |
| proxy | External vision model describes images as text | Higher (two-pass) | Good | Moderate |
| native | Direct multi-modal processing within the model | Lower (single-pass) | Best | Low (model-dependent) |

## Proxy Vision Mode

An external vision model processes images and generates text descriptions, which are then fed to the primary LLM.

### Configuration

```yaml
vision:
  mode: proxy
  vision_proxy:
    hf_id: "google/gemma-3-4b-it"  # Vision model to use
    prompt_template: "Describe this image in detail: {image}"
```

### When to Use Proxy Mode

- The primary LLM is text-only but vision capabilities are needed
- You want to decouple the vision model from the text model (e.g., use a small vision model with a large text model)
- The primary model does not have a native vision variant
- You want to use a specialized vision model for image understanding

### Proxy Mode Considerations

- Adds latency due to two-pass processing (vision model + text model)
- Image description quality depends on the proxy vision model
- Text descriptions may lose fine-grained visual details
- The prompt_template controls how the vision model is prompted

## Native Vision Mode

The model directly processes both text and image inputs in a single forward pass.

### Supported Native Vision Models

| Model | Architecture | Image Resolution | Notes |
|---|---|---|---|
| Gemma-3-VL | gemma3_vl | Variable | Google's vision-language model |
| Mistral-3-VL | mistral3_vl | Variable | Mistral's vision-language model |
| Qwen3-VL | qwen3_vl | Variable | Alibaba's vision-language model |

### Configuration

```yaml
vision:
  mode: native
```

No additional configuration needed for native mode -- the model architecture handles multi-modal inputs directly.

### When to Use Native Mode

- The model has a native vision variant (Gemma3-VL, Mistral3-VL, Qwen3-VL)
- Lowest latency is required (single-pass processing)
- Best image understanding quality is needed
- Simplified deployment (single model handles both modalities)

## Decision Guide

1. Does the model support native vision? (Gemma3-VL, Mistral3-VL, Qwen3-VL)
   - Yes -> Use native mode
   - No -> Continue to step 2
2. Is vision capability needed?
   - Yes -> Use proxy mode with a suitable vision model
   - No -> Use disabled mode (default)
3. Is latency critical?
   - Yes -> Prefer native mode with a vision-capable model
   - No -> Proxy mode is acceptable
