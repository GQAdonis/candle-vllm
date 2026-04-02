# Data Type Selection Guide

## Available Data Types

| dtype | Bits | Memory per Param | Best For |
|---|---|---|---|
| bf16 | 16 | 2 bytes | Ampere+ NVIDIA GPUs, Apple Silicon |
| f16 | 16 | 2 bytes | Universal GPU support |
| f32 | 32 | 4 bytes | CPU-only, precision-critical tasks |

## Selection Rules

### bf16 (bfloat16)
- **Recommended for**: Ampere+ NVIDIA GPUs (A100, RTX 30xx/40xx, H100) and Apple Silicon (M1+)
- **Advantages**: Best performance/quality balance; wider dynamic range than f16; native hardware support on modern GPUs
- **Limitations**: Not supported on pre-Ampere NVIDIA GPUs (V100, RTX 20xx)

### f16 (float16)
- **Recommended for**: Universal GPU support including older NVIDIA GPUs
- **Advantages**: Widest hardware compatibility; same memory footprint as bf16
- **Limitations**: Narrower dynamic range than bf16; slight quality loss on some models compared to bf16

### f32 (float32)
- **Recommended for**: CPU-only inference; when maximum precision is required
- **Advantages**: Highest precision; no rounding artifacts
- **Limitations**: 2x memory of f16/bf16; significantly slower on GPUs

## Model-Specific Constraints

| Model Family | Allowed dtypes | Notes |
|---|---|---|
| Mistral 3 / Ministral | bf16, f16 | FP8 models are not supported |
| Flash Attention models | bf16, f16 | f32 not compatible with flash-attn |
| Metal (Apple Silicon) | f16, bf16 | f32 works but is very slow |
| CPU inference | f32 | f16/bf16 may work but f32 is recommended |
| DeepSeek MoE | bf16 | bf16 recommended for stability |

## Decision Flowchart

1. Is the target hardware Apple Silicon? -> Use bf16 (preferred) or f16
2. Is the target hardware NVIDIA Ampere+? -> Use bf16
3. Is the target hardware older NVIDIA? -> Use f16
4. Is the target CPU-only? -> Use f32
5. Is the model Mistral 3 or Ministral? -> Use bf16 or f16 only (no FP8)
6. Using flash attention? -> Must use bf16 or f16
