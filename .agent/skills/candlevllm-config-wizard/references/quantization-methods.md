# Quantization Methods Reference

## ISQ (In-Situ Quantization)

Applied at model load time via the `--isq` flag. The model weights are quantized from their original precision to the target format.

Available options: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2k`, `q3k`, `q4k`, `q5k`, `q6k`

## GGUF (Pre-Quantized Format)

Pre-quantized models distributed as `.gguf` files. Common quantization levels:

| Level | Bits | Quality | Speed | Memory vs FP16 |
|---|---|---|---|---|
| Q2_K | 2-3 | Low | Fastest | ~25% |
| Q3_K_M | 3-4 | Fair | Fast | ~35% |
| Q4_K_M | 4-5 | Good | Fast | ~50% |
| Q5_K_M | 5-6 | Very Good | Moderate | ~60% |
| Q6_K | 6 | Excellent | Moderate | ~70% |
| Q8_0 | 8 | Near-lossless | Slower | ~75% |

## GPTQ / Marlin

- 4-bit quantization with hardware acceleration
- CUDA-only (not available on Metal or CPU)
- Marlin format provides significant speedups via optimized CUDA kernels
- AWQ models can be used after conversion to Marlin-compatible format

## Quality Ranking (best to worst)

`q8_0` > `q6k` > `q5k` > `q4k` > `q3k` > `q2k`

## Speed Ranking (fastest to slowest)

`q4k` (fastest) > `q3k` > `q2k` > `q5k` > `q6k` > `q8_0` (slowest)

## Memory Usage Estimates

| Method | Memory vs FP16 | Example (7B model) |
|---|---|---|
| fp16/bf16 | 100% | ~14 GB |
| q8_0 | ~75% | ~10.5 GB |
| q6k | ~70% | ~9.8 GB |
| q5k | ~60% | ~8.4 GB |
| q4k | ~50% | ~7.0 GB |
| q3k | ~35% | ~4.9 GB |
| q2k | ~25% | ~3.5 GB |

## Recommendations

- **Best quality**: Use unquantized (fp16/bf16) if VRAM allows
- **Balanced**: `q4k` provides the best speed/quality/memory tradeoff
- **Memory constrained**: `q3k` or `q2k` for fitting large models in limited VRAM
- **CUDA with speed priority**: GPTQ/Marlin for 4-bit hardware-accelerated inference
- **Pre-quantized convenience**: GGUF files from HuggingFace (e.g., unsloth, TheBloke)
