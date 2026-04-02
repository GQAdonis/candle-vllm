# Hardware Profiles Reference

## Common Hardware Configurations

| Hardware | VRAM | Recommended dtype | Max Model (approx) | Feature Flag |
|---|---|---|---|---|
| A100 80GB | 80 GB | bf16 | 70B | cuda |
| A100 40GB | 40 GB | bf16 | 34B | cuda |
| H100 80GB | 80 GB | bf16 | 70B | cuda |
| RTX 4090 | 24 GB | f16/bf16 | 13B | cuda |
| RTX 4080 | 16 GB | f16/bf16 | 7B | cuda |
| RTX 3090 | 24 GB | f16 | 13B | cuda |
| RTX 3080 | 10 GB | f16 | 7B q4k | cuda |
| M1 Pro 16GB | 16 GB | f16 | 7B q4k | metal |
| M2 Max 32GB | 32 GB | f16/bf16 | 13B | metal |
| M3 Max 48GB | 48 GB | bf16 | 34B | metal |
| M4 Max 128GB | 128 GB | bf16 | 70B | metal |
| CPU 32GB RAM | 32 GB | f32 | 3B | none |

## Feature Flag Mapping

| Hardware Backend | Build Command |
|---|---|
| NVIDIA GPU (single) | `cargo build --release --features cuda` |
| NVIDIA GPU (multi, single node) | `cargo build --release --features cuda,nccl` |
| NVIDIA GPU (with CUDA graphs) | `cargo build --release --features cuda,nccl,graph` |
| NVIDIA GPU (with flash attention) | `cargo build --release --features cuda,nccl,graph,flash-attn` |
| NVIDIA GPU (multi-node) | `cargo build --release --features cuda,nccl,mpi` |
| Apple Silicon | `cargo build --release --features metal` |
| CPU only | `cargo build --release` |

## Multi-GPU Considerations

- GPU count must be a power of 2: 2, 4, or 8 GPUs
- Use `--d 0,1` for 2 GPUs, `--d 0,1,2,3` for 4 GPUs, etc.
- Disable P2P with `NCCL_P2P_DISABLE=1` if experiencing communication issues
- Multi-GPU is CUDA-only; Metal supports single-device only

## Memory Budget Guidelines

- Reserve ~500 MB for runtime overhead
- Model weights: see quantization-methods.md for per-method estimates
- KV cache: remaining VRAM after model + overhead
- Increase `--mem` parameter for larger batch sizes or longer contexts
