# syntax=docker/dockerfile:1
#
# CUDA Version Selection:
# - CUDA 12.8 requires driver 560+ (your GKE cluster has 580.x)
# - Tesla T4 is compute capability 7.5 (Turing)
# See: docs/GKE_GPU_DEPLOYMENT.md for driver/CUDA compatibility matrix

FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libssl-dev \
        pkg-config \
        clang \
        libclang-dev \
        libopenmpi-dev \
        openmpi-bin && \
    rm -rf /var/lib/apt/lists/*

# Standardize on a modern stable toolchain (required by some deps using Edition 2024 features).
ARG RUST_TOOLCHAIN=1.92.0
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y --profile minimal --default-toolchain ${RUST_TOOLCHAIN}
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default ${RUST_TOOLCHAIN}

WORKDIR /candle-vllm

COPY . .

# Rayon threads are limited to minimize memory requirements in CI, avoiding OOM
# NOTE: Avoid nightly-only `-Z` flags in Docker builds.
# Keep Docker builds minimal by default. Override WITH_FEATURES for nccl/mpi/cudnn/etc.
#
# CUDA_COMPUTE_CAP: Set based on your target GPU architecture:
#   - 75: Tesla T4, Quadro RTX series (Turing)
#   - 80: A100, A30 (Ampere)
#   - 86: RTX 30xx series (Ampere consumer)
#   - 89: RTX 40xx series, L4, L40 (Ada Lovelace)
#   - 90: H100 (Hopper)
ARG CUDA_COMPUTE_CAP=75
ARG RAYON_NUM_THREADS=4
ARG WITH_FEATURES="cuda"
ENV CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP}" \
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS}"
RUN cargo build --release --workspace --locked --features "${WITH_FEATURES}"

FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base
ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        openmpi-bin \
        libssl3 && \
    rm -rf /var/lib/apt/lists/*

FROM base

COPY --from=builder /candle-vllm/target/release/candle-vllm /usr/local/bin/candle-vllm
RUN chmod +x /usr/local/bin/candle-vllm

# Some runtime images may not include the `libnccl.so` linker symlink.
# If NCCL is installed, restore it so `-lnccl` works at runtime.
RUN if [ ! -e /usr/lib/x86_64-linux-gnu/libnccl.so ] && [ -e /usr/lib/x86_64-linux-gnu/libnccl.so.2 ]; then \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so; \
    fi

EXPOSE 80

# Default to serving the OpenAI-compatible API on $PORT.
CMD ["bash", "-lc", "exec /usr/local/bin/candle-vllm --h 0.0.0.0 --p ${PORT} --d 0"]
