# syntax=docker/dockerfile:1

FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN <<HEREDOC
    apt-get update && \
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
HEREDOC

# Use the workspace-declared minimum Rust version (Rust 1.83+) instead of nightly.
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup toolchain install 1.83.0
RUN rustup default 1.83.0

WORKDIR /candle-vllm

COPY . .

# Rayon threads are limited to minimize memory requirements in CI, avoiding OOM
# NOTE: Avoid nightly-only `-Z` flags in Docker builds.
# NOTE: NCCL feature has known compilation issues when combined with MPI (see DEFAULT_MODEL_FIX.md)
# Use cuda,cudnn,mpi (without nccl) as default. Override WITH_FEATURES if nccl is needed.
ARG CUDA_COMPUTE_CAP=80
ARG RAYON_NUM_THREADS=4
ARG WITH_FEATURES="cuda,cudnn,mpi"
ENV CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP}" \
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS}"
RUN cargo build --release --workspace --locked --features "${WITH_FEATURES}"

FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base
ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80

ARG DEBIAN_FRONTEND=noninteractive

RUN <<HEREDOC
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        openmpi-bin \
        libssl3 && \

    rm -rf /var/lib/apt/lists/*
HEREDOC

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
