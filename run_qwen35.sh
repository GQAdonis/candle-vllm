#!/usr/bin/env bash
cargo run --release --features cuda,cudnn,nccl,graph -- \
  --m unsloth/Qwen3.5-9B-GGUF \
  --f Qwen3.5-9B-Q8_0.gguf \
  --ui-server
