#!/usr/bin/env bash
candle-vllm \
  --m unsloth/Qwen3.5-9B-GGUF \
  --f Qwen3.5-9B-Q4_K_M.gguf \
  --mem 512 \
  --kvcache-compression-bits 3 \
  --kvcache-compression-policy always \
  --ui-server \
  --p 2000
