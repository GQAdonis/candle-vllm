#!/usr/bin/env bash
candle-vllm \
  --m unsloth/Ministral-3-8B-Reasoning-2512-GGUF \
  --f Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf \
  --mem 256 \
  --kvcache-compression-bits 3 \
  --kvcache-compression-policy always \
  --ui-server \
  --p 2000
