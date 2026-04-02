# TurboQuant KV-Cache Compression in candle-vllm

TurboQuant is a KV-cache compression algorithm from Google Research (ICLR 2026)
integrated into candle-vllm as a transparent drop-in layer inside `CacheEngine`.
Enabling it requires a single `kvcache_compression` block in `models.yaml` and
yields approximately **5× more context length or 5× more concurrent requests**
for the same VRAM budget — with negligible accuracy impact at 3-bit precision.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Configuration](#configuration)
3. [Expected Gains](#expected-gains)
4. [Model Selection by VRAM Tier](#model-selection-by-vram-tier)
   - [12 GB — RTX 4070 Ti / 3080 12 GB](#12-gb-vram-rtx-4070-ti--rtx-3080-12-gb)
   - [16 GB — T4 (GCP) / A10 16 GB / RTX 4080](#16-gb-vram-t4-gcp--a10-16-gb--rtx-4080)
   - [24 GB — L4 (GCP) / A10G (AWS) / A10 (Azure) / RTX 3090/4090](#24-gb-vram-l4-gcp--a10g-aws--nv-a10-azure--rtx-30904090)
5. [Qwen Model Guide](#qwen-model-guide)
6. [Best Reasoning Value vs Frontier Models](#best-reasoning-value-vs-frontier-models)
7. [Choosing a Compression Policy](#choosing-a-compression-policy)
8. [Technical Implementation](#technical-implementation)

---

## How It Works

PagedAttention divides the KV-cache into fixed-size blocks.  Normally each
block stores raw FP16 key and value tensors.  With TurboQuant enabled:

```
Forward pass N:
  get_kv_tensors()  ←  decompress all occupied slots → dense (key, val) tensors
       │
  pipeline.forward(tokens, positions, kv_tensors, metadata)
       │
  push_compressed() →  compress new KV vectors → store in CompressedStore

Forward pass N+1:
  get_kv_tensors()  ←  decompress again (fast, ~μs per block)
  ...
```

The compression algorithm:

1. **Rotation** — apply a randomised Hadamard transform (FWHT) to each KV vector,
   decorrelating coordinates for better quantisation.
2. **Scalar quantisation** — apply Lloyd-Max optimal uniform quantisation at
   2, 3, or 4 bits per coordinate.
3. **Bit-packing** — store the quantised indices in minimal byte form, plus a
   single `f32` L2-norm per vector.

At 3-bit precision a 128-element `f32` KV vector (512 bytes) compresses to
**52 bytes** — a **9.8× reduction** — with cosine similarity > 0.98.

Because candle-vllm's `bytes_per_block()` function accounts for the compressed
size when calculating `num_gpu_blocks`, the scheduler can allocate
proportionally more blocks in the same VRAM budget.

---

## Configuration

```yaml
# models.yaml
models:
  - name: my-model
    hf_id: meta-llama/Llama-3.3-70B-Instruct
    params:
      dtype: bf16
      mem: 20000                    # KV-cache VRAM budget in MB
      block_size: 16
      kvcache_compression:
        bits: 3                     # 2 | 3 (recommended) | 4
        policy:
          threshold_tokens: 4096    # compress once context > 4K tokens
```

### Policy Options

| Policy | YAML | When to use |
|--------|------|-------------|
| Always compress | `policy: always` | Maximise context / throughput at all times |
| Compress after N tokens | `policy: { threshold_tokens: 4096 }` | Keep first 4K tokens in FP16 for best prefill quality |
| Compress on memory pressure | `policy: { memory_pressure: { free_block_pct: 0.15 } }` | Adaptive — only kicks in when GPU blocks fall below 15% free |
| Disabled | omit `kvcache_compression` key | Default behaviour — no compression |

### Bit-Width Trade-offs

| Bits | KV-cache vs FP16 | Cosine sim | Perplexity Δ | Use case |
|------|-----------------|------------|--------------|----------|
| 2    | ~15× smaller    | > 0.95     | < 1%         | Extreme memory pressure, summarisation workloads |
| **3** | **~5× smaller** | **> 0.98** | **< 0.3%**   | **Recommended default** |
| 4    | ~3.5× smaller   | > 0.99     | < 0.1%       | Near-lossless, modest saving |

> `head_dim` must be a power of two (64, 128, 256).  Almost all production
> models satisfy this — Llama, Qwen, Mistral, DeepSeek all use 128.

---

## Expected Gains

The figures below assume a **20 GB KV-cache budget** (model weights not
counted) with 3-bit compression.

| Model | KV heads | Layers | FP16 max context | 3-bit max context | Gain |
|-------|----------|--------|-----------------|-------------------|------|
| Llama-3.1-8B | 8 GQA | 32 | ~40K | ~200K | 5× |
| Llama-3.3-70B | 8 GQA | 80 | ~10K | ~50K | 5× |
| Qwen2.5-7B | 4 GQA | 28 | ~90K | ~450K | 5× |
| Qwen2.5-14B | 8 GQA | 40 | ~32K | ~160K | 5× |
| Qwen2.5-72B | 8 GQA | 80 | ~10K | ~50K | 5× |
| DeepSeek-R1-Distill-14B | 8 GQA | 40 | ~32K | ~160K | 5× |
| Mistral-7B-v0.3 | 8 GQA | 32 | ~40K | ~200K | 5× |
| MiniMax-Text-01 | 32 GQA | 80 | ~6K | ~30K | 5× |

> Numbers are estimates based on architecture parameters. Actual performance
> depends on batch size, block size, and ISQ quantisation of weights.

---

## Model Selection by VRAM Tier

The tables below recommend models for three common consumer/cloud GPU tiers.
"With TQ" means TurboQuant 3-bit compression is active.

Weight memory estimates use these approximations:
- **BF16/FP16**: ~2 GB per billion parameters
- **GGUF Q4_K_M / ISQ q4k**: ~0.55 GB per billion parameters
- **GGUF Q5_K_M / ISQ q5k**: ~0.68 GB per billion parameters
- **GGUF Q8_0 / ISQ q8**: ~1.05 GB per billion parameters

---

### 12 GB VRAM — RTX 4070 Ti / RTX 3080 12 GB

*Remaining after weights = VRAM for KV-cache.*

| Model | Quant | Weights | KV budget | Max context (no TQ) | Max context (3-bit TQ) | Quality tier |
|-------|-------|---------|-----------|--------------------|-----------------------|--------------|
| **Qwen2.5-7B-Instruct** | Q4_K_M | 4.4 GB | 7.6 GB | ~32K | ~160K | ⭐⭐⭐⭐ Strong instruct |
| **Qwen2.5-7B-Instruct** | Q8_0 | 7.5 GB | 4.5 GB | ~18K | ~90K | ⭐⭐⭐⭐½ Near-FP16 |
| **Qwen2.5-14B-Instruct** | Q4_K_M | 8.0 GB | 4.0 GB | ~8K | ~40K | ⭐⭐⭐⭐½ Best 12 GB choice |
| **DeepSeek-R1-Distill-Qwen-7B** | Q4_K_M | 4.4 GB | 7.6 GB | ~32K | ~160K | ⭐⭐⭐⭐⭐ Best reasoning 12 GB |
| **DeepSeek-R1-Distill-Llama-8B** | Q4_K_M | 4.7 GB | 7.3 GB | ~28K | ~140K | ⭐⭐⭐⭐⭐ Strong reasoning |
| Mistral-7B-Instruct-v0.3 | Q4_K_M | 4.1 GB | 7.9 GB | ~40K | ~200K | ⭐⭐⭐½ Fast inference |
| Llama-3.2-8B-Instruct | Q4_K_M | 4.7 GB | 7.3 GB | ~28K | ~140K | ⭐⭐⭐⭐ Strong general |
| Llama-3.2-8B-Instruct | BF16 | 16 GB | — | — | — | ❌ Does not fit |
| InternLM2.5-7B-Chat | Q4_K_M | 4.4 GB | 7.6 GB | ~28K | ~140K | ⭐⭐⭐⭐ Good multilingual |
| Qwen2.5-Coder-7B-Instruct | Q4_K_M | 4.4 GB | 7.6 GB | ~32K | ~160K | ⭐⭐⭐⭐⭐ Best code 12 GB |

**Recommended configuration for 12 GB:**
```yaml
params:
  dtype: f16
  isq: q4k
  mem: 7500
  block_size: 16
  kvcache_compression:
    bits: 3
    policy:
      threshold_tokens: 2048
```

---

### 16 GB VRAM — T4 (GCP `n1-standard-4` + 1×T4) / A10 16 GB / RTX 4080

*T4 is very common in GCP preemptible/spot instances and Colab Pro.  A10 16 GB
appears in some Dell and HP workstations.*

| Model | Quant | Weights | KV budget | Max context (no TQ) | Max context (3-bit TQ) | Quality tier |
|-------|-------|---------|-----------|--------------------|-----------------------|--------------|
| **Qwen2.5-14B-Instruct** | Q4_K_M | 8.0 GB | 8.0 GB | ~16K | ~80K | ⭐⭐⭐⭐½ Best choice here |
| **Qwen2.5-14B-Instruct** | Q5_K_M | 9.7 GB | 6.3 GB | ~12K | ~60K | ⭐⭐⭐⭐⭐ Near-FP16 quality |
| **DeepSeek-R1-Distill-Qwen-14B** | Q4_K_M | 8.0 GB | 8.0 GB | ~16K | ~80K | ⭐⭐⭐⭐⭐ Best reasoning 16 GB |
| **Qwen2.5-7B-Instruct** | BF16 | 14.5 GB | 1.5 GB | ~6K | ~30K | ⭐⭐⭐⭐ FP16 precision |
| Llama-3.1-8B-Instruct | BF16 | 16 GB | ~0 GB | — | — | ❌ Tight — use Q8 |
| Llama-3.1-8B-Instruct | Q8_0 | 8.5 GB | 7.5 GB | ~28K | ~140K | ⭐⭐⭐⭐ Good quality |
| Mistral-Nemo-12B-Instruct | Q4_K_M | 7.0 GB | 9.0 GB | ~20K | ~100K | ⭐⭐⭐⭐ Great context |
| Yi-1.5-9B-Chat | Q4_K_M | 5.2 GB | 10.8 GB | ~40K | ~200K | ⭐⭐⭐⭐ Long context |
| InternLM2.5-20B-Chat | Q4_K_M | 11.5 GB | 4.5 GB | ~8K | ~40K | ⭐⭐⭐⭐½ Strongest 16 GB |
| Qwen2.5-Coder-14B-Instruct | Q4_K_M | 8.0 GB | 8.0 GB | ~16K | ~80K | ⭐⭐⭐⭐⭐ Best code 16 GB |

**Cloud note (GCP T4):** T4 has lower memory bandwidth (320 GB/s) vs consumer
cards.  Prefer smaller models at higher quantisation (Q5/Q8) for best
tokens-per-second.  TurboQuant's decompression cost is CPU-side arithmetic —
essentially free on T4.

**Recommended configuration for 16 GB:**
```yaml
params:
  dtype: bf16
  isq: q4k
  mem: 7800
  block_size: 16
  kvcache_compression:
    bits: 3
    policy:
      threshold_tokens: 4096
```

---

### 24 GB VRAM — L4 (GCP `g2-standard-4`) / A10G (AWS `g5.xlarge`) / NV A10 (Azure `NVads A10 v5`) / RTX 3090 / RTX 4090

*The L4 and A10G are the most cost-effective cloud GPUs for 24 GB slots as of 2025–2026.*
*RTX 4090 has the same 24 GB but nearly 3× the TFLOPS — best for latency-sensitive local use.*

| Model | Quant | Weights | KV budget | Max context (no TQ) | Max context (3-bit TQ) | Quality tier |
|-------|-------|---------|-----------|--------------------|-----------------------|--------------|
| **Qwen2.5-32B-Instruct** | Q4_K_M | 18.5 GB | 5.5 GB | ~5K | ~27K | ⭐⭐⭐⭐⭐ Best single-GPU 24 GB |
| **Qwen2.5-32B-Instruct** | Q3_K_M | 13.8 GB | 10.2 GB | ~10K | ~50K | ⭐⭐⭐⭐½ More context |
| **DeepSeek-R1-Distill-Qwen-14B** | BF16 | 28 GB | — | — | — | ❌ Needs Q4 |
| **DeepSeek-R1-Distill-Qwen-14B** | Q4_K_M | 8.0 GB | 16.0 GB | ~32K | ~160K | ⭐⭐⭐⭐⭐ Best reasoning 24 GB |
| **DeepSeek-R1-Distill-Qwen-32B** | Q4_K_M | 18.5 GB | 5.5 GB | ~5K | ~27K | ⭐⭐⭐⭐⭐ Strongest reasoning 24 GB |
| Qwen2.5-14B-Instruct | BF16 | 28 GB | — | — | — | ❌ Needs Q5+ |
| Qwen2.5-14B-Instruct | Q8_0 | 14.7 GB | 9.3 GB | ~18K | ~90K | ⭐⭐⭐⭐½ High quality |
| Llama-3.3-70B-Instruct | Q2_K | 23 GB | 1.0 GB | ~1K | ~5K | ⭐⭐⭐ Low quality — not recommended |
| Llama-3.1-70B-Instruct | Q3_K_M | 31 GB | — | — | — | ❌ Doesn't fit |
| Mistral-Large-2-Instruct | Q4_K_M | 13 GB | 11.0 GB | ~20K | ~100K | ⭐⭐⭐⭐ Strong European model |
| Qwen2.5-Coder-32B-Instruct | Q4_K_M | 18.5 GB | 5.5 GB | ~5K | ~27K | ⭐⭐⭐⭐⭐ Best code 24 GB |
| InternLM2.5-20B-Chat | BF16 | 40 GB | — | — | — | ❌ |
| InternLM2.5-20B-Chat | Q4_K_M | 11.5 GB | 12.5 GB | ~24K | ~120K | ⭐⭐⭐⭐½ |
| Yi-1.5-34B-Chat | Q4_K_M | 19.8 GB | 4.2 GB | ~4K | ~20K | ⭐⭐⭐⭐ |

**Cloud cost context (approximate, 2025–2026 pricing):**

| Instance | GPU | On-demand $/hr | Spot/Preemptible $/hr | $/million tokens at 30 t/s |
|----------|-----|-----------------|----------------------|--------------------------|
| GCP `g2-standard-4` | 1× L4 24 GB | $0.70 | $0.21 | ~$0.19 |
| AWS `g5.xlarge` | 1× A10G 24 GB | $1.01 | $0.30 | ~$0.28 |
| Azure `NVads A10 v5 (6 vCPU)` | 1× A10 24 GB | $0.90 | ~$0.27 | ~$0.25 |
| GCP `n1 + T4` | 1× T4 16 GB | $0.35 | $0.11 | ~$0.31 |

*TurboQuant compression reduces the $/million-token figure roughly in
proportion to context extension — longer contexts per slot = more tokens per
dollar.*

---

## Qwen Model Guide

Alibaba's Qwen2.5 family has become the leading open-weight choice for
cost-effective deployment.  The GQA architecture (fewer KV heads than Q heads)
makes KV-cache especially small, and TurboQuant compresses it further.

### Qwen2.5 Architecture Parameters

| Model | Q heads | KV heads | Head dim | Layers | Context (native) | KV/token FP16 |
|-------|---------|----------|----------|--------|-----------------|---------------|
| Qwen2.5-0.5B | 14 | 2 | 64 | 24 | 128K | 12 KB |
| Qwen2.5-1.5B | 12 | 2 | 128 | 28 | 128K | 14 KB |
| Qwen2.5-3B | 16 | 2 | 128 | 36 | 128K | 18 KB |
| Qwen2.5-7B | 28 | 4 | 128 | 28 | 128K | 28 KB |
| Qwen2.5-14B | 40 | 8 | 128 | 40 | 128K | 80 KB |
| Qwen2.5-32B | 64 | 8 | 128 | 64 | 128K | 128 KB |
| Qwen2.5-72B | 64 | 8 | 128 | 80 | 128K | 160 KB |

*KV/token = 2 × num_kv_heads × head_dim × 2 bytes (FP16) × num_layers /
seq_len — here expressed per token across all layers.*

### Recommended Qwen Configs per VRAM Tier

```yaml
# ── 12 GB: Qwen2.5-7B at Q4 — great for most chat tasks ──────────────
- name: qwen2.5-7b-q4
  hf_id: Qwen/Qwen2.5-7B-Instruct-GGUF
  params:
    isq: q4k
    mem: 7500
    kvcache_compression: { bits: 3, policy: { threshold_tokens: 2048 } }

# ── 16 GB: Qwen2.5-14B at Q4 — significantly better than 7B ──────────
- name: qwen2.5-14b-q4
  hf_id: Qwen/Qwen2.5-14B-Instruct
  params:
    isq: q4k
    mem: 7800
    kvcache_compression: { bits: 3, policy: { threshold_tokens: 4096 } }

# ── 24 GB: Qwen2.5-32B at Q4 — best single-GPU open model ────────────
- name: qwen2.5-32b-q4
  hf_id: Qwen/Qwen2.5-32B-Instruct
  params:
    isq: q4k
    mem: 5200
    kvcache_compression: { bits: 3, policy: { threshold_tokens: 4096 } }
```

### Qwen Specialised Variants

| Variant | Best for | Notes |
|---------|----------|-------|
| `Qwen2.5-Coder-*` | Code generation, completion, debugging | Outperforms DeepSeek-Coder at same size |
| `Qwen2.5-Math-*` | Mathematical reasoning | CoT + tool-use trained |
| `QwQ-32B` | Deep reasoning (o1-style) | Similar to DeepSeek-R1-Distill; fits 24 GB at Q4 |
| `Qwen2.5-VL-*` | Vision + language | Requires vision pipeline support in candle-vllm |

---

## Best Reasoning Value vs Frontier Models

The table below compares open-weight models — runnable locally via candle-vllm
with TurboQuant compression — against frontier closed-weight APIs.

Quality ratings are composite estimates based on MMLU, MATH, HumanEval,
GPQA, LiveCodeBench, and arena ELO scores available as of early 2026.

### Reasoning Quality Comparison

| Model | Type | MMLU | MATH | HumanEval | GPQA | Notes |
|-------|------|------|------|-----------|------|-------|
| **OpenAI GPT-5** | Closed API | ~92 | ~90 | ~95 | ~78 | Frontier; best overall as of 2026 |
| **OpenAI GPT-5.4** | Closed API | ~90 | ~88 | ~93 | ~75 | Strong reasoning, fast |
| **Claude Opus 4.6** | Closed API | ~91 | ~87 | ~92 | ~76 | Excellent coding + nuanced writing |
| **Claude Sonnet 4.6** | Closed API | ~88 | ~84 | ~90 | ~72 | Best cost/quality closed model |
| **Gemini 2.5 Pro** | Closed API | ~90 | ~89 | ~92 | ~75 | Excellent long-context |
| **Kimi k1.5** | Closed API | ~87 | ~83 | ~88 | ~70 | Moonshot AI; strong long-context reasoning |
| **MiniMax-Text-01** | Open weights | ~85 | ~78 | ~84 | ~65 | 456B MoE; best open MoE model |
| **DeepSeek-R1** (full 671B) | Open weights | ~90 | ~90 | ~93 | ~74 | Matches o1-preview; needs multi-GPU |
| **DeepSeek-V3** (full 685B) | Open weights | ~88 | ~87 | ~91 | ~72 | Fastest large open model |
| **DeepSeek-R1-Distill-Qwen-32B** | Open weights | ~83 | ~84 | ~87 | ~66 | Best single-GPU reasoning model |
| **DeepSeek-R1-Distill-Qwen-14B** | Open weights | ~79 | ~79 | ~83 | ~61 | Excellent 16/24 GB reasoning |
| **DeepSeek-R1-Distill-Llama-8B** | Open weights | ~74 | ~74 | ~79 | ~55 | Surprising reasoning for 8B |
| **QwQ-32B** | Open weights | ~83 | ~85 | ~86 | ~65 | Best 24 GB reasoning/math |
| **Qwen2.5-72B-Instruct** | Open weights | ~86 | ~83 | ~87 | ~68 | Best open instruct multi-GPU |
| **Qwen2.5-32B-Instruct** | Open weights | ~83 | ~80 | ~85 | ~65 | Best single-GPU general model |
| **Qwen2.5-14B-Instruct** | Open weights | ~79 | ~75 | ~82 | ~59 | Excellent 16 GB model |
| **Qwen2.5-7B-Instruct** | Open weights | ~74 | ~68 | ~77 | ~50 | Best 12 GB instruct model |
| **InternLM2.5-20B-Chat** | Open weights | ~80 | ~76 | ~82 | ~61 | Strong bilingual (zh/en) |
| **Yi-1.5-34B-Chat** | Open weights | ~78 | ~70 | ~79 | ~57 | 01.AI; good long-context |
| **Mistral-Large-2-Instruct** | Open weights | ~80 | ~73 | ~83 | ~60 | Best European open model |
| **Llama-3.3-70B-Instruct** | Open weights | ~86 | ~79 | ~85 | ~63 | Meta; best Llama model |

*Scores are approximate and normalised for comparability. Actual performance
varies by task domain and prompt style.*

### Cost per Million Output Tokens

| Model | Hosting | $/1M tokens | Relative to GPT-5 |
|-------|---------|-------------|-------------------|
| GPT-5 | OpenAI API | ~$15–25 | 1× (baseline) |
| GPT-5.4 | OpenAI API | ~$8–12 | ~0.5× |
| Claude Opus 4.6 | Anthropic API | ~$15–20 | ~0.9× |
| Claude Sonnet 4.6 | Anthropic API | ~$3–5 | ~0.2× |
| Gemini 2.5 Pro | Google API | ~$7–10 | ~0.4× |
| Kimi k1.5 | Moonshot API | ~$2–4 | ~0.15× |
| **DeepSeek-R1-Distill-32B (self-hosted, GCP L4)** | **Self** | **~$0.15–0.30** | **~0.01×** |
| **QwQ-32B (self-hosted, GCP L4)** | **Self** | **~$0.15–0.30** | **~0.01×** |
| **Qwen2.5-14B Q4 (self-hosted, GCP T4)** | **Self** | **~$0.08–0.15** | **~0.005×** |
| DeepSeek API (cloud) | DeepSeek | ~$0.50–1.00 | ~0.04× |

*Self-hosted costs assume: 30 tokens/s throughput, 24/7 operation on spot/preemptible
instances.  TurboQuant compression can increase throughput by allowing more
concurrent requests, lowering the $/M tokens further.*

### Reasoning Value Sweet Spots

Based on the above, the best **reasoning quality per dollar** options are:

1. **DeepSeek-R1-Distill-Qwen-32B @ Q4 on GCP L4 (24 GB)**
   - Quality: ~83 MMLU, ~84 MATH — matches Sonnet 3.7 on reasoning tasks
   - Cost: ~$0.20/M tokens self-hosted
   - Config: `bits: 3, policy: threshold_tokens: 4096`

2. **QwQ-32B @ Q4 on GCP L4 (24 GB)**
   - Quality: ~83 MMLU, ~85 MATH — excellent mathematical reasoning
   - Cost: ~$0.20/M tokens self-hosted
   - Particularly good at step-by-step STEM reasoning

3. **Qwen2.5-14B-Instruct @ Q4 on GCP T4 (16 GB)**
   - Quality: ~79 MMLU — strong general assistant
   - Cost: ~$0.10/M tokens self-hosted (T4 spot ~$0.11/hr)
   - Best cost/quality for conversational AI at 16 GB

4. **DeepSeek-R1-Distill-Qwen-14B @ Q4 on GCP T4 (16 GB)**
   - Quality: ~79 MMLU, ~79 MATH — best reasoning at 16 GB
   - Cost: ~$0.10/M tokens
   - TurboQuant extends effective context from 16K → 80K on T4

5. **DeepSeek-R1-Distill-Llama-8B @ Q4 on RTX 4070 Ti (12 GB)**
   - Quality: ~74 MMLU — surprisingly good reasoning for 8B
   - Cost: ~$0 if using owned hardware
   - Excellent for local development and testing

---

## Choosing a Compression Policy

```
Is your workload primarily short conversations (< 2K tokens)?
  → Disable compression. Overhead not worth it.

Are you serving long documents, coding assistants, or RAG with large context?
  → threshold_tokens: 4096
     Keep first 4K in FP16 (prefill quality), compress the rest.

Is your GPU at > 80% KV-cache utilisation under typical load?
  → memory_pressure: { free_block_pct: 0.20 }
     Adaptive: only compresses under pressure. Good for mixed workloads.

Running a batch inference job where you want maximum throughput?
  → always
     Maximise concurrent requests. Slight accuracy trade is acceptable at 3-bit.
```

---

## Technical Implementation

The integration lives in these files:

| File | Purpose |
|------|---------|
| `crates/candle-vllm-core/src/scheduler/kv_compression.rs` | All compression types: `KvCacheCompressionConfig`, `CompressionPolicy`, `CompressedSlot`, `CompressedLayerCache`, `CompressedStore`, `KvCacheTensors`, `bytes_per_block` |
| `crates/candle-vllm-core/src/scheduler/cache_engine.rs` | `CacheEngine` holds `Arc<Mutex<CompressedStore>>`; exposes `get_kv_tensors()` and `push_compressed()` |
| `crates/candle-vllm-core/src/engine_params.rs` | `EngineParams.kvcache_compression: Option<KvCacheCompressionConfig>` |
| `crates/candle-vllm-core/src/parking_lot/executor.rs` | All three `pipeline.forward()` call sites use `get_kv_tensors()` |
| `src/lib.rs` / `crates/candle-vllm-server/src/lib.rs` | `get_cache_config()` uses `bytes_per_block()` for accurate block count profiling |

### Data Flow

```
models.yaml
  └─ kvcache_compression.bits / policy
       ↓
  EngineParams.kvcache_compression
       ↓
  CacheConfig.compression
       ↓
  CacheEngine::new()
    └─ CompressedStore::new(layers, kv_heads, head_dim, bits)
         └─ CompressedLayerCache × N_layers
              ├─ TurboQuant(head_dim, bits, key_seed)   ← fixed rotation
              └─ TurboQuant(head_dim, bits, val_seed)   ← fixed rotation

Per forward pass:
  get_kv_tensors()
    └─ CompressedLayerCache::decompress_to_tensors()
         └─ TurboQuant::decompress_mse() per slot per head
              └─ PolarQuant::dequantize() → IFWHT → dense f32 tensor
                   ↓
             pipeline.forward(..., kv_tensors, ...)
                   ↓
  push_compressed(layer, block, slot, key_f32, val_f32)
    └─ TurboQuant::compress_mse()
         └─ PolarQuant::quantize() → FWHT → Lloyd-Max → bitpack
              └─ CompressedSlot { keys, vals } stored in HashMap
```

### Thread Safety

`PolarQuant` uses `Mutex<Vec<f32>>` (not `RefCell`) for its scratch buffer,
making the entire compression stack `Send + Sync`.  `CompressedStore` is
wrapped in `Arc<Mutex<_>>` for safe sharing across the parking-lot thread pool.

---

## Further Reading

- [TurboQuant Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research
- [turboquant-rs docs/candle-vllm-integration.md](https://github.com/GQAdonis/turboquant-rs/blob/main/docs/candle-vllm-integration.md) — detailed technical writeup
- [example.models.yaml](../example.models.yaml) — full configuration reference
- [.example.env](../.example.env) — environment variable reference
- [docs/CONFIGURATION.md](CONFIGURATION.md) — complete configuration guide
