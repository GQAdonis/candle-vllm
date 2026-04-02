# Model Selection Prompt

Use this structured prompt to guide users through model selection for candle-vllm. Ask the questions in order, then use the mapping table to produce recommendations.

---

## Question 1: Primary Task

What is the primary task for this model?

1. **Chat / Conversation** - Interactive dialogue, customer support, assistants
2. **Code Generation** - Writing, completing, or reviewing code
3. **Reasoning / Analysis** - Complex problem solving, chain-of-thought, math
4. **Vision / Multimodal** - Image understanding, visual Q&A, document analysis
5. **General Purpose** - Mixed workloads, no single dominant task

## Question 2: Language Requirements

What languages does the model need to support?

1. **English only** - Primarily English content
2. **English + Chinese** - Bilingual English/Chinese
3. **Western European** - English, French, German, Spanish, etc.
4. **Broad multilingual** - Many languages including CJK, Arabic, Indic
5. **Specific language(s)** - (ask user to specify)

## Question 3: Latency Tolerance

What is your acceptable latency for first-token response?

1. **Real-time** (<1 second) - Interactive applications, live chat
2. **Interactive** (<5 seconds) - Web applications, API services
3. **Batch / Flexible** (>5 seconds acceptable) - Offline processing, bulk generation

## Question 4: Context Window

How much context do you need the model to handle?

1. **Short** (up to 4K tokens) - Simple queries, short conversations
2. **Medium** (8K-32K tokens) - Documents, longer conversations
3. **Long** (64K-128K tokens) - Large documents, extended reasoning
4. **Very Long** (128K+ tokens) - Book-length content, massive context

---

## Recommendation Mapping

### By Primary Task

| Task | Top Picks (Large, 30B+) | Mid-Range (7-14B) | Compact (1-4B) |
|---|---|---|---|
| Chat | Qwen3-30B-A3B, Llama-3.1-70B | Qwen3-8B, Llama-3.1-8B, Mistral-3-24B | Qwen3-4B, Phi-3-mini |
| Code | DeepSeek-R1-Qwen3-8B, DeepSeek-V3 | Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B | Qwen3-4B |
| Reasoning | DeepSeek-R1, Qwen3-30B-A3B | DeepSeek-R1-0528-Qwen3-8B, Qwen3-8B | Qwen3-4B |
| Vision | Gemma-3-27B-VL, Qwen-VL-72B | Gemma-3-12B-VL, Mistral-3-VL, Qwen-VL-7B | Gemma-3-4B-VL |
| General | Qwen3-30B-A3B, Llama-3.1-70B | Qwen3-8B, Llama-3.1-8B | Phi-3-mini, Qwen3-4B |

### By Language Requirements

| Language Need | Recommended Families |
|---|---|
| English only | Llama 3.x, Mistral 3, Phi |
| English + Chinese | Qwen3, DeepSeek, GLM4 |
| Western European | Mistral 3, Llama 3.x, Gemma-3 |
| Broad multilingual | Qwen3, Gemma-3 |

### By Context Window

| Context Need | Models with Native Support |
|---|---|
| Short (4K) | Any model (all support at least 4K) |
| Medium (8-32K) | Most models default to this range |
| Long (64-128K) | Qwen3 (32-128K), Llama 3.1 (128K), Mistral 3 (128K) |
| Very Long (128K+) | Qwen3-8B (128K), Llama 3.1 (128K), DeepSeek-V3 (128K) |

### By Latency Tolerance and Hardware

| Latency | VRAM < 16GB | VRAM 16-24GB | VRAM 24-48GB | VRAM 48GB+ |
|---|---|---|---|---|
| Real-time | 1-4B + q4k | 7-8B + q4k | 7-14B + fp16 | 30B+ + fp16 |
| Interactive | 4-8B + q4k | 8-14B + q4k | 14-30B + q4k | 70B+ + q4k |
| Batch | 7-8B + q4k | 14B + q4k | 30B + q4k | 70B+ + fp16 |

### Apple Silicon (Metal) Sizing Guide

| Chip | Unified Memory | Recommended Max Model Size |
|---|---|---|
| M1/M2 (8GB) | 8GB | 1-4B quantized (q4k) |
| M1/M2 (16GB) | 16GB | 7-8B quantized (q4k) |
| M1 Pro/Max (32GB) | 32GB | 14B quantized or 8B fp16 |
| M2/M3 Max (64GB) | 64GB | 30B quantized or 14B fp16 |
| M2/M3 Ultra (128GB) | 128GB | 70B quantized or 30B fp16 |
| M4 Max (128GB) | 128GB | 70B quantized or 30B fp16 |

---

## Decision Output Template

After collecting answers, present the recommendation in this format:

```
Based on your requirements:
- Task: [selected task]
- Languages: [selected languages]
- Latency: [selected tolerance]
- Context: [selected window]
- Hardware: [from hardware profile]

Recommended models (ranked):

1. [Model Name] - [why it fits]
   - Size: [parameter count]
   - Est. VRAM: [estimate at recommended quantization]
   - Quantization: [recommended level]
   - Context: [max context window]

2. [Model Name] - [alternative option]
   ...

3. [Model Name] - [budget/compact option]
   ...
```
