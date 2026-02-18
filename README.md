<p align="center">
    <img src="./res/candle_vllm_logo.png" alt="candle vLLM" width=55%/>
</p>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README-CN.md">简体中文</a> |
</p>

Efficient, easy-to-use platform for inference and serving local LLMs including an OpenAI-compatible API server. **This is the Prometheus fork** of [candle-vllm](https://github.com/EricLBuehler/candle-vllm), re-architected as a modular, library-first Rust workspace with production-grade scheduling, MCP integration, and multi-backend request queuing.

## What's Different from Upstream

This fork (`feature/god-mode`) restructures the upstream monolithic binary into a **four-crate workspace** and adds several production-oriented features:

| Feature | This Fork | Upstream |
|---------|-----------|----------|
| **Architecture** | 4-crate workspace (`core`, `openai`, `server`, `responses`) with public library APIs | Monolithic single-crate binary |
| **Scheduler** | Resource-aware parking-lot scheduler with backpressure and dedicated OS thread pool | Direct tokio task spawning |
| **MCP Integration** | Full MCP protocol client (stdio + HTTP), multi-server manager, agentic tool-execution loops | None |
| **Model Config** | Declarative `models.yaml` with aliases, per-model overrides, vision config, prompt cache settings | CLI flags only |
| **Request Queue** | Pluggable queue (memory / postgres / sqlite / surrealdb) + mailbox for async result retrieval | None |
| **Vision** | Vision proxy preprocessing mode with pluggable image description backends | None |
| **Prompt Caching** | Two-tier: block-level prefix cache + persistent KV tensor cache (memory / sled / redis) | Block-level prefix cache only |
| **Embeddings** | `/v1/embeddings` endpoint with mean/last pooling, Float/Base64 encoding | Basic embedding support |
| **Web UI** | Embedded UI with model switching, context caching indicators, reasoning block rendering | Basic embedded UI |
| **Webhooks** | Per-request webhook delivery on completion (`x-webhook-url` header) | None |
| **Examples** | Agent framework, AI gateway, Tauri desktop app, library usage examples | Python scripts only |

## Workspace Architecture

```
candle-vllm/                         # Binary entry point (v0.5.1)
├── src/
│   ├── main.rs                      # Thin entry: calls candle_vllm_server::run()
│   ├── lib.rs                       # Re-exports all crates for backward compat
│   ├── api.rs                       # Standalone Engine/EngineBuilder API
│   ├── mcp/                         # Hand-rolled MCP protocol (client, server, manager)
│   └── tools/                       # Tool definitions, parsers, schema builder
├── crates/
│   ├── candle-vllm-core/            # Core inference engine, scheduler, all model architectures
│   ├── candle-vllm-openai/          # OpenAI-compatible adapter, conversation, tool calling
│   ├── candle-vllm-server/          # Axum HTTP server, routes, config, queue/mailbox/webhook
│   └── candle-vllm-responses/       # MCP orchestration, agentic multi-turn sessions
├── kernels/                         # CUDA paged attention kernels
├── metal-kernels/                   # Metal paged attention kernels (Apple Silicon)
└── examples/
    ├── simple_gen.rs                # Library usage: text generation
    ├── simple_embed.rs              # Library usage: embeddings
    ├── agent_framework/             # Full agent with tool-calling loop
    ├── ai_gateway/                  # Multi-model routing gateway
    └── tauri_app/                   # Desktop app integration
```

## Features

- OpenAI-compatible API server provided for serving LLMs
- **Library-first design** — use `candle-vllm-core`, `candle-vllm-openai`, or `candle-vllm-responses` as Rust crate dependencies
- **Resource-aware parking-lot scheduler** with configurable worker pool, backpressure, and capacity limits
- Streaming support in generation, **including incremental tool-call deltas**
- Efficient management of key-value cache with PagedAttention
- **Prompt caching** with multiple backends (memory, sled, redis) for KV tensor persistence
- Continuous batching (batched decoding for incoming requests over time)
- `In-situ` quantization (and `In-situ` marlin format conversion)
- `GPTQ/Marlin` format quantization (4-bit)
- Support `Mac/Metal` devices
- Support `Multi-GPU` inference (both `multi-process` and `multi-threaded` mode)
- Support `Multi-node` inference with MPI runner
- Support Chunked Prefilling (default chunk size 8K)
- Support CUDA Graph
- **Tool Calling** support (OpenAI-compatible function calling API) with automatic MCP tool injection from `mcp.json`
- **MCP Integration** — multi-server tool orchestration with agentic conversation loops
- **Declarative model configuration** via `models.yaml` with aliases, per-model parameters, and parking-lot overrides
- **Vision proxy** for multimodal requests (image description preprocessing)
- **Embeddings API** (`/v1/embeddings`) with mean/last pooling modes
- **Request queue and mailbox** — pluggable backends for async request handling and result retrieval
- **Webhook delivery** — per-request `x-webhook-url` for completion notifications
- Support for Mistral 3 / Ministral models (BF16/FP16 variants with nested rope parameters)
- Support Prefix Caching
- Support Block-wise FP8 Models (SM90+, Qwen3 Series)
- **Built-in ChatGPT-like Web UI** with model switching, reasoning block rendering, and context caching indicators

## Supported Models

Currently, candle-vllm supports chat serving for the following model architectures.

<details>
  <summary>Show supported model architectures</summary>

  | Model ID | Model Type | Supported | Speed (A100, `BF16`) | Throughput (`BF16`, `bs=16`) | Quantized (A100, `Q4K` or `Marlin`) | Throughput (`GTPQ/Marlin`, `bs=16`) |
  |--|--|--|--|--|--|--|
  | #1 | **LLAMA** |✅|65 tks/s (8B) | 553 tks/s (8B) | 75 tks/s (8B), 115 tks/s (8B, **Marlin**) |968 tks/s (8B)|
  | #2 | **Mistral/Ministral** |✅|70 tks/s (7B)| 585 tks/s (7B) | 96 tks/s (7B), 115 tks/s (7B, **Marlin**) |981 tks/s (7B)|
  | #3 | **Phi** (Phi-3, Phi-4) |✅|107 tks/s (3.8B)| 744 tks/s (3.8B)|135 tks/s (3.8B)|TBD|
  | #4 | **QWen2/Qwen3** |✅|81 tks/s (8B)|831 tks/s (8B) |-|TBD|
  | #5 | **Yi** |✅|75 tks/s (6B)| 566 tks/s (6B) | 105 tks/s (6B)|TBD|
  | #6 | **StableLM** |✅|99 tks/s (3B)|TBD|-|TBD|
  | #7 | **Gemma-2/Gemma-3** |✅|60 tks/s (9B)|TBD |73 tks/s (9B, **Marlin**) |587 tks/s (9B)|
  | #8 | **DeepSeek-R1-Distill-QWen** |✅|48 tks (14B)|TBD|62 tks (14B)|TBD|
  | #9 | **DeepSeek-R1-Distill-LLaMa** |✅|65 tks (8B)|TBD|108 tks (8B)|TBD|
  | #10 | **DeepSeek V2/V3/R1** |✅|TBD|TBD|~20 tks **(AWQ 671B, tp=8, offloading)**|TBD|
  | #11 | **QwQ-32B** |✅|30 tks/s **(32B, tp=2)**|TBD |36 tks/s **(32B, Q4K, GGUF)**|TBD|
  | #12 | **GLM4** |✅|55 tks/s **(9B)**|TBD |92 tks/s **(9B, Q4K, GGUF)**|TBD|
  | #13 | **QWen2 MoE** |✅|TBD|TBD |65 tks/s (14B, Q4K)|TBD|
  | #14 | **QWen3 MoE** |✅|TBD|TBD |76 tks/s **(32B, Q4K)** |TBD|
</details>

### Mistral 3 / Ministral Model Notes

- Supports **Mistral 3** and **Ministral** models with nested `rope_parameters` configuration format.
- **Important**: Use **BF16 or FP16** model variants. FP8 (F8_E4M3) quantized models are not currently supported.
- Recommended models:
  - `mistralai/Mistral-7B-Instruct-v0.3` (BF16)
  - `mistralai/Ministral-8B-Instruct-2410` (BF16)

### Demo Video

<details>
  <summary>Show Demo Video</summary>

  Chat demo on **GPU** (A100, BF16, QWen3-8B Reasoning Model)
  <img src="res/Qwen3-8B-Reasoning-A100.gif" width="85%" height="85%" >

  Chat demo on **Apple Silicon** (M4 with 16GB unified memory, Q2K, QWen3-8B)
  <img src="res/Qwen3-8B-Apple-M4.gif" width="85%" height="85%" >
</details>

## Using as a Library

candle-vllm can be used as a Rust library. Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
# Core inference engine
candle-vllm-core = { git = "https://github.com/GQAdonis/candle-vllm.git", branch = "feature/god-mode" }

# OpenAI-compatible adapter (conversation management, tool calling)
candle-vllm-openai = { git = "https://github.com/GQAdonis/candle-vllm.git", branch = "feature/god-mode" }

# Multi-turn agent conversations with MCP tool orchestration
candle-vllm-responses = { git = "https://github.com/GQAdonis/candle-vllm.git", branch = "feature/god-mode" }
```

### Quick Example — Text Generation

```rust
use candle_vllm::api::{EngineBuilder, ModelRepo};

fn main() -> anyhow::Result<()> {
    let engine = EngineBuilder::from(ModelRepo::from_hf_id("Qwen/Qwen3-0.6B"))
        .build()?;

    let output = engine.generate("Explain how to best learn Rust.")?;
    println!("{}", output);
    Ok(())
}
```

### Quick Example — Embeddings

```rust
use candle_vllm::api::{EngineBuilder, ModelRepo};

fn main() -> anyhow::Result<()> {
    let engine = EngineBuilder::from(ModelRepo::from_path("/path/to/model/"))
        .build()?;

    let embeddings = engine.embed("Hello, world!")?;
    println!("Embedding dims: {}", embeddings.len());
    Ok(())
}
```

### Quick Example — Agent Framework

See `examples/agent_framework/` for a complete agent with tool-calling loops, or `examples/ai_gateway/` for a multi-model routing gateway.

For detailed API documentation, see [Library API docs](docs/rust_crate.md).

## Installation

### Clone

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  # Rust 1.92.0+ required
sudo apt install libssl-dev pkg-config -y
git clone git@github.com:GQAdonis/candle-vllm.git
cd candle-vllm
```

### CUDA (CUDA 11+, 12+, 13.0)

> **Option 1: Docker**

```bash
# Standard build (pass hardware arch and cuda version)
./build_docker.sh "cuda,nccl,graph,flash-attn,flash-decoding" sm_80 12.9.0

# +cutlass feature for optimized fp8 models (Qwen3 series, sm90+) with CUDA 13
./build_docker.sh "cuda,nccl,graph,flash-attn,flash-decoding,cutlass" sm_90 13.0.0

# Use Rust crate China Mirror
./build_docker.sh "cuda,nccl,graph,flash-attn,flash-decoding" sm_80 12.9.0 1
```

> **Option 2: Manual Installation**

```shell
sudo apt update
sudo apt install git libssl-dev pkg-config curl -y
sudo apt install -y cuda-toolkit-12-9  # optional
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
export PATH=$PATH:/usr/local/cuda/bin/
```

Install for single node:

```shell
cargo install --release --features cuda,nccl --path .

# +CUDA Graph
cargo install --features cuda,nccl,graph --path .

# +Flash attention (sm_80+)
cargo install --features cuda,nccl,graph,flash-attn --path .

# +Flash attention for prefill and decoding (sm_80+)
cargo install --features cuda,nccl,graph,flash-attn,flash-decoding --path .
```

Install for multi-node (MPI):

```shell
sudo apt install libopenmpi-dev openmpi-bin clang libclang-dev -y
cargo install --features cuda,nccl,mpi --path .
```

### Mac/Metal (single-node only)

Install [Xcode command line tools](https://mac.install.guide/commandlinetools/), then:

```shell
cargo install --features metal --path .
```

## Configuration

### models.yaml

This fork uses a declarative `models.yaml` for model configuration. The server searches `~/.candle-vllm/models.yaml`, then `./models.yaml`. See `example.models.yaml` for the full schema.

```yaml
default_model: mistral-7b
idle_unload_secs: 300

parking_lot:
  pool:
    worker_threads: 8
  limits:
    max_units: 4096
    max_queue_depth: 100
    timeout_secs: 300

models:
  - name: mistral-7b
    hf_id: mistralai/Mistral-7B-Instruct-v0.3
    params:
      dtype: bf16
      mem: 8192
      device_ids: [0]
      temperature: 0.7
      top_p: 0.9
    parking_lot:
      limits:
        max_queue_depth: 50

  - name: qwen3-8b-gguf
    hf_id: unsloth/Qwen3-8B-GGUF
    weight_file: Qwen3-8B-Q4_K_M.gguf
    params:
      mem: 4096
```

### Environment Variables

Copy `.example.env` to `.env` or export variables directly. Key variables:

| Variable | Description |
|----------|-------------|
| `CANDLE_VLLM_MODELS_CONFIG` | Path to `models.yaml` |
| `CANDLE_VLLM_MCP_CONFIG` | Path to `mcp.json` |
| `CANDLE_VLLM_PROMPT_CACHE_ENABLED` | Enable prompt caching (`true`/`false`) |
| `CANDLE_VLLM_PROMPT_CACHE_BACKEND` | `memory` / `sled` / `redis` |
| `RUST_LOG` | Log level (supports per-module levels) |
| `HF_TOKEN` | HuggingFace authentication token |

See [Configuration docs](docs/CONFIGURATION.md) for details.

### MCP Configuration

Create `~/.candle-vllm/mcp.json` (Claude Desktop-compatible format):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "mcp-server-filesystem",
      "args": ["/home/user/documents"],
      "env": {}
    },
    "web-search": {
      "type": "http",
      "url": "http://localhost:3000/search"
    }
  }
}
```

MCP tools are automatically injected into chat requests as OpenAI-format `tools`. Set `"tools": []` in a request to opt out.

## Running the Server

### Command Structure

```
[ENV_PARAM] cargo run [BUILD_PARAM] -- [PROGRAM_PARAM] [MODEL] [CACHE] [UI]
```

<details open>
  <summary>Show parameter reference</summary>

  | Parameter | Description |
  |-----------|-------------|
  | `--p <PORT>` | Server port (default: 2000) |
  | `--d <DEVICE_IDS>` | Comma-separated GPU device IDs |
  | `--w <WEIGHT_PATH>` | Local safetensors folder |
  | `--f <WEIGHT_FILE>` | GGUF file path |
  | `--m <MODEL_ID>` | HuggingFace model ID or `models.yaml` alias |
  | `--isq <FORMAT>` | In-situ quantization: q4_0, q4_1, q5_0, q5_1, q8_0, q2k, q3k, q4k, q5k, q6k |
  | `--mem <MB>` | KV cache GPU memory in MB (default: 4096) |
  | `--dtype <TYPE>` | Data type: bf16, fp16, fp32 |
  | `--prefill-chunk-size <N>` | Chunked prefill size (default: 8K, 0 to disable) |
  | `--fp8-kvcache` | Enable FP8 KV cache |
  | `--prefix-cache` | Enable prefix cache reuse |
  | `--prompt-cache` | Enable prompt caching |
  | `--prompt-cache-backend <BE>` | memory / sled / redis |
  | `--ui-server` | Enable built-in ChatGPT-like Web UI |
  | `--log` | Enable debug logging |
  | `--multithread` | Multi-GPU threaded mode (instead of multi-process) |
  | `--frequency-penalty <F>` | Repetition penalty (-2.0 to 2.0) |
  | `--presence-penalty <F>` | Presence penalty (-2.0 to 2.0) |
  | `--temperature <F>` | Sampling temperature |
  | `--top-p <F>` | Nucleus sampling threshold |
  | `--top-k <N>` | Top-k sampling |
  | `--models-config <PATH>` | Path to models.yaml |
  | `--enable-vision` | Enable vision model support |
  | `--request-timeout <SECS>` | Per-request timeout (default: 30) |
</details>

### Examples

**Using models.yaml alias (recommended):**

```shell
candle-vllm --p 2000 --ui-server
# Loads default_model from models.yaml
```

**Uncompressed models:**

```shell
# Local path with ISQ quantization
candle-vllm --p 2000 --d 0,1 --w /home/Qwen3-30B-A3B-Instruct-2507/ --isq q4k --ui-server

# HuggingFace model ID
candle-vllm --m deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --ui-server --prefix-cache

# FP8 model (requires cutlass feature)
candle-vllm --w Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/ --ui-server
```

**GGUF models:**

```shell
# Local file
candle-vllm --f /home/data/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server

# HuggingFace GGUF
candle-vllm --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server
```

**GGUF on Apple Silicon:**

```shell
candle-vllm --m Qwen/QwQ-32B-GGUF --f qwq-32b-q4_k_m.gguf --ui-server
```

**Multi-GPU:**

```shell
# Multi-process mode (default, 2/4/8 GPUs)
candle-vllm --d 0,1 --w /home/QwQ-32B/

# Multi-threaded mode (for debugging)
candle-vllm --multithread --d 0,1 --w /home/QwQ-32B/
```

**Docker:**

```shell
docker run --rm -it --gpus all --network host -v /home:/home -v /data:/data candle-vllm:latest bash
candle-vllm --p 8000 --d 0,1 --w /home/Qwen3-30B-A3B-Instruct-2507/ --ui-server
```

<details>
  <summary>Show advanced deployment options (GPTQ/Marlin, AWQ, DeepSeek-R1 671B, Multi-node MPI, NUMA)</summary>

  **Marlin-compatible GPTQ models (4-bit, 128-group, desc_act=False):**

  ```shell
  candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
  candle-vllm --m thesven/Llama-3-8B-GPTQ-4bit
  ```

  **Convert uncompressed to marlin format:**

  ```shell
  python3 examples/convert_marlin.py --src /home/DeepSeek-R1-Distill-Qwen-14B/ --dst /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
  ```

  **Convert AWQ to Marlin-compatible format:**

  ```shell
  python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
  candle-vllm --d 0 --w /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/
  ```

  **DeepSeek-R1 (671B) with CPU offloading:**

  ```shell
  python3 examples/convert_awq_marlin.py --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Marlin/
  candle-vllm --log --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin/ --num-experts-offload-per-rank 15
  ```

  **Multi-node MPI deployment:**

  ```shell
  sudo apt install libopenmpi-dev openmpi-bin clang libclang-dev -y
  cargo install --features cuda,nccl,mpi --path .
  # hostfile example (two nodes, 8 GPUs each):
  # 192.168.1.100 slots=8
  # 192.168.1.101 slots=8
  sudo mpirun -np 16 -x RUST_LOG=info -hostfile ./hostfile --allow-run-as-root \
    -bind-to none -map-by slot \
    --mca plm_rsh_args "-p 22" --mca btl_tcp_if_include %NET_INTERFACE% \
    candle-vllm --log --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin/
  ```

  **NUMA binding:**

  ```shell
  MAP_NUMA_NODE=0,0,0,0,1,1,1,1 numactl --cpunodebind=0 --membind=0 \
    candle-vllm --d 0,1,2,3,4,5,6,7 --w /home/data/DeepSeek-V2-Chat-AWQ-Marlin
  ```
</details>

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion (streaming + non-streaming) |
| `POST` | `/v1/embeddings` | Text embedding generation |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/status` | Active model status |
| `POST` | `/v1/models/select` | Switch active model |
| `GET` | `/v1/mcp/tools` | List active MCP tools |
| `GET` | `/v1/queues` | List queued requests |
| `GET` | `/v1/queues/{model}` | Queued requests for a model |
| `GET` | `/v1/mailbox` | List completed async results |
| `GET` | `/v1/mailbox/{id}` | Retrieve a specific result |
| `DELETE` | `/v1/mailbox/{id}` | Delete a mailbox record |

### Per-Request Headers

| Header | Description |
|--------|-------------|
| `x-webhook-url` | URL to POST completion result to |
| `x-webhook-mode` | `Always` / `Never` / `OnDisconnect` |
| `x-webhook-bearer` | Bearer token for webhook authentication |
| `x-conversation-id` | Conversation tracking ID |
| `x-resource-id` | Resource tracking ID |

### Chat Completion

```shell
curl -X POST "http://127.0.0.1:2000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
        "model": "mistral-7b",
        "messages": [
            {"role": "user", "content": "Explain how to best learn Rust."}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": true
    }'
```

### Embeddings

```shell
curl -X POST "http://127.0.0.1:2000/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistral-7b",
        "input": "Hello, world!",
        "encoding_format": "float"
    }'
```

### Tool Calling

```shell
curl http://localhost:2000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "mistral-7b",
    "messages": [
      {"role": "user", "content": "What is the weather like in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              }
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_tokens": 256
  }'
```

**Tool Choice Options:**

| Option | Description |
|--------|-------------|
| `"auto"` | Model decides whether to call a tool (default) |
| `"none"` | Model won't call any tools |
| `"required"` | Model must call at least one tool |
| `{"type": "function", "function": {"name": "..."}}` | Force a specific tool |

**Supported Tool Call Formats** (auto-detected):

| Format | Models | Pattern |
|--------|--------|---------|
| Mistral/Ministral | Mistral-7B-Instruct-v0.3, Ministral-8B | `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]` |
| Llama | Llama-3.x-Instruct | `<function=func_name>{"arg": "value"}</function>` |
| Qwen | Qwen2/Qwen3 | `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` |
| Generic JSON | Various | `{"name": "...", "arguments": {...}}` |

**Response with Tool Call:**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "arguments": "{\"location\": \"San Francisco, CA\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Python Client

```python
import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:2000/v1/"

completion = openai.chat.completions.create(
    model="mistral-7b",
    messages=[{"role": "user", "content": "Explain how to best learn Rust."}],
    max_tokens=512,
)
print(completion.choices[0].message.content)
```

**Batch benchmark:**

```shell
python3 examples/benchmark.py --batch 16 --max_tokens 1024
```

## Parking-Lot Scheduler

The parking-lot scheduler is this fork's core scheduling innovation. It replaces simple tokio task spawning with a resource-aware thread pool that provides backpressure and capacity management.

### How It Works

1. **Resource Accounting** — Each request's token count is mapped to KV-cache block units. The scheduler tracks total GPU memory capacity.
2. **Worker Pool** — Dedicated OS threads (default: `num_cpus`) process inference jobs. No tokio runtime blocking.
3. **Backpressure** — When GPU capacity is exhausted, excess requests queue up to `max_queue_depth`. Beyond that, clients receive HTTP 503.
4. **Streaming** — Streaming requests spawn a side thread per job. Tokens flow through `flume` channels to the HTTP layer via a `StreamingRegistry`.
5. **Panic Safety** — Every job is wrapped in `catch_unwind` to prevent worker thread crashes.

### Configuration

In `models.yaml`:

```yaml
parking_lot:
  pool:
    worker_threads: 8          # Dedicated OS threads (default: num_cpus)
  limits:
    max_units: 4096            # Max KV-cache blocks (null = auto from --mem)
    max_queue_depth: 100       # 503 beyond this depth
    timeout_secs: 300          # Per-request timeout
  queue:
    backend: "memory"          # memory | postgres | sqlite | surrealdb
    persistence: false
  mailbox:
    backend: "memory"          # memory | postgres
    retention_secs: 3600
```

## In-Situ Quantization

<details>
  <summary>Show quantization details</summary>

  Transform default weights (F32/F16/BF16) into any GGML/GGUF format, or 4-bit GPTQ/AWQ weights into marlin format during model loading:

  ```shell
  # GGML quantization
  candle-vllm --p 2000 --w /home/Meta-Llama-3.1-8B-Instruct/ --isq q4k

  # GPTQ model (auto-detects marlin-compatible format)
  candle-vllm --p 2000 --w /home/mistral_7b-int4/
  ```

  Options: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2k`, `q3k`, `q4k`, `q5k`, `q6k`

  Notes:
  - Loading F32/F16/BF16 models into quantized format may take a few minutes
  - Marlin in-situ conversion supports 4-bit GPTQ (`sym=True`, `groupsize=128` or -1, `desc_act=False`) and 4-bit AWQ
  - Marlin format is CUDA-only
</details>

## Development

### Build Commands

```shell
# Mac/Metal (always include metal feature)
cargo build --release --features metal
cargo test --features metal

# CUDA (single or multi-GPU)
cargo build --release --features cuda,nccl

# CUDA with graph optimization
cargo build --release --features cuda,nccl,graph

# CUDA with flash attention (CUDA_ARCH >= 800)
cargo build --release --features cuda,nccl,graph,flash-attn,flash-decoding
```

### Test

```shell
# Full test suite
cargo test --features metal                            # macOS
cargo test --features cuda                             # Linux/CUDA

# Specific packages
cargo test --package candle-vllm-core --lib --features metal
cargo test --package candle-vllm-core --lib --features metal tool_streaming
cargo test --package candle-vllm-core --lib --features metal chunk_collector
```

### Lint and Format

```shell
cargo fmt --all
cargo clippy --all-targets --all-features -D warnings
```

### Debug Logging

```shell
RUST_LOG=debug cargo run --release --features metal -- --log --p 2000 --ui-server
```

## Additional Docs

| Document | Description |
|----------|-------------|
| [Library API](docs/rust_crate.md) | Rust crate usage guide |
| [Configuration](docs/CONFIGURATION.md) | Full configuration reference |
| [Embedding API](docs/embedding.md) | Embedding endpoint guide |
| [MCP & Tool Calling](docs/mcp_tool_calling.md) | MCP integration guide |
| [Prefix Cache](docs/prefix_cache.md) | Prompt caching guide |
| [Goose AI Agent](docs/goose.md) | Integration with Goose |
| [Coding Standards](docs/coding-standards/README.md) | Rust coding standards |
| [Parking-Lot Scheduler](docs/PARKING_LOT_SCHEDULER.md) | Scheduler architecture |
| [GKE Deployment](docs/GKE_GPU_DEPLOYMENT.md) | Kubernetes GPU deployment |
| [Web UI](docs/UI.md) | Built-in chat UI docs |

## Report Issues

If you have problems, please create an [issue](https://github.com/GQAdonis/candle-vllm/issues).

Upstream project: [EricLBuehler/candle-vllm](https://github.com/EricLBuehler/candle-vllm)

## Contributing

Contributions are welcome. Key areas:

- Sampling methods: beam search ([huggingface/candle#1319](https://github.com/huggingface/candle/issues/1319))
- Additional model architectures (from `candle-transformers`)
- Queue backend implementations (postgres, sqlite persistence)
- Vision model native support
- MCP transport improvements

## Resources

- Upstream project: [EricLBuehler/candle-vllm](https://github.com/EricLBuehler/candle-vllm)
- Python implementation: [vllm-project](https://github.com/vllm-project/vllm)
- [vLLM paper](https://arxiv.org/abs/2309.06180)
- [Model Context Protocol](https://modelcontextprotocol.io)
