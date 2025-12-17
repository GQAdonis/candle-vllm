# Web UI Configuration

This document describes how the built-in ChatGPT-like Web UI (RustChatUI) integrates with candle-vllm and how model selection works.

## Overview

Candle-vllm includes an embedded web UI powered by [RustChatUI](https://github.com/guoqingbao/rustchatui), which provides a modern chat interface with features like:

- Light/Dark mode
- Streaming responses
- Code highlighting
- Chain-of-thought rendering
- Context caching support
- Token usage indicators
- Chat history management

## Enabling the Web UI

Start the server with the `--ui-server` flag:

```bash
# macOS with Metal
cargo run --release --features metal -- --p 2000 --ui-server

# CUDA
cargo run --release --features cuda -- --p 2000 --ui-server
```

The UI will be available at:
- **Web UI**: `http://localhost:1999` (port - 1)
- **API Server**: `http://localhost:2000`

## Default Model Selection

### How It Works

When the web UI starts, it fetches the list of available models from the `/v1/models` endpoint. The server response includes:

1. **Model list** with each model marked as default or not
2. **Sorted order** with the default model appearing first
3. **Top-level default field** indicating the default model ID

Example response:

```json
{
  "object": "list",
  "default": "mistral-3-ministral-3B-reasoning",
  "data": [
    {
      "id": "mistral-3-ministral-3B-reasoning",
      "object": "model",
      "created": 1234567890,
      "owned_by": "owner",
      "permission": [],
      "default": true
    },
    {
      "id": "qwen-7b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "owner",
      "permission": [],
      "default": false
    }
  ]
}
```

### Configuration

The default model is specified in your `models.yaml` configuration file:

```yaml
default_model: "mistral-3-ministral-3B-reasoning"

models:
  - name: mistral-3-ministral-3B-reasoning
    hf_id: mistralai/Ministral-3B-Reasoning-Release
    params:
      dtype: bf16
      max_num_seqs: 8
      
  - name: qwen-7b
    hf_id: Qwen/Qwen2.5-7B-Instruct
    params:
      dtype: bf16
```

### CLI Override

If you specify a model via CLI arguments, it takes precedence:

```bash
# This will use the specified model, overriding models.yaml default
cargo run --release --features metal -- --m Qwen/Qwen2.5-7B-Instruct --ui-server
```

### Behavior

1. **With models.yaml**: The UI automatically selects the model marked as `default`
2. **Without models.yaml**: The UI shows a single "default" model entry
3. **Model not found**: If the default model specified in `models.yaml` doesn't exist in the registry, the server will fail to start with a helpful error message

## Model Switching

The web UI supports dynamic model switching when you have multiple models configured:

1. Click the **Settings** (gear icon) in the sidebar
2. Under **Model Selection**, choose from available models
3. The UI will automatically use the selected model for new conversations

**Note**: Existing conversations continue with their original model. Start a new chat to use the newly selected model.

## Context Caching

When context caching is enabled:

```yaml
models:
  - name: my-model
    hf_id: model/id
    params:
      # ... other params
```

The UI will:
- Display token usage statistics in the footer
- Show session status indicators (Running, Waiting, Cached, Swapped, Finished)
- Poll the `/v1/usage` endpoint to update stats in real-time
- Create new session IDs when you regenerate responses to prevent cache pollution

Enable context caching in the UI Settings panel:
1. Click **Settings** → **Backend Settings**
2. Toggle **Context Caching** ON

## Troubleshooting

### UI Not Loading

If the UI doesn't load:

1. Verify the server is running: `curl http://localhost:2000/v1/models`
2. Check that the UI port (API port - 1) is not in use
3. Look for CORS errors in the browser console

### Wrong Model Selected

If the UI selects the wrong model:

1. Check your `models.yaml` has a `default_model` field
2. Verify the default model name matches a model in your `models` list
3. Restart the server after changing `models.yaml`
4. Clear your browser's localStorage and refresh

### Model Not Found Error

If you get "Model not found" errors:

1. Ensure the model specified in CLI or `models.yaml` exists
2. Check that HuggingFace model IDs are correct
3. For local models, verify the path is correct and readable
4. Check logs for model loading errors

## Advanced Configuration

### Custom UI Port

The UI port is automatically set to `API_PORT - 1`. To use a different configuration:

```rust
// In your custom server implementation
start_ui_server(
    custom_ui_port,           // UI server port
    Some(api_port),          // API server port
    None,                    // Remote server URL (for external APIs)
    None                     // API key (for external APIs)
).await
```

### Remote API Server

You can configure the UI to connect to a remote API server:

1. Click **Settings** → **Backend Settings**
2. Change **Server URL** to your remote API endpoint (e.g., `https://api.example.com/v1`)
3. Enter your **API Key** if required

This is useful for:
- Connecting to cloud-hosted models
- Using external OpenAI-compatible APIs
- Load balancing across multiple servers

## Integration with Model Manager

When using the model manager for dynamic loading/unloading:

- The UI will show the currently active model
- Model switches are queued and processed sequentially
- Requests are held during model switches and processed after completion
- Status indicators show when a model is loading or switching

See [DYNAMIC_MODEL_LOADING.md](DYNAMIC_MODEL_LOADING.md) for more details on model management.

## API Compatibility

The web UI is compatible with any OpenAI-compatible API server. Key endpoints used:

- `GET /v1/models` - List available models and identify default
- `POST /v1/chat/completions` - Send chat messages (streaming or non-streaming)
- `GET /v1/usage?session_id=<id>` - Token usage stats (optional, for context caching)

## References

- [RustChatUI GitHub](https://github.com/guoqingbao/rustchatui)
- [ChatClient Frontend](https://github.com/guoqingbao/chatclient)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)