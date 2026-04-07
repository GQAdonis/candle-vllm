RUST_LOG=info /usr/local/bin/candle-vllm \
  --w "$HOME/models/khazarai-Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF/" \
  --f Qwen3-4B-Thinking-2507.Q4_1.gguf \
  --d 0 \
  --mem 7168 \
  --p 2000 \
  --ui-server \
  --kvcache-compression-bits 3 \
  --kvcache-compression-policy threshold-tokens \
  --kvcache-compression-threshold-tokens 4096
2026-04-07T13:56:47.396980Z  INFO candle_vllm_server::config::parking_lot_merge: 📋 CONFIG: Applying models.yaml global parking_lot settings
2026-04-07T13:56:47.396992Z  INFO candle_vllm_server::config::parking_lot_merge: ✅ CONFIG: Final merged config - workers=4, max_units=Some(4096), queue_depth=100, timeout=300s
2026-04-07T13:56:47.397002Z  INFO candle_vllm_server: Loaded models configuration: vision_enabled=false
2026-04-07T13:56:47.452461Z  INFO candle_vllm_server: CUDA compute capability: 8.9
2026-04-07T13:56:47.452478Z  INFO candle_vllm_core::openai::communicator: Running on single node!
2026-04-07T13:56:47.454240Z  INFO candle_vllm_core::openai::communicator: build data channel for the main process!
2026-04-07T13:56:47.454253Z  WARN candle_vllm_core::openai::communicator: data channel is built!
2026-04-07T13:56:47.454258Z  WARN candle_vllm_core::openai::communicator: All subprocess workers have connected to the main processes!
2026-04-07T13:56:47.454259Z  INFO candle_vllm_core::openai::communicator: local_rank 0, global_rank 0, local_world_size 1, global_world_size 1
2026-04-07T13:56:47.454265Z  WARN candle_vllm_server: subprocess rank 0 started!
2026-04-07T13:56:47.454335Z  INFO candle_vllm_core::openai::communicator: build command channel for the main process!
2026-04-07T13:56:47.454351Z  WARN candle_vllm_core::openai::communicator: command channel is built!
2026-04-07T13:56:47.454359Z  INFO candle_vllm_core::backend::heartbeat: enter heartbeat processing loop (Ok(DaemonManager { daemon_streams: Some([]), main_stream: None, mpi_rank: None, mpi_size: None }))
2026-04-07T13:56:47.541512Z  INFO candle_vllm_core::openai::pipelines::pipeline: Loading quantized model from file /home/gqadonis/models/khazarai-Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF/Qwen3-4B-Thinking-2507.Q4_1.gguf
2026-04-07T13:56:47.699752Z  INFO candle_vllm_core::openai::pipelines::pipeline: Quantized qwen3 model has 36 layers.
2026-04-07T13:56:47.699933Z  INFO candle_vllm_core::openai::communicator: build command channel for the main process!
2026-04-07T13:56:47.699957Z  WARN candle_vllm_core::openai::communicator: command channel is built!
2026-04-07T13:56:50.700674Z  WARN candle_vllm_core::backend::progress: all ranks finished model loading!
2026-04-07T13:56:50.710094Z  WARN candle_vllm_core::openai::pipelines::pipeline: Done loading.
2026-04-07T13:56:50.868087Z  INFO candle_vllm_core::backend::gguf: general.size_label : "4.0B"
2026-04-07T13:56:50.868096Z  INFO candle_vllm_core::backend::gguf: qwen3.attention.head_count : "32"
2026-04-07T13:56:50.868099Z  INFO candle_vllm_core::backend::gguf: general.tags : "unsloth, llama.cpp"
2026-04-07T13:56:50.868101Z  INFO candle_vllm_core::backend::gguf: general.file_type : "3"
2026-04-07T13:56:50.868102Z  INFO candle_vllm_core::backend::gguf: qwen3.attention.value_length : "128"
2026-04-07T13:56:50.868104Z  INFO candle_vllm_core::backend::gguf: general.type : "model"
2026-04-07T13:56:50.868105Z  INFO candle_vllm_core::backend::gguf: qwen3.block_count : "36"
2026-04-07T13:56:50.868107Z  INFO candle_vllm_core::backend::gguf: general.quantized_by : "Unsloth"
2026-04-07T13:56:50.868108Z  INFO candle_vllm_core::backend::gguf: qwen3.attention.key_length : "128"
2026-04-07T13:56:50.868111Z  INFO candle_vllm_core::backend::gguf: qwen3.attention.layer_norm_rms_epsilon : "0.000001"
2026-04-07T13:56:50.868112Z  INFO candle_vllm_core::backend::gguf: general.quantization_version : "2"
2026-04-07T13:56:50.868114Z  INFO candle_vllm_core::backend::gguf: general.name : "Unsloth_Gguf_Ghe5Te28"
2026-04-07T13:56:50.868115Z  INFO candle_vllm_core::backend::gguf: general.architecture : "qwen3"
2026-04-07T13:56:50.868117Z  INFO candle_vllm_core::backend::gguf: qwen3.embedding_length : "2560"
2026-04-07T13:56:50.868118Z  INFO candle_vllm_core::backend::gguf: qwen3.context_length : "262144"
2026-04-07T13:56:50.868120Z  INFO candle_vllm_core::backend::gguf: qwen3.feed_forward_length : "9728"
2026-04-07T13:56:50.868664Z  INFO candle_vllm_core::backend::gguf: qwen3.rope.freq_base : "5000000"
2026-04-07T13:56:50.868667Z  INFO candle_vllm_core::backend::gguf: qwen3.attention.head_count_kv : "8"
2026-04-07T13:56:50.868668Z  INFO candle_vllm_core::backend::gguf: general.repo_url : "https://huggingface.co/unsloth"
2026-04-07T13:56:50.985038Z  INFO candle_vllm_core::backend::gguf: GGUF tokenizer model is `gpt2`, kind: `Bpe`, num tokens: 151936, num special tokens: 293, num added tokens: 0, num merges: 151387, num scores: 0
2026-04-07T13:56:50.985153Z  WARN candle_vllm_core::backend::gguf: ChatML encode test prompt="<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n" num_tokens=9 ids=[151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198] tokens=["<|im_start|>", "user", "Ċ", "hello", "<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ"]
2026-04-07T13:56:50.996385Z  INFO candle_vllm_core::openai::pipelines::pipeline: Chat Template {%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %} 

2026-04-07T13:56:50.996474Z  INFO candle_vllm_core::openai::pipelines::pipeline: PipelineConfig { max_model_len: 262144, default_max_tokens: 16384, generation_cfg: None }
2026-04-07T13:56:51.009230Z  WARN candle_vllm_core::openai::pipelines::pipeline: stop_token_ids [151645]
The following batches for capture: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
2026-04-07T13:56:51.009393Z  INFO candle_vllm_server: parallel model: multiprocess!
2026-04-07T13:56:51.009487Z  INFO candle_vllm_core::scheduler::cache_engine: TurboQuant KV-cache compression enabled bits=3 num_layers=36 num_kv_heads=8 head_dim=128
Using FP8 KV Cache? false, cache dtype BF16
candle-vllm error: Failed to allocate KV cache (likely CUDA out-of-memory): DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory"). Try reducing `--mem`/`mem` in models.yaml, lowering `max_num_seqs`, closing other GPU workloads, or switching to a smaller model.