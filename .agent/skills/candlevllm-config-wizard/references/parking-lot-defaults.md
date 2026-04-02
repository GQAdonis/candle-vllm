# Parking Lot Configuration Defaults

The parking lot manages request scheduling, queuing, and resource allocation for the inference server.

## Default Configurations by Use Case

| Use Case | worker_threads | max_units | max_queue_depth | timeout_secs |
|---|---|---|---|---|
| Development | 2 | 1024 | 10 | 120 |
| Chat (low latency) | num_cpus | 2048 | 50 | 120 |
| Batch processing | num_cpus | 4096 | 200 | 600 |
| High throughput API | num_cpus | 8192 | 500 | 300 |
| Resource constrained | 2 | 512 | 25 | 180 |

## Queue Backend Options

| Backend | Type | Persistence | Distributed | Best For |
|---|---|---|---|---|
| memory | In-memory | No | No | Development, single-instance |
| postgres | Database | Yes | Yes | Multi-instance, production |
| yaque | File-based | Yes | No | Single-instance with persistence |

## Parameter Descriptions

### Pool
- **worker_threads**: Number of worker threads for processing requests. Use `num_cpus` for production, 2 for development.

### Limits
- **max_units**: Maximum number of token units that can be processed concurrently. Higher values allow more parallelism but consume more memory.
- **max_queue_depth**: Maximum number of requests that can wait in the queue. Requests beyond this are rejected with a 429 status.
- **timeout_secs**: Maximum time a request can wait in the queue before being timed out.

### Queue
- **backend**: Storage backend for the request queue (memory, postgres, yaque).
- **persistence**: Whether queued requests survive server restarts (true/false).

### Mailbox
- **backend**: Backend for the completion mailbox (memory, postgres, yaque).
- **retention_secs**: How long completed results are retained for retrieval.

## Tuning Guidelines

1. **Low latency chat**: Keep `max_queue_depth` low (50) and `timeout_secs` moderate (120). Users expect fast responses.
2. **Batch processing**: High `max_queue_depth` (200+) and long `timeout_secs` (600). Throughput matters more than latency.
3. **High throughput API**: Balance between queue depth and timeout. Use persistent queue backend for reliability.
4. **Resource constrained**: Limit `max_units` and `worker_threads` to prevent OOM. Use shorter timeouts to free resources.
