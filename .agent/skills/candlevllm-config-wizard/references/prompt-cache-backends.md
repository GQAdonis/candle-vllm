# Prompt Cache Backend Reference

Prompt caching stores computed KV cache entries for common prefixes, avoiding redundant computation for repeated prompts (e.g., system prompts, few-shot examples).

## Backend Comparison

| Backend | Persistence | Distributed | Latency | Setup Required | Feature Flag |
|---|---|---|---|---|---|
| memory | No | No | <1ms | None | (default) |
| sled | Yes | No | ~1ms | cache_path | prompt-cache-sled |
| redis | Yes | Yes | ~5ms | redis_url | prompt-cache-redis |

## Configuration Parameters

| Parameter | Description | Default |
|---|---|---|
| backend | Cache storage backend | memory |
| min_prefix_length | Minimum token count for a prefix to be cached | 16 |
| max_cached_prefixes | Maximum number of cached prefix entries | Depends on memory |
| cache_path | Path for sled database (sled only) | ./prompt_cache |
| redis_url | Redis connection URL (redis only) | redis://localhost:6379 |

## Backend Details

### Memory (Default)
- Zero configuration required
- Fastest access latency
- Cache is lost on server restart
- Best for: development, single-session deployments, low-latency requirements

### Sled (File-Based Persistent)
- Requires `prompt-cache-sled` feature flag at build time
- Persists cache to disk at `cache_path`
- Survives server restarts
- Best for: single-instance production deployments where cache warmup is expensive

### Redis (Distributed Persistent)
- Requires `prompt-cache-redis` feature flag at build time
- Shared cache across multiple server instances
- Highest latency but enables distributed caching
- Best for: multi-instance deployments, horizontal scaling

## Sizing Guidelines

- Each cached prefix stores the full KV cache for the prefix tokens
- Memory per cached prefix: `prefix_length * per_token_kv_size` (see memory-calculator.md)
- Set `max_cached_prefixes` based on: available memory / average prefix cache size
- Common system prompts are excellent candidates for caching (typically 100-500 tokens)

## When to Use Prompt Caching

- Many requests share the same system prompt
- Few-shot examples are prepended to most requests
- RAG pipelines with common document prefixes
- Tool-calling scenarios with large tool descriptions
