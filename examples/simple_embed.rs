//! Tokenization demo using [`InferenceEngine`] (same load path as generation).
//!
//! The standalone library API does not expose a dedicated embedding tensor helper yet;
//! run the `candle-vllm` server and call `POST /v1/embeddings` for OpenAI-style embeddings.
//!
//! Usage:
//!   cargo run --example simple_embed --release --features metal -- /path/to/model

use candle_vllm::{EngineConfig, InferenceEngine};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/Meta-Llama-3.1-8B-Instruct".to_string());

    println!("Loading model from {}", model_path);

    let config = EngineConfig::builder()
        .model_path(model_path)
        .kv_cache_memory(512)
        .build()?;
    let engine = InferenceEngine::new(config).await?;

    let input = "Hello, world!";
    println!("Input: {}", input);

    let ids = engine.tokenize(input)?;
    println!("Token count: {}", ids.len());
    let n = ids.len().min(8);
    println!("First token ids: {:?}", &ids[..n]);

    Ok(())
}
