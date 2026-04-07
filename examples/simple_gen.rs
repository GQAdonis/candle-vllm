//! Minimal chat generation using the library [`InferenceEngine`] API.
//!
//! Usage:
//!   cargo run --example simple_gen --release --features metal -- /path/to/model
//!   (on Linux, use `--features cuda` or CPU-appropriate flags as needed.)

use candle_vllm::{EngineConfig, GenerationParams, InferenceEngine};
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

    let params = GenerationParams {
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.95),
        ..Default::default()
    };

    let out = engine.generate("Talk about China.", params).await?;
    println!("Response: {:?}", out.text);

    Ok(())
}
