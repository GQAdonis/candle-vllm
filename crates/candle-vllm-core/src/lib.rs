#![warn(clippy::cast_lossless)]
use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Result as CandleResult};
use candle_core as candle;
use std::path::Path;
use tracing::warn;
pub mod api;
pub mod attention;
pub mod backend;
pub mod config;
pub mod engine_builder_ext;
pub mod engine_params;
pub mod engine_state;
pub mod models_config;
pub mod models_engine_builder;
pub mod openai;
pub mod parking_lot;
pub mod prompt_cache;
pub mod scheduler;
pub mod tools;
pub mod vision;
pub use attention::{InputMetadata, PagedAttention};

pub const MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV: &str = "VLLM_RS_MAMBA_SNAPSHOT_STRIDE_BLOCKS";
pub const DEFAULT_MAMBA_SNAPSHOT_BLOCK_STRIDE: usize = 8;

// Re-export public API types
pub use api::{
    EngineConfig, EngineConfigBuilder, Error, FinishReason, GenerationOutput, GenerationParams,
    GenerationStats, InferenceEngine, InferenceEngineBuilder, ModelInfo, Result,
};
pub use engine_builder_ext::{
    EngineBuilderResult, ExtendedEngineBuilder, ExtendedEngineConfigBuilder,
};
pub use engine_state::{
    EngineHealthStatus, EngineStateConfig, EngineStateManager, EngineStateManagerBuilder,
    EngineStats, VisionToolHealth,
};
pub use models_engine_builder::{
    ModelsEngineBuilder, ModelsEngineBuilderConfig, ModelsEngineBuilderFactory,
    ModelsEngineBuilderResult,
};

pub fn hub_load_local_safetensors(
    path: &String,
    json_file: &str,
) -> CandleResult<Vec<std::path::PathBuf>> {
    tracing::info!("{:}", Path::new(path).join(json_file).display());
    let jsfile = std::fs::File::open(Path::new(path).join(json_file))?;
    let json: serde_json::Value = serde_json::from_reader(&jsfile).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file);
        }
    }
    let safetensors_files: Vec<_> = safetensors_files
        .into_iter()
        .map(|v| Path::new(path).join(v))
        .collect();
    Ok(safetensors_files)
}

pub fn new_device(ordinal: usize) -> CandleResult<Device> {
    if cuda_is_available() {
        use candle_core::CudaDevice;
        let device = Device::Cuda(CudaDevice::new_with_stream(ordinal)?);
        Ok(device)
    } else if metal_is_available() {
        Ok(Device::new_metal(ordinal)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            warn!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            warn!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub fn mamba_snapshot_block_stride_blocks() -> usize {
    let default = DEFAULT_MAMBA_SNAPSHOT_BLOCK_STRIDE;
    let Ok(raw) = std::env::var(MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV) else {
        return default;
    };
    match raw.trim().parse::<usize>() {
        Ok(0) => {
            warn!(
                "{} must be >= 1, got 0. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV, default
            );
            default
        }
        Ok(v) => v,
        Err(_) => {
            warn!(
                "Invalid {}='{}'. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV, raw, default
            );
            default
        }
    }
}
