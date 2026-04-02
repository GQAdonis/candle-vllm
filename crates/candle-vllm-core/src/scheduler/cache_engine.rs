use crate::openai::models::Config;
use crate::scheduler::kv_compression::{
    CompressedStore, KvCacheCompressionConfig, KvCacheTensors,
};
use candle_core::{DType, Device, Result, Tensor};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::backend::{copy_blocks, swap_blocks};

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: Option<usize>, // Set after profiling init
    pub num_cpu_blocks: Option<usize>, // Set after profiling init
    pub fully_init: bool,
    pub dtype: DType,
    pub kvcache_mem_gpu: usize, // in MB
    pub mamba_cache_budget_bytes: usize,
    /// Optional TurboQuant compression configuration.
    pub compression: Option<KvCacheCompressionConfig>,
}

impl CacheConfig {
    pub fn set_num_gpu_blocks(&mut self, num_gpu_blocks: usize) {
        if self.num_cpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_gpu_blocks = Some(num_gpu_blocks);
    }
    pub fn set_num_cpu_blocks(&mut self, num_cpu_blocks: usize) {
        if self.num_gpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_cpu_blocks = Some(num_cpu_blocks);
    }
}

pub type KVCache = (Tensor, Tensor);

#[derive(Debug)]
pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
    /// CPU KV cache for offloading (reserved for future use)
    #[allow(dead_code)]
    cpu_cache: Vec<KVCache>,
    /// Number of model layers (reserved for layer-wise cache management)
    #[allow(dead_code)]
    num_layers: usize,
    /// Optional TurboQuant compressed store (None when compression is disabled).
    compressed_store: Option<Arc<Mutex<CompressedStore>>>,
    /// Inference device (needed for on-demand decompression).
    device: Device,
    /// KV-cache dtype (needed for on-demand decompression).
    dtype: DType,
    /// Number of GPU blocks (needed for full-tensor decompression).
    num_gpu_blocks: usize,
    /// Whether flash-attention layout is in use.
    flash_layout: bool,
}

impl CacheEngine {
    pub fn new(
        model_config: &Config,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        num_shards: usize,
    ) -> Result<Self> {
        let num_gpu_blocks = if cfg!(feature = "cuda") {
            cache_config.num_gpu_blocks.unwrap_or(32)
        } else if device.is_cpu() {
            1
        } else {
            cache_config.num_gpu_blocks.unwrap_or(32)
        };

        let flash_layout = cfg!(any(feature = "flashattn", feature = "flashinfer"));

        let compressed_store = if let Some(ref cfg) = cache_config.compression {
            let num_kv_heads =
                model_config.num_key_value_heads.unwrap_or(model_config.num_attention_heads)
                    / num_shards;
            let head_dim = model_config
                .head_dim
                .unwrap_or(model_config.hidden_size / model_config.num_attention_heads);
            let store = CompressedStore::new(
                model_config.kv_cache_num_layers(),
                num_kv_heads,
                head_dim,
                cache_config.block_size,
                cfg.bits,
            )?;
            tracing::info!(
                bits = cfg.bits,
                num_layers = model_config.kv_cache_num_layers(),
                num_kv_heads,
                head_dim,
                "TurboQuant KV-cache compression enabled"
            );
            Some(Arc::new(Mutex::new(store)))
        } else {
            None
        };

        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_kv_cache(
                model_config,
                cache_config,
                dtype,
                device,
                num_shards,
            )?)),
            cpu_cache: Self::allocate_kv_cache(
                model_config,
                cache_config,
                dtype,
                &Device::Cpu,
                num_shards,
            )?,
            num_layers: model_config.kv_cache_num_layers(),
            compressed_store,
            device: device.clone(),
            dtype,
            num_gpu_blocks,
            flash_layout,
        })
    }

    /// Returns the raw mutex guard over the uncompressed GPU cache.
    ///
    /// Prefer `get_kv_tensors()` which transparently handles both compressed
    /// and uncompressed paths.
    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        loop {
            if let Ok(v) = self.gpu_cache.try_lock() {
                return v;
            }
        }
    }

    /// Returns a KV-cache handle suitable for a model forward pass.
    ///
    /// - Uncompressed path: returns a shallow clone of the pre-allocated tensors
    ///   (cheap — no data copy; candle tensors are reference-counted).
    /// - Compressed path: decompresses all allocated blocks into fresh tensors.
    ///   These tensors are dropped after the forward pass, keeping the at-rest
    ///   memory footprint small.
    pub fn get_kv_tensors(&self) -> Result<KvCacheTensors> {
        if let Some(ref store_mu) = self.compressed_store {
            let store = loop {
                if let Ok(g) = store_mu.try_lock() {
                    break g;
                }
            };
            let mut tensors = Vec::with_capacity(store.layers.len());
            for layer in &store.layers {
                let (k, v) = if self.flash_layout {
                    layer.decompress_all_flash(
                        self.num_gpu_blocks,
                        self.dtype,
                        &self.device,
                    )?
                } else {
                    layer.decompress_all_standard(
                        self.num_gpu_blocks,
                        self.dtype,
                        &self.device,
                    )?
                };
                tensors.push((k, v));
            }
            Ok(KvCacheTensors::Decompressed(tensors))
        } else {
            let guard = self.get_kv_cache();
            let cloned: Vec<(Tensor, Tensor)> = guard
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Ok(KvCacheTensors::Uncompressed(cloned))
        }
    }

    /// Compress and store one token slot for the given layer.
    ///
    /// `keys` and `vals` are flat `f32` slices of length `num_kv_heads * head_dim`.
    /// No-op when compression is disabled.
    pub fn push_compressed(
        &self,
        layer: usize,
        block_id: usize,
        slot_in_block: usize,
        keys: &[f32],
        vals: &[f32],
    ) -> Result<()> {
        if let Some(ref store_mu) = self.compressed_store {
            let mut store = loop {
                if let Ok(g) = store_mu.try_lock() {
                    break g;
                }
            };
            store.push_slot(layer, block_id, slot_in_block, keys, vals)?;
        }
        Ok(())
    }

    /// Returns `true` if TurboQuant compression is currently enabled.
    pub fn compression_enabled(&self) -> bool {
        self.compressed_store.is_some()
    }

    fn allocate_kv_cache(
        model_config: &Config,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        num_shards: usize,
    ) -> Result<Vec<KVCache>> {
        #[cfg(feature = "cuda")]
        let num_blocks = cache_config.num_gpu_blocks.unwrap_or(32);
        // dummy cpu kvcache on Metal
        #[cfg(not(feature = "cuda"))]
        let num_blocks = if device.is_cpu() {
            1
        } else {
            cache_config.num_gpu_blocks.unwrap_or(32)
        };

        if cfg!(any(feature = "flashattn", feature = "flashinfer")) {
            let kv_shape = Self::calculate_flash_key_value_block_shape(
                model_config,
                cache_config.block_size,
                num_shards,
            );

            let mut cache = Vec::new();
            for _ in 0..model_config.kv_cache_num_layers() {
                let key_blocks = Tensor::zeros(
                    (num_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    device,
                )?;
                let value_blocks = Tensor::zeros(
                    (num_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    device,
                )?;
                cache.push((key_blocks, value_blocks));
            }
            Ok(cache)
        } else {
            let fp8_kvcache = matches!(dtype, DType::U8);
            if !device.is_cpu() {
                println!(
                    "Using FP8 KV Cache? {}, cache dtype {:?}",
                    fp8_kvcache, dtype
                );
            }

            let kshape = Self::calculate_key_block_shape(
                model_config,
                dtype,
                cache_config.block_size,
                num_shards,
            );
            let vshape = Self::calculate_value_block_shape(
                model_config,
                cache_config.block_size,
                num_shards,
            );

            let mut cache = Vec::new();
            for _ in 0..model_config.kv_cache_num_layers() {
                let key_blocks = Tensor::zeros(
                    (num_blocks, kshape.0, kshape.1, kshape.2, kshape.3),
                    dtype,
                    device,
                )?;
                let value_blocks =
                    Tensor::zeros((num_blocks, vshape.0, vshape.1, vshape.2), dtype, device)?;
                cache.push((key_blocks, value_blocks));
            }
            Ok(cache)
        }
    }
}

impl CacheEngine {
    fn calculate_key_block_shape(
        cfg: &Config,
        dtype: DType,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) / num_shards,
            cfg.k_head_dim() / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        cfg: &Config,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize) {
        (
            cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) / num_shards,
            cfg.v_head_dim(),
            block_size,
        )
    }

    //[num_blocks, block_size, num_kv_heads, head_size]
    fn calculate_flash_key_value_block_shape(
        cfg: &Config,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize) {
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);

        (
            block_size,
            cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) / num_shards,
            head_dim,
        )
    }
}

impl CacheEngine {
    pub fn swap_in(&self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            // Dummy implementation when cuda/metal features are not enabled
            let _ = src_to_dst; // Avoid unused variable warning
            return Ok(());
        }

        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            for i in 0..self.num_layers {
                let (src_key_cache, src_value_cache) = self.cpu_cache.get(i).unwrap();
                let mut gpu_cache = self.get_kv_cache();
                let (dst_key_cache, dst_value_cache) = gpu_cache.get_mut(i).unwrap();
                // Swap (copy) key blocks
                swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone())?;
                // Swap (copy) key blocks
                swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone())?;
            }
            Ok(())
        }
    }

    pub fn swap_out(&mut self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            // Dummy implementation when cuda/metal features are not enabled
            let _ = src_to_dst; // Avoid unused variable warning
            return Ok(());
        }

        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            for i in 0..self.num_layers {
                let gpu_cache = self.get_kv_cache();
                let (src_key_cache, src_value_cache) = gpu_cache.get(i).unwrap().clone();
                drop(gpu_cache);

                let (dst_key_cache, dst_value_cache) = self.cpu_cache.get_mut(i).unwrap();
                // Swap (copy) key blocks
                swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone())?;
                // Swap (copy) key blocks
                swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone())?;
            }
            Ok(())
        }
    }
    #[allow(unused_unsafe)]
    pub fn copy(&mut self, src_to_dst: HashMap<usize, Vec<usize>>) -> Result<()> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            // Dummy implementation when cuda/metal features are not enabled
            let _ = src_to_dst; // Avoid unused variable warning
            return Ok(());
        }

        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            let mut gpu_cache = self.get_kv_cache();
            #[allow(clippy::map_identity)]
            let caches: (Vec<&mut Tensor>, Vec<&mut Tensor>) =
                gpu_cache.iter_mut().map(|(a, b)| (a, b)).unzip();
            let (key_caches, value_caches) = caches;

            // NOTE(EricLBuehler): This may synchronize the CPU and GPU
            unsafe {
                copy_blocks(key_caches, value_caches, src_to_dst)?;
            }
            Ok(())
        }
    }
}
