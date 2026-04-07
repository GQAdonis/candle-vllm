//! TurboQuant KV-cache compression layer for candle-vllm.
//!
//! Integrates the `turboquant` crate (ICLR 2026, Google Research) as a
//! runtime-configurable hybrid compression layer:
//!
//! - KV blocks are stored in turboquant-compressed form (6–10x smaller than fp16).
//! - At each decode step the active working-set of blocks is decompressed on-demand
//!   into short-lived candle `Tensor` slices that are fed to `PagedAttention`.
//! - The uncompressed tensor path is unchanged when compression is disabled.
//!
//! # Memory layout
//!
//! Each `CompressedLayerCache` stores key/value vectors for one transformer layer.
//! Slots are indexed by:
//!   `slot_id = block_id * block_size + slot_in_block`
//!
//! Decompressed tensors have the paged-attention layouts:
//! - Standard: key `(n_blocks, num_kv_heads, head_dim/x, block_size, x)`,
//!             value `(n_blocks, num_kv_heads, head_dim, block_size)`
//! - Flash:    key/value `(n_blocks, block_size, num_kv_heads, head_dim)`

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use turboquant::{TurboQuant, TurboVectorMse};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for TurboQuant KV-cache compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheCompressionConfig {
    /// Bits per coordinate: 2, 3 (default), or 4.
    pub bits: u8,
    /// Policy controlling when compression is applied.
    pub policy: CompressionPolicy,
}

impl Default for KvCacheCompressionConfig {
    fn default() -> Self {
        Self {
            bits: 3,
            policy: CompressionPolicy::ThresholdTokens(4096),
        }
    }
}

/// Policy controlling when KV-cache compression is applied to a sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionPolicy {
    /// Always compress all KV vectors.
    Always,
    /// Compress when the sequence context length exceeds `n` tokens.
    ThresholdTokens(usize),
    /// Compress when the fraction of free GPU blocks falls below `free_block_pct`.
    MemoryPressure { free_block_pct: f32 },
    /// Never compress (disabled).
    Disabled,
}

impl CompressionPolicy {
    /// Returns `true` if compression should be applied given the current state.
    pub fn should_compress(&self, ctx_len: usize, free_blocks: usize, total_blocks: usize) -> bool {
        match self {
            CompressionPolicy::Always => true,
            CompressionPolicy::ThresholdTokens(n) => ctx_len >= *n,
            CompressionPolicy::MemoryPressure { free_block_pct } => {
                if total_blocks == 0 {
                    return false;
                }
                let pct = free_blocks as f32 / total_blocks as f32;
                pct < *free_block_pct
            }
            CompressionPolicy::Disabled => false,
        }
    }
}

// ── Per-slot storage ─────────────────────────────────────────────────────────

/// Compressed key and value vectors for one token slot across all KV heads.
#[derive(Clone, Debug)]
pub struct CompressedSlot {
    /// Compressed key vectors, one per KV head.
    pub keys: Vec<TurboVectorMse>,
    /// Compressed value vectors, one per KV head.
    pub vals: Vec<TurboVectorMse>,
}

// ── Per-layer compressed store ───────────────────────────────────────────────

/// Compressed KV-cache for one transformer layer.
///
/// Slots are stored in a `HashMap` keyed by `slot_id`:
///   `slot_id = block_id * block_size + slot_in_block`
///
/// This design allows incremental population (slots are `None` until filled)
/// and avoids pre-allocating memory for the maximum number of blocks.
#[derive(Debug)]
pub struct CompressedLayerCache {
    slots: HashMap<u64, CompressedSlot>,
    tq_key: TurboQuant,
    tq_val: TurboQuant,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
}

impl CompressedLayerCache {
    /// Creates an empty cache for one layer.
    ///
    /// `head_dim` must be a power of two (TurboQuant requirement).
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        bits: u8,
        key_seed: u64,
        val_seed: u64,
    ) -> candle_core::Result<Self> {
        if !head_dim.is_power_of_two() {
            candle_core::bail!("TurboQuant requires head_dim to be a power of two, got {head_dim}");
        }
        let tq_key = TurboQuant::new(head_dim, bits, key_seed)
            .map_err(|e| candle_core::Error::Msg(format!("TurboQuant key init: {e}")))?;
        let tq_val = TurboQuant::new(head_dim, bits, val_seed)
            .map_err(|e| candle_core::Error::Msg(format!("TurboQuant val init: {e}")))?;
        Ok(Self {
            slots: HashMap::new(),
            tq_key,
            tq_val,
            num_kv_heads,
            head_dim,
            block_size,
        })
    }

    /// Compress and store one token slot's KV vectors for all heads.
    ///
    /// `keys` and `vals` are flat `f32` slices of length `num_kv_heads * head_dim`.
    pub fn push_slot(
        &mut self,
        block_id: usize,
        slot_in_block: usize,
        keys: &[f32],
        vals: &[f32],
    ) -> candle_core::Result<()> {
        let expected = self.num_kv_heads * self.head_dim;
        if keys.len() != expected || vals.len() != expected {
            candle_core::bail!(
                "push_slot: expected {expected} floats per KV vector, got keys={} vals={}",
                keys.len(),
                vals.len()
            );
        }
        let mut ck = Vec::with_capacity(self.num_kv_heads);
        let mut cv = Vec::with_capacity(self.num_kv_heads);
        for h in 0..self.num_kv_heads {
            let start = h * self.head_dim;
            ck.push(
                self.tq_key
                    .compress_mse(&keys[start..start + self.head_dim])
                    .map_err(|e| candle_core::Error::Msg(format!("compress key h={h}: {e}")))?,
            );
            cv.push(
                self.tq_val
                    .compress_mse(&vals[start..start + self.head_dim])
                    .map_err(|e| candle_core::Error::Msg(format!("compress val h={h}: {e}")))?,
            );
        }
        let slot_id = (block_id * self.block_size + slot_in_block) as u64;
        self.slots
            .insert(slot_id, CompressedSlot { keys: ck, vals: cv });
        Ok(())
    }

    /// Decompress the blocks identified by `block_ids` into f32 buffers.
    ///
    /// Buffer layout (both key and value):
    ///   `[block_idx, slot, head, head_dim_elem]` — all dimensions contiguous.
    ///
    /// Unfilled slots are zero-initialised.
    fn decompress_to_f32(&self, block_ids: &[usize]) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
        let n = block_ids.len();
        let stride = self.block_size * self.num_kv_heads * self.head_dim;
        let total = n * stride;
        let mut key_buf = vec![0.0f32; total];
        let mut val_buf = vec![0.0f32; total];

        for (bi, &block_id) in block_ids.iter().enumerate() {
            for slot in 0..self.block_size {
                let slot_id = (block_id * self.block_size + slot) as u64;
                if let Some(cs) = self.slots.get(&slot_id) {
                    for h in 0..self.num_kv_heads {
                        let out = bi * stride
                            + slot * self.num_kv_heads * self.head_dim
                            + h * self.head_dim;
                        let dk = self
                            .tq_key
                            .decompress_mse(&cs.keys[h])
                            .map_err(|e| candle_core::Error::Msg(format!("decompress key: {e}")))?;
                        let dv = self
                            .tq_val
                            .decompress_mse(&cs.vals[h])
                            .map_err(|e| candle_core::Error::Msg(format!("decompress val: {e}")))?;
                        key_buf[out..out + self.head_dim].copy_from_slice(&dk);
                        val_buf[out..out + self.head_dim].copy_from_slice(&dv);
                    }
                }
                // unfilled slot → zeros remain
            }
        }
        Ok((key_buf, val_buf))
    }

    /// Decompress `block_ids` into tensors with the **standard** paged-attention layout.
    ///
    /// Key:   `(n, num_kv_heads, head_dim/x, block_size, x)`  where `x = 16 / dtype_bytes`
    /// Value: `(n, num_kv_heads, head_dim, block_size)`
    pub fn decompress_to_standard_tensors(
        &self,
        block_ids: &[usize],
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let n = block_ids.len();
        let (key_f32, val_f32) = self.decompress_to_f32(block_ids)?;
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;

        // Buffer is [n, block_size, num_kv_heads, head_dim] in row-major order.
        let key_t = Tensor::from_vec(
            key_f32,
            (n, self.block_size, self.num_kv_heads, self.head_dim),
            device,
        )?
        .to_dtype(dtype)?
        // → [n, num_kv_heads, block_size, head_dim]
        .permute([0, 2, 1, 3])?
        // → [n, num_kv_heads, block_size, head_dim/x, x]
        .reshape((n, self.num_kv_heads, self.block_size, self.head_dim / x, x))?
        // → [n, num_kv_heads, head_dim/x, block_size, x]
        .permute([0, 1, 3, 2, 4])?
        .contiguous()?;

        let val_t = Tensor::from_vec(
            val_f32,
            (n, self.block_size, self.num_kv_heads, self.head_dim),
            device,
        )?
        .to_dtype(dtype)?
        // → [n, num_kv_heads, head_dim, block_size]
        .permute([0, 2, 3, 1])?
        .contiguous()?;

        Ok((key_t, val_t))
    }

    /// Decompress `block_ids` into tensors with the **flash-attention** layout.
    ///
    /// Key and Value: `(n, block_size, num_kv_heads, head_dim)`
    pub fn decompress_to_flash_tensors(
        &self,
        block_ids: &[usize],
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let n = block_ids.len();
        let (key_f32, val_f32) = self.decompress_to_f32(block_ids)?;

        let key_t = Tensor::from_vec(
            key_f32,
            (n, self.block_size, self.num_kv_heads, self.head_dim),
            device,
        )?
        .to_dtype(dtype)?;

        let val_t = Tensor::from_vec(
            val_f32,
            (n, self.block_size, self.num_kv_heads, self.head_dim),
            device,
        )?
        .to_dtype(dtype)?;

        Ok((key_t, val_t))
    }

    /// Decompress all filled blocks into a full-sized tensor covering block IDs 0..`num_blocks`.
    ///
    /// Unfilled blocks remain zero. Returns tensors that can be indexed directly by the
    /// existing (unmodified) block tables — no remapping required.
    ///
    /// This is the simplest decompression path; use `decompress_to_*_tensors` with explicit
    /// block IDs and a remapped block table for lower peak memory.
    pub fn decompress_all_standard(
        &self,
        num_blocks: usize,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let all_ids: Vec<usize> = (0..num_blocks).collect();
        self.decompress_to_standard_tensors(&all_ids, dtype, device)
    }

    /// Same as `decompress_all_standard` but for flash-attention layout.
    pub fn decompress_all_flash(
        &self,
        num_blocks: usize,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let all_ids: Vec<usize> = (0..num_blocks).collect();
        self.decompress_to_flash_tensors(&all_ids, dtype, device)
    }

    /// Bytes occupied by all compressed data in this layer.
    pub fn compressed_size_bytes(&self) -> usize {
        self.slots
            .values()
            .map(|s| {
                s.keys.iter().map(|q| q.byte_size()).sum::<usize>()
                    + s.vals.iter().map(|q| q.byte_size()).sum::<usize>()
            })
            .sum()
    }

    /// Number of filled (compressed) token slots.
    pub fn filled_slots(&self) -> usize {
        self.slots.len()
    }

    /// Approximate compression ratio vs fp16 storage.
    pub fn compression_ratio(&self) -> f32 {
        if self.slots.is_empty() {
            return 1.0;
        }
        let compressed = self.compressed_size_bytes();
        let uncompressed = self.slots.len() * self.num_kv_heads * self.head_dim * 2 * 2; // key+val f16
        uncompressed as f32 / compressed as f32
    }

    /// Returns the `head_dim` this cache was initialised for.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

// ── Full compressed store (all layers) ───────────────────────────────────────

/// Compressed KV store for all transformer layers.
#[derive(Debug)]
pub struct CompressedStore {
    /// One `CompressedLayerCache` per model layer.
    pub layers: Vec<CompressedLayerCache>,
}

impl CompressedStore {
    /// Creates a new store with one `CompressedLayerCache` per layer.
    ///
    /// Seeds are deterministically derived from the layer index so each layer
    /// uses a distinct rotation matrix.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        bits: u8,
    ) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let key_seed = 0x5A5A_5A5A_u64.wrapping_add(layer as u64 * 0x1111_1111);
            let val_seed = 0xA5A5_A5A5_u64.wrapping_add(layer as u64 * 0x2222_2222);
            layers.push(CompressedLayerCache::new(
                num_kv_heads,
                head_dim,
                block_size,
                bits,
                key_seed,
                val_seed,
            )?);
        }
        Ok(Self { layers })
    }

    /// Compress and store one token slot for a specific layer.
    pub fn push_slot(
        &mut self,
        layer: usize,
        block_id: usize,
        slot_in_block: usize,
        keys: &[f32],
        vals: &[f32],
    ) -> candle_core::Result<()> {
        self.layers[layer].push_slot(block_id, slot_in_block, keys, vals)
    }

    /// Total compressed memory across all layers, in bytes.
    pub fn total_compressed_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.compressed_size_bytes()).sum()
    }
}

// ── KvCacheTensors handle ─────────────────────────────────────────────────────

/// A KV-cache handle that is either a reference to pre-allocated tensors (uncompressed
/// path) or an owned set of tensors decompressed on demand (compressed path).
///
/// Implements `Deref<Target = Vec<(Tensor, Tensor)>>` so call-sites treat both
/// variants identically.
pub enum KvCacheTensors {
    /// Uncompressed tensors referenced via an owned clone of the per-layer vec.
    /// (We clone `Arc<Tensor>` handles, not the tensor data itself.)
    Uncompressed(Vec<(Tensor, Tensor)>),
    /// Tensors decompressed on-the-fly; dropped after the forward pass.
    Decompressed(Vec<(Tensor, Tensor)>),
}

impl std::ops::Deref for KvCacheTensors {
    type Target = Vec<(Tensor, Tensor)>;

    fn deref(&self) -> &Self::Target {
        match self {
            KvCacheTensors::Uncompressed(v) | KvCacheTensors::Decompressed(v) => v,
        }
    }
}

// ── Block-count profiling helper ──────────────────────────────────────────────

/// Bytes per physical KV block for block-count profiling.
///
/// Used by the engine initialiser to scale up `num_gpu_blocks` when compression
/// is enabled (compressed blocks are smaller so more fit in the same memory budget).
pub fn bytes_per_block(
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    dtype: DType,
    compression: Option<&KvCacheCompressionConfig>,
) -> usize {
    // Number of key (or value) vectors in one block: one per (slot, head) pair.
    let num_vectors = block_size * num_kv_heads;
    if let Some(cfg) = compression {
        // Compressed: one TurboVectorMse per K or V vector.
        //   norm (f32, 4 B) + bit-packed indices for `head_dim` coordinates.
        let packed = turboquant::bitpack::packed_byte_size(head_dim, cfg.bits);
        let compressed_vec_bytes = 4 + packed; // norm + indices
        num_vectors * 2 * compressed_vec_bytes // ×2 for key + value
    } else {
        // Uncompressed: `head_dim` elements per vector, dtype bytes each, ×2 for K+V.
        num_vectors * head_dim * 2 * dtype.size_in_bytes()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_vec(len: usize, freq: f32) -> Vec<f32> {
        (0..len).map(|i| (i as f32 * freq).sin()).collect()
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        dot / (na * nb).max(f32::EPSILON)
    }

    #[test]
    fn test_compression_roundtrip_cosine() {
        let head_dim = 128;
        let num_heads = 4;
        let block_size = 16;
        let num_blocks = 2;

        let mut cache =
            CompressedLayerCache::new(num_heads, head_dim, block_size, 3, 42, 99).unwrap();

        let mut originals: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for block in 0..num_blocks {
            for slot in 0..block_size {
                let k: Vec<f32> = (0..num_heads)
                    .flat_map(|h| sine_vec(head_dim, 0.05 + h as f32 * 0.01))
                    .collect();
                let v: Vec<f32> = (0..num_heads)
                    .flat_map(|h| sine_vec(head_dim, 0.03 + h as f32 * 0.01))
                    .collect();
                cache.push_slot(block, slot, &k, &v).unwrap();
                originals.push((k, v));
            }
        }

        let block_ids: Vec<usize> = (0..num_blocks).collect();
        let (key_buf, val_buf) = cache.decompress_to_f32(&block_ids).unwrap();

        for block in 0..num_blocks {
            for slot in 0..block_size {
                let idx = block * block_size + slot;
                let stride = num_heads * head_dim;
                for h in 0..num_heads {
                    let orig_k = &originals[idx].0[h * head_dim..(h + 1) * head_dim];
                    let rec_off = block * block_size * num_heads * head_dim
                        + slot * num_heads * head_dim
                        + h * head_dim;
                    let rec_k = &key_buf[rec_off..rec_off + head_dim];
                    let sim = cosine_similarity(orig_k, rec_k);
                    assert!(
                        sim > 0.95,
                        "cosine similarity too low: block={block} slot={slot} head={h} sim={sim:.4}"
                    );
                    let _ = stride; // suppress unused warning
                }
            }
        }
    }

    #[test]
    fn test_compression_ratio_reasonable() {
        let mut cache = CompressedLayerCache::new(8, 128, 16, 3, 1, 2).unwrap();
        for b in 0..4 {
            for s in 0..16 {
                let k = sine_vec(8 * 128, 0.02 + b as f32 * 0.001 + s as f32 * 0.0001);
                let v = sine_vec(8 * 128, 0.04 + b as f32 * 0.001);
                cache.push_slot(b, s, &k, &v).unwrap();
            }
        }
        let ratio = cache.compression_ratio();
        assert!(
            ratio > 2.0,
            "expected compression ratio > 2×, got {ratio:.2}"
        );
    }

    #[test]
    fn test_bytes_per_block_compressed_smaller() {
        let cfg = KvCacheCompressionConfig {
            bits: 3,
            policy: CompressionPolicy::Always,
        };
        let uncompressed = bytes_per_block(8, 128, 16, DType::F16, None);
        let compressed = bytes_per_block(8, 128, 16, DType::F16, Some(&cfg));
        assert!(
            compressed < uncompressed,
            "compressed bytes ({compressed}) should be < uncompressed ({uncompressed})"
        );
        let ratio = uncompressed as f32 / compressed as f32;
        assert!(ratio > 2.0, "expected > 2× size reduction, got {ratio:.2}");
    }

    #[test]
    fn test_policy_threshold_tokens() {
        let policy = CompressionPolicy::ThresholdTokens(4096);
        assert!(!policy.should_compress(1024, 100, 200));
        assert!(policy.should_compress(4096, 100, 200));
        assert!(policy.should_compress(8192, 10, 200));
    }

    #[test]
    fn test_policy_memory_pressure() {
        let policy = CompressionPolicy::MemoryPressure {
            free_block_pct: 0.2,
        };
        assert!(!policy.should_compress(0, 50, 100)); // 50% free → no pressure
        assert!(policy.should_compress(0, 10, 100)); // 10% free → pressure
        assert!(!policy.should_compress(0, 100, 0)); // total=0 → never compress
    }
}
