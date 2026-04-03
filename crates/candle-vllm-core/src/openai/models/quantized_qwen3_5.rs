use super::{attention::QuantizedAttention, rotary_emb::ScalingRotaryEmbedding, Config};
use super::resolve_qwen3_hybrid_config;
use crate::attention::mamba_cache::MambaCache;
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::utils::{resolve_input_seqlens, resolve_mamba_seq_slots};
use crate::InputMetadata;
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::attention::gdn;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use either::Either;
use parking_lot::{RwLock, RwLockWriteGuard};
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul, // gate
    feed_forward_w2: QMatMul, // down
    feed_forward_w3: QMatMul, // up
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

struct QuantizedGatedDeltaNet {
    in_proj_qkv: QMatMul,
    in_proj_z: QMatMul,
    in_proj_a: QMatMul,
    in_proj_b: QMatMul,
    out_proj: QMatMul,
    conv_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    gdn_norm_weight: Tensor,
    gdn_norm_bias: Option<Tensor>,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    kv_group_size: usize,
    gdn_layer_idx: usize,
    rms_norm_eps: f64,
    scale: f64,
}

impl QuantizedGatedDeltaNet {
    fn repeat_kv_heads(&self, x: Tensor) -> Result<Tensor> {
        if self.num_k_heads == self.num_v_heads {
            return Ok(x);
        }
        let (seq_len, _h, _d) = x.dims3()?;
        x.unsqueeze(2)?
            .broadcast_as((
                seq_len,
                self.num_k_heads,
                self.kv_group_size,
                self.head_k_dim,
            ))?
            .reshape((seq_len, self.num_v_heads, self.head_k_dim))
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn forward(
        &self,
        xs: &Tensor,
        mamba_cache: &mut MambaCache,
        input_metadata: &InputMetadata,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let slot_count = seq_slots.dim(0)?;
        if slot_count == 0 {
            candle_core::bail!("Linear attention requires non-empty sequence slots");
        }
        let (token_count, _hidden) = xs.dims2()?;
        let is_prefill = input_metadata.is_prefill;

        // Project inputs (SplitQkvZaLegacy pattern for GGUF)
        let proj_qkv = self.in_proj_qkv.forward(xs)?;
        let q = proj_qkv.narrow(1, 0, self.key_dim)?.contiguous()?;
        let k = proj_qkv
            .narrow(1, self.key_dim, self.key_dim)?
            .contiguous()?;
        let v = proj_qkv
            .narrow(1, self.key_dim * 2, self.value_dim)?
            .contiguous()?;
        let z = self.in_proj_z.forward(xs)?;
        let b = self.in_proj_b.forward(xs)?;
        let a = self.in_proj_a.forward(xs)?;

        let mixed_qkv = Tensor::cat(&[&q, &k, &v], 1)?;

        // Causal conv1d
        let (kv_conv, prefill_conv_state) = if is_prefill {
            let mut conv_state =
                mamba_cache.get_batch_conv_state(self.gdn_layer_idx, seq_slots)?;
            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");

            let out = gdn::causal_conv1d_fwd(
                &mixed_qkv,
                &self.conv_weight,
                None, // no conv bias in GGUF
                &mut conv_state,
                Some(cu_seqlens),
                true, // SiLU activation
            )?;
            (out, Some(conv_state))
        } else {
            if token_count != slot_count {
                candle_core::bail!(
                    "Linear attention decode mismatch: {} tokens vs {} sequence slots",
                    token_count,
                    slot_count
                );
            }
            let out = gdn::causal_conv1d_update_slots(
                &mixed_qkv,
                &self.conv_weight,
                None,
                mamba_cache.conv_state_mut(self.gdn_layer_idx),
                seq_slots,
                true,
            )?;
            (out, None)
        };
        if let Some(conv_state) = prefill_conv_state {
            mamba_cache.set_batch_conv_state(self.gdn_layer_idx, seq_slots, &conv_state)?;
        }

        // Split convolved output back into q', k', v'
        let q_conv = kv_conv.narrow(1, 0, self.key_dim)?;
        let k_conv = kv_conv.narrow(1, self.key_dim, self.key_dim)?;
        let v_conv = kv_conv.narrow(1, self.key_dim * 2, self.value_dim)?;

        // Fused GDN gating
        let (a_expanded, b_expanded) = (a.unsqueeze(0)?, b.unsqueeze(0)?);
        let (g, beta) =
            gdn::fused_gdn_gating(&self.a_log, &a_expanded, &b_expanded, &self.dt_bias)?;
        let (g, beta) = (g.squeeze(0)?, beta.squeeze(0)?);

        let q: Tensor = q_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let k: Tensor = k_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let v: Tensor = v_conv.reshape((token_count, self.num_v_heads, self.head_v_dim))?;
        let q = gdn::l2_norm_last_dim(&q, 1e-6)?;
        let k = gdn::l2_norm_last_dim(&k, 1e-6)?;
        let (q, k) = (self.repeat_kv_heads(q)?, self.repeat_kv_heads(k)?);

        let output = if is_prefill {
            let q_scaled = (&q * self.scale)?;

            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");

            let global_state = mamba_cache.recurrent_state_mut(self.gdn_layer_idx);
            xs.device().synchronize()?;

            gdn::gated_delta_rule_recurrence_varlen(
                &q_scaled,
                &k,
                &v,
                &g,
                &beta,
                global_state,
                seq_slots,
                cu_seqlens,
            )?
        } else {
            let batch = slot_count;
            let q_b: Tensor =
                (q.reshape((batch, self.num_v_heads, self.head_k_dim))? * self.scale)?;
            let k_b: Tensor = k.reshape((batch, self.num_v_heads, self.head_k_dim))?;
            let v_b: Tensor = v.reshape((batch, self.num_v_heads, self.head_v_dim))?;
            let g_b: Tensor = g.reshape((batch, self.num_v_heads))?;
            let beta_b: Tensor = beta.reshape((batch, self.num_v_heads))?;
            let global_state = mamba_cache.recurrent_state_mut(self.gdn_layer_idx);
            gdn::gated_delta_rule_decode_slots(
                &q_b, &k_b, &v_b, &g_b, &beta_b, global_state, seq_slots,
            )?
        };

        // output: [seq_len, num_v_heads, head_v_dim] -> [seq_len, value_dim]
        let output = output.reshape((token_count, self.value_dim))?;

        // Gated RMSNorm: norm(output) * silu(z) via fused kernel
        let gated_output = gdn::gated_rmsnorm_silu_mul(
            &output,
            &z,
            &self.gdn_norm_weight,
            self.gdn_norm_bias.as_ref(),
            self.rms_norm_eps,
            self.head_v_dim,
        )?;

        // Output projection
        self.out_proj
            .forward(&gated_output.to_dtype(xs.dtype())?)
    }

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    fn forward(
        &self,
        _xs: &Tensor,
        _mamba_cache: &mut MambaCache,
        _input_metadata: &InputMetadata,
        _seq_slots: &Tensor,
    ) -> Result<Tensor> {
        candle_core::bail!("Quantized GatedDeltaNet requires CUDA or Metal backend")
    }
}

enum LayerType {
    FullAttention(QuantizedAttention),
    LinearAttention(QuantizedGatedDeltaNet),
}

struct DecoderLayer {
    attn: LayerType,
    mlp: Mlp,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
}

impl DecoderLayer {
    fn is_full_attention(&self) -> bool {
        matches!(self.attn, LayerType::FullAttention(_))
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        mamba_cache: &mut MambaCache,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = match &self.attn {
            LayerType::FullAttention(attn) => {
                attn.forward(&xs, attention_mask, input_positions, cache, input_metadata)?
            }
            LayerType::LinearAttention(gdn) => {
                gdn.forward(&xs, mamba_cache, input_metadata, seq_slots)?
            }
        };
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

pub struct GGUFQWen3_5 {
    tok_embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    norm: QRmsNorm,
    output: QMatMul,
    mamba_cache: RwLock<MambaCache>,
    cfg: Config,
    dtype: DType,
    device: Device,
}

impl GGUFQWen3_5 {
    pub fn into_config(
        embedding_length: usize,
        head_dim: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        rope_theta: f64,
        rms_eps: f64,
        max_seq_len: usize,
        partial_rotary_factor: Option<f32>,
        kv_cache_dtype: DType,
        full_attention_interval: usize,
        conv_kernel_size: usize,
        ssm_group_count: usize,
        ssm_state_size: usize,
        ssm_time_step_rank: usize,
        ssm_inner_size: usize,
    ) -> Config {
        // Compute linear attention head dimensions:
        // num_v_heads = ssm_time_step_rank, num_k_heads = ssm_group_count
        // head_v_dim = ssm_state_size, head_k_dim = ssm_inner_size / ssm_group_count
        let head_k_dim_linear = ssm_inner_size / ssm_group_count;

        let hybrid_json = serde_json::json!({
            "full_attention_interval": full_attention_interval,
            "conv_kernel_size": conv_kernel_size,
            "linear_num_heads": ssm_group_count,
            "linear_num_value_heads": ssm_time_step_rank,
            "linear_key_head_dim": head_k_dim_linear,
            "linear_value_head_dim": ssm_state_size,
        });

        Config {
            architectures: Some(vec!["Qwen3_5ForCausalLM".to_string()]),
            hidden_size: embedding_length,
            head_dim: Some(head_dim),
            intermediate_size: 0,
            vocab_size: 0,
            num_hidden_layers: block_count,
            num_attention_heads: head_count,
            num_key_value_heads: Some(head_count_kv),
            rms_norm_eps: rms_eps,
            rope_theta,
            rope_local_base_freq: None,
            bos_token_id: Some(super::TokenID(Either::Left(Some(151644)))),
            eos_token_id: Some(super::TokenID(Either::Left(Some(151645)))),
            max_seq_len,
            sliding_window: None,
            sliding_window_pattern: None,
            hidden_act: None,
            hidden_activation: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            max_position_embeddings: Some(max_seq_len),
            original_max_position_embeddings: None,
            attention_bias: Some(false),
            partial_rotary_factor,
            qk_layernorm: false,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quant: None,
            quantization_config: None,
            moe_config: None,
            isq_quant: None,
            fp8_kvcache: Some(kv_cache_dtype == DType::U8),
            extra_config_json: Some(hybrid_json.to_string()),
        }
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
        kv_cache_dtype: DType,
        yarn_scaling_factor: Option<f64>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let reporter = progress_reporter.clone();
        let arch = md_get("general.architecture")?.to_string()?;

        // Extract metadata
        let head_count =
            md_get(format!("{arch}.attention.head_count").as_str())?.to_u32()? as usize;
        let head_count_kv =
            md_get(format!("{arch}.attention.head_count_kv").as_str())?.to_u32()? as usize;
        let embedding_length =
            md_get(format!("{arch}.embedding_length").as_str())?.to_u32()? as usize;
        let head_dim = md_get(format!("{arch}.attention.key_length").as_str());
        let head_dim = if head_dim.is_ok() {
            head_dim.unwrap().to_u32()? as usize
        } else {
            embedding_length / head_count
        };
        let context_length =
            md_get(format!("{arch}.context_length").as_str())?.to_u32()? as usize;
        let block_count =
            md_get(format!("{arch}.block_count").as_str())?.to_u32()? as usize;
        let rms_norm_eps =
            md_get(format!("{arch}.attention.layer_norm_rms_epsilon").as_str())?.to_f32()? as f64;
        let rope_freq_base = md_get(format!("{arch}.rope.freq_base").as_str())
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        // Partial rotary factor from rope.dimension_count
        let rope_dim = md_get(format!("{arch}.rope.dimension_count").as_str());
        let partial_rotary_factor = if rope_dim.is_ok() {
            let rope_dim = rope_dim.unwrap().to_u32()? as usize;
            if rope_dim != head_dim {
                Some(rope_dim as f32 / head_dim as f32)
            } else {
                None
            }
        } else {
            None
        };

        // SSM / hybrid parameters
        let full_attention_interval =
            md_get(format!("{arch}.full_attention_interval").as_str())?.to_u32()? as usize;
        let ssm_conv_kernel =
            md_get(format!("{arch}.ssm.conv_kernel").as_str())?.to_u32()? as usize;
        let ssm_state_size =
            md_get(format!("{arch}.ssm.state_size").as_str())?.to_u32()? as usize;
        let ssm_group_count =
            md_get(format!("{arch}.ssm.group_count").as_str())?.to_u32()? as usize;
        let ssm_time_step_rank =
            md_get(format!("{arch}.ssm.time_step_rank").as_str())?.to_u32()? as usize;
        let ssm_inner_size = md_get(format!("{arch}.ssm.inner_size").as_str())
            .and_then(|m| m.to_u32())
            .unwrap_or(embedding_length as u32) as usize;

        // Build config
        let mut cfg = GGUFQWen3_5::into_config(
            embedding_length,
            head_dim,
            block_count,
            head_count,
            head_count_kv,
            rope_freq_base as f64,
            rms_norm_eps,
            context_length,
            partial_rotary_factor,
            kv_cache_dtype,
            full_attention_interval,
            ssm_conv_kernel,
            ssm_group_count,
            ssm_state_size,
            ssm_time_step_rank,
            ssm_inner_size,
        );
        cfg.apply_runtime_rope_overrides(yarn_scaling_factor);

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, &cfg, device, true)?);

        // Resolve hybrid config to get layer types and GDN parameters
        let hybrid = resolve_qwen3_hybrid_config(&cfg);

        // Compute GDN dimensions
        let num_v_heads = hybrid.num_v_heads;
        let num_k_heads = hybrid.num_k_heads;
        let head_k_dim_gdn = hybrid.key_head_dim;
        let head_v_dim_gdn = hybrid.value_head_dim;
        let key_dim = num_k_heads * head_k_dim_gdn;
        let value_dim = num_v_heads * head_v_dim_gdn;
        let kv_group_size = num_v_heads / num_k_heads;
        let d_conv = key_dim * 2 + value_dim;
        let scale = 1.0f64 / (head_k_dim_gdn as f64).sqrt();

        // Load global tensors
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            _ => QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?,
        };

        // Build layers
        let mut layers = Vec::with_capacity(block_count);
        let mut gdn_layer_idx = 0usize;

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let layer_type_str = hybrid
                .layer_types
                .get(layer_idx)
                .map(String::as_str)
                .unwrap_or("full_attention");

            // MLP (same for both layer types)
            let mlp = {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                }
            };

            let input_layernorm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let post_attention_layernorm =
                ct.tensor(reader, &format!("{prefix}.post_attention_norm.weight"), device)?;

            let attn = if layer_type_str == "full_attention" {
                LayerType::FullAttention(QuantizedAttention::new(
                    &cfg,
                    ct,
                    reader,
                    &prefix,
                    device,
                    dtype,
                    rotary_emb.clone(),
                    cfg.sliding_window,
                )?)
            } else {
                // Linear attention layer (QuantizedGatedDeltaNet)
                let cur_gdn_idx = gdn_layer_idx;
                gdn_layer_idx += 1;

                let in_proj_qkv = QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.attn_qkv.weight"), device)?,
                )?;
                let in_proj_z = QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.attn_gate.weight"), device)?,
                )?;
                let in_proj_a = QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.ssm_alpha.weight"), device)?,
                )?;
                let in_proj_b = QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.ssm_beta.weight"), device)?,
                )?;
                let out_proj = QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.ssm_out.weight"), device)?,
                )?;

                // Dequantize small tensors needed for GDN kernels
                // conv_weight from GGUF: [kernel_size, channels] -> need [channels, 1, kernel_size]
                let conv_weight_raw = ct
                    .tensor(reader, &format!("{prefix}.ssm_conv1d.weight"), device)?
                    .dequantize(device)?;
                let conv_weight = conv_weight_raw.t()?.unsqueeze(1)?;

                let a_log = ct
                    .tensor(reader, &format!("{prefix}.ssm_a"), device)?
                    .dequantize(device)?;
                let dt_bias = ct
                    .tensor(reader, &format!("{prefix}.ssm_dt.bias"), device)?
                    .dequantize(device)?;
                let gdn_norm_weight = ct
                    .tensor(reader, &format!("{prefix}.ssm_norm.weight"), device)?
                    .dequantize(device)?;

                LayerType::LinearAttention(QuantizedGatedDeltaNet {
                    in_proj_qkv,
                    in_proj_z,
                    in_proj_a,
                    in_proj_b,
                    out_proj,
                    conv_weight,
                    a_log,
                    dt_bias,
                    gdn_norm_weight,
                    gdn_norm_bias: None,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim: head_k_dim_gdn,
                    head_v_dim: head_v_dim_gdn,
                    key_dim,
                    value_dim,
                    kv_group_size,
                    gdn_layer_idx: cur_gdn_idx,
                    rms_norm_eps,
                    scale,
                })
            };

            layers.push(DecoderLayer {
                attn,
                mlp,
                input_layernorm: QRmsNorm::from_qtensor(input_layernorm, rms_norm_eps)?,
                post_attention_layernorm: QRmsNorm::from_qtensor(
                    post_attention_layernorm,
                    rms_norm_eps,
                )?,
            });
            reporter.write().set_progress(layer_idx + 1);
        }

        // Initialize MambaCache for linear attention layers
        let num_gdn_layers = gdn_layer_idx;
        let mamba_cache = if num_gdn_layers > 0 {
            MambaCache::new(
                num_gdn_layers,
                1,
                d_conv,
                ssm_conv_kernel,
                num_v_heads,
                head_k_dim_gdn,
                head_v_dim_gdn,
                dtype,
                DType::F32,
                device,
            )?
        } else {
            MambaCache::new(0, 1, 1, 2, 1, 1, 1, dtype, DType::F32, device)?
        };

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            mamba_cache: RwLock::new(mamba_cache),
            cfg,
            dtype,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, false)
    }

    pub fn forward_embedding(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, true)
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let seqlens = resolve_input_seqlens(input_metadata)?;

        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens,
            self.cfg.sliding_window,
            input_metadata.is_prefill,
        );

        let mut xs = self.tok_embeddings.forward(x)?;
        let mut mamba_cache = self.mamba_cache.write();
        let seq_slots = resolve_mamba_seq_slots(
            "GGUFQWen3.5",
            &self.device,
            input_metadata,
            xs.dim(0)?,
            &mut mamba_cache,
        )?;

        let mut kv_cache_idx = 0usize;
        for (_idx, layer) in self.layers.iter().enumerate() {
            let cache = if layer.is_full_attention() {
                if let Some(kv_caches) = kv_caches {
                    let c = &kv_caches[kv_cache_idx];
                    kv_cache_idx += 1;
                    Some((&c.0, &c.1))
                } else {
                    None
                }
            } else {
                None
            };

            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                input_positions,
                cache,
                input_metadata,
                &mut mamba_cache,
                &seq_slots,
            )?;
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }

        let xs = self.norm.forward(&xs)?;

        if return_hidden {
            return Ok(xs);
        }
        self.output.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }

    pub fn release_sequence_state(&self, sequence_id: usize) {
        self.mamba_cache.write().free_slot(sequence_id);
    }

    pub fn ensure_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .ensure_slots_for_sequences(sequence_ids)
    }

    pub fn get_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .get_slots_for_sequences(sequence_ids)
    }

    pub fn has_mamba_slot_for_sequence(&self, sequence_id: usize) -> bool {
        self.mamba_cache.read().get_slot(sequence_id).is_some()
    }

    pub fn lock_mamba_cache_for_graph(&self) -> RwLockWriteGuard<'_, MambaCache> {
        self.mamba_cache.write()
    }

    pub fn preallocate_mamba_cache(&self, max_num_seqs: usize) -> Result<()> {
        self.mamba_cache.write().reserve_capacity(max_num_seqs)
    }

    pub fn set_mamba_prefix_cache_capacity(&self, capacity: usize) {
        self.mamba_cache.write().set_prefix_cache_capacity(capacity);
    }

    pub fn capture_mamba_prefix_state(
        &self,
        seq_id: usize,
        hash: u64,
        preserve: bool,
    ) -> Result<bool> {
        self.mamba_cache
            .write()
            .capture_prefix_state(seq_id, hash, preserve)
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> bool {
        self.mamba_cache.write().has_prefix_state(hash)
    }

    pub fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        self.mamba_cache.write().restore_prefix_state(seq_id, hash)
    }

    pub fn reset_mamba_cache(&self) -> Result<()> {
        self.mamba_cache.write().reset_all()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
