// Original implementation:
// https://github.com/guoqingbao/vllm.rs/blob/main/src/utils/graph.rs
#[cfg(feature = "flashinfer")]
use crate::FlashInferKvParams;
use std::collections::BTreeMap;
use std::ptr;
use std::sync::Arc;

use crate::attention::InputMetadata;
use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtr};
use candle_core::cuda_backend::cudarc::driver::sys::{
    CUgraphInstantiate_flags, CUmemPool_attribute, CUmemoryPool, CUstreamCaptureMode,
    CUstreamCaptureStatus,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{DType, Device, Result, Tensor};
use parking_lot::RwLock;
use std::mem::MaybeUninit;
use tqdm::tqdm;

/// In-place zero fill for a CUDA tensor (equivalent to the upstream `zero_()` method).
fn tensor_zero_(t: &Tensor) -> Result<()> {
    t.zero_set()
}

/// In-place copy from `src` into `dst` at byte offset 0 on the GPU.
/// This performs a raw device-to-device memcpy so that CUDA graph replay
/// sees the data at the same captured addresses.
fn tensor_copy_(dst: &Tensor, src: &Tensor, _offset: usize) -> Result<()> {
    let dst_dev = match dst.device() {
        Device::Cuda(dev) => dev.clone(),
        _ => candle_core::bail!("tensor_copy_ requires CUDA tensors"),
    };
    let stream = dst_dev.cuda_stream();

    let elem_size = dst.dtype().size_in_bytes();
    let src_elems = src.elem_count();

    let (dst_storage, dst_layout) = dst.storage_and_layout();
    let dst_cuda = match &*dst_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("tensor_copy_ requires CUDA tensors"),
    };

    let (src_storage, src_layout) = src.storage_and_layout();
    let src_cuda = match &*src_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("tensor_copy_ requires CUDA tensors"),
    };

    // For the raw memcpy we only need the base device pointers, not typed slices.
    // Use u8 slices to get byte-level pointers.
    let dst_slice = dst_cuda.as_cuda_slice::<u8>()?;
    let src_slice = src_cuda.as_cuda_slice::<u8>()?;

    let src_byte_offset = src_layout.start_offset() * elem_size;
    let dst_byte_offset = dst_layout.start_offset() * elem_size;
    let byte_count = src_elems * elem_size;

    let src_view = src_slice.slice(src_byte_offset..src_byte_offset + byte_count);
    let dst_view = dst_slice.slice(dst_byte_offset..dst_byte_offset + byte_count);

    let (src_ptr, _sg) = src_view.device_ptr(&stream);
    let (dst_ptr, _dg) = dst_view.device_ptr(&stream);

    unsafe {
        sys::cuMemcpyDtoDAsync_v2(
            dst_ptr,
            src_ptr,
            byte_count as _,
            dst_dev.cu_stream(),
        )
        .result()
        .map_err(|e| candle_core::Error::Msg(format!("cuMemcpyDtoDAsync failed: {e:?}")))?;
    }

    Ok(())
}

#[allow(dead_code)]
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: sys::CUstream,
}

impl CudaGraph {
    pub fn begin_capture(stream: sys::CUstream, mode: sys::CUstreamCaptureMode) -> Result<()> {
        unsafe {
            sys::cuStreamBeginCapture_v2(stream, mode)
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("begin_capture failed: {e:?}")))
        }
    }

    pub fn end_capture(
        stream: sys::CUstream,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<CudaGraph> {
        let mut graph = MaybeUninit::uninit();
        let cu_graph = unsafe {
            sys::cuStreamEndCapture(stream, graph.as_mut_ptr())
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuStreamEndCapture failed: {e:?}"))
                })?;
            graph.assume_init()
        };

        let mut graph_exec = MaybeUninit::uninit();
        let cu_graph_exec = unsafe {
            sys::cuGraphInstantiateWithFlags(graph_exec.as_mut_ptr(), cu_graph, flags as u32 as u64)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuGraphInstantiateWithFlags failed: {e:?}"))
                })?;
            graph_exec.assume_init()
        };
        Ok(CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream,
        })
    }

    pub fn capture_status(stream: sys::CUstream) -> Result<sys::CUstreamCaptureStatus> {
        let mut status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        unsafe {
            sys::cuStreamIsCapturing(stream, &mut status)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuGraphInstantiateWithFlags failed: {e:?}"))
                })?;
        }
        Ok(status)
    }

    pub fn launch(&self) -> Result<()> {
        unsafe {
            sys::cuGraphLaunch(self.cu_graph_exec, self.stream)
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("cuGraphLaunch failed: {e:?}")))
        }
    }
}

pub trait CudaGraphModule {
    fn start_capture(&mut self, bs: usize) -> Result<()>;
    fn end_capture(&mut self) -> Result<()>;
    fn replay(&self, bs: usize) -> Result<()>;
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor>;
    fn report_graph_pool_usage(&self) -> Result<()>;
}

pub struct CudaGraphHandle {
    graph: Arc<CudaGraph>,
}

impl CudaGraphHandle {
    pub fn new(graph: Arc<CudaGraph>) -> Self {
        Self { graph }
    }

    pub fn replay(&self) -> Result<()> {
        self.graph
            .launch()
            .map_err(|e| candle_core::Error::Msg(format!("CUDA Graph launch failed: {:?}", e)))?;
        Ok(())
    }
}

pub struct CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
    ) -> Result<Tensor>,
{
    module: M,
    captured_graphs: BTreeMap<usize, CudaGraphHandle>,
    capturing: bool,
    current_bs: Option<usize>,
    device: Arc<CudaDevice>,
    pub pool_handle: RwLock<Option<i64>>,
    captured_bs: Vec<usize>,
}

impl<M> CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
    ) -> Result<Tensor>,
{
    pub fn new(module: M, device: Arc<CudaDevice>) -> Self {
        Self {
            module,
            captured_graphs: BTreeMap::new(),
            capturing: false,
            current_bs: None,
            device,
            pool_handle: RwLock::new(None),
            captured_bs: Vec::new(),
        }
    }

    fn sync_stream(&self) -> Result<()> {
        unsafe {
            sys::cuStreamSynchronize(self.device.cu_stream())
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("cuStreamSynchronize failed: {e:?}")))
        }
    }

    fn create_capture_pool(&self) -> Result<CUmemoryPool> {
        let mut pool: CUmemoryPool = ptr::null_mut();
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, self.device.cu_device())
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuDeviceGetDefaultMemPool failed: {e:?}"))
                })?;

            let handle = pool as *mut std::ffi::c_void as usize as i64;
            *self.pool_handle.write() = Some(handle);

            let threshold: u64 = u64::MAX;
            sys::cuMemPoolSetAttribute(
                    pool,
                    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const _ as _,
                )
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuMemPoolSetAttribute failed: {e:?}"))
                })?;
        }
        Ok(pool)
    }

    fn set_capture_mem_pool(&self) -> Result<()> {
        if self.pool_handle.read().is_some() {
            return Ok(());
        }

        unsafe {
            let status = CudaGraph::capture_status(self.device.cu_stream())?;
            if status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
                let pool = self.create_capture_pool()?;
                sys::cuDeviceSetMemPool(self.device.cu_device(), pool)
                    .result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("cuDeviceSetMemPool failed: {e:?}"))
                    })?;
            }
        }

        Ok(())
    }

    /// Reads a usize attribute from the given CUDA memory pool.
    fn get_mem_pool_attribute(pool: CUmemoryPool, attr: CUmemPool_attribute) -> Result<usize> {
        let mut value: usize = 0;
        unsafe {
            sys::cuMemPoolGetAttribute(pool, attr, &mut value as *mut _ as *mut std::ffi::c_void)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuMemPoolGetAttribute failed: {e:?}"))
                })?;
        }
        Ok(value)
    }

    /// Returns peak memory used (in bytes) from a given CUDA memory pool.
    pub fn get_peak_memory_usage(pool: CUmemoryPool) -> Result<usize> {
        Self::get_mem_pool_attribute(pool, CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)
    }

    /// Returns current memory usage (in bytes) from a given CUDA memory pool.
    pub fn get_current_memory_usage(pool: CUmemoryPool) -> Result<usize> {
        Self::get_mem_pool_attribute(pool, CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
    }

    /// Retrieves the default CUDA memory pool for a device.
    pub fn get_current_mem_pool(&self) -> Result<CUmemoryPool> {
        if self.pool_handle.read().is_some() {
            let pool_handle = self.pool_handle.read().unwrap();
            let pool: CUmemoryPool = pool_handle as usize as *mut sys::CUmemPoolHandle_st;
            Ok(pool)
        } else {
            candle_core::bail!("Memory pool for graph is not init!")
        }
    }
}

impl<M> CudaGraphModule for CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
    ) -> Result<Tensor>,
{
    fn start_capture(&mut self, bs: usize) -> Result<()> {
        self.capturing = true;
        self.current_bs = Some(bs);
        self.sync_stream()?;
        self.set_capture_mem_pool()?;
        CudaGraph::begin_capture(
            self.device.cu_stream(),
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )?;
        Ok(())
    }

    fn end_capture(&mut self) -> Result<()> {
        self.capturing = false;
        let bs = self.current_bs.take().unwrap();

        let graph = CudaGraph::end_capture(
            self.device.cu_stream(),
            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        )?;
        self.captured_graphs
            .insert(bs, CudaGraphHandle::new(Arc::new(graph)));

        self.captured_bs.push(bs);
        self.captured_bs.sort_unstable(); // keep it sorted for binary search
        self.sync_stream()?;
        Ok(())
    }

    fn replay(&self, bs: usize) -> Result<()> {
        if let Some(&next_bs) = self.captured_bs.iter().find(|&&x| x >= bs) {
            if let Some(graph) = self.captured_graphs.get(&next_bs) {
                self.sync_stream()?;
                graph.replay()?;
                self.sync_stream()
            } else {
                candle_core::bail!("No suitable graph is found for batch size {}!", next_bs)
            }
        } else {
            candle_core::bail!("Batch size {} is not captured in graph!", bs)
        }
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        (self.module)(input_ids, positions, kv_caches, input_metadata)
    }

    fn report_graph_pool_usage(&self) -> Result<()> {
        let pool = self.get_current_mem_pool()?;
        let peak = Self::get_peak_memory_usage(pool)?;
        let current = Self::get_current_memory_usage(pool)?;
        tracing::info!(
            "Default pool usage: {:.2} MB (current), {:.2} MB (peak)",
            current as f64 / 1e6,
            peak as f64 / 1e6
        );
        Ok(())
    }
}

pub struct GraphCaptureVars {
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub mamba_slot_mapping: Tensor,
    pub slot_mapping: Tensor,
    pub context_lens: Tensor,
    pub block_tables: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indptr: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indices: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_last_len: Tensor,
    pub outputs: BTreeMap<usize, Tensor>,
}

pub struct GraphCapturer<M: CudaGraphModule> {
    pub model: M,
    pub graph_bs: Vec<usize>,
    pub graph_vars: Option<GraphCaptureVars>,
    pub max_num_seqs: usize,
    pub max_model_len: usize,
    pub block_size: usize,
    pub hidden_size: usize,
    pub device: Option<Device>,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_kv_params: Option<FlashInferKvParams>,
}

pub fn planned_graph_capture_batches(max_num_seqs: usize) -> Vec<usize> {
    let small_max = max_num_seqs.clamp(1, 15);
    let mut graph_bs = (1..=small_max).collect::<Vec<_>>();
    if max_num_seqs >= 16 {
        graph_bs.extend((16..=max_num_seqs.min(32)).step_by(16));
    }
    graph_bs
}

impl<M: CudaGraphModule> GraphCapturer<M> {
    pub fn new(
        model: M,
        max_num_seqs: usize,
        max_model_len: usize,
        block_size: usize,
        hidden_size: usize,
        #[cfg(feature = "flashinfer")] flashinfer_kv_params: &Option<FlashInferKvParams>,
    ) -> Self {
        let graph_bs = planned_graph_capture_batches(max_num_seqs);
        println!("The following batches for capture: {:?}", graph_bs);

        Self {
            model,
            graph_bs,
            graph_vars: None,
            max_num_seqs,
            max_model_len,
            block_size,
            hidden_size,
            device: None,
            #[cfg(feature = "flashinfer")]
            flashinfer_kv_params: *flashinfer_kv_params,
        }
    }

    pub fn capture(
        &mut self,
        device: &Device,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
    ) -> Result<()> {
        self.device = Some(device.clone());
        let max_bs = self.graph_bs[self.graph_bs.len() - 1];
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;

        let input_ids = Tensor::zeros((max_bs,), DType::U32, device)?;
        let positions = Tensor::zeros((max_bs,), DType::I64, device)?;
        let mamba_slot_mapping = Tensor::from_vec(
            (0..max_bs).map(|i| i as i64).collect::<Vec<_>>(),
            (max_bs,),
            device,
        )?;
        let slot_mapping = Tensor::zeros((max_bs,), DType::I64, device)?;
        let context_lens = Tensor::zeros((max_bs,), DType::U32, device)?;
        let block_tables = Tensor::zeros((max_bs, max_num_blocks), DType::U32, device)?;
        #[cfg(feature = "flashinfer")]
        let (flashinfer_indptr, flashinfer_indices, flashinfer_last_len, last_len_host) = {
            let mut indptr = Vec::with_capacity(max_bs + 1);
            indptr.push(0u32);
            let mut indices = Vec::with_capacity(max_bs * max_num_blocks);
            for _ in 0..max_bs {
                for i in 0..max_num_blocks {
                    indices.push(i as u32);
                }
                indptr.push(indices.len() as u32);
            }
            let last = if self.max_model_len == 0 {
                0u32
            } else {
                ((self.max_model_len - 1) % self.block_size + 1) as u32
            };
            let last_len = vec![last; max_bs];
            (
                Tensor::from_vec(indptr, (max_bs + 1,), device)?,
                Tensor::from_vec(indices, (max_bs * max_num_blocks,), device)?,
                Tensor::from_vec(last_len.clone(), (max_bs,), device)?,
                last_len,
            )
        };
        let mut outputs = BTreeMap::<usize, Tensor>::new();
        for i in tqdm(0..self.graph_bs.len()).desc(Some("Graph capturing")) {
            let bs = self.graph_bs[self.graph_bs.len() - i - 1];
            let input_ids_bs = input_ids.narrow(0, 0, bs)?;
            let positions_bs = positions.narrow(0, 0, bs)?;
            #[cfg(feature = "flashinfer")]
            let flashinfer_metadata = {
                let mut indptr_host = Vec::with_capacity(bs + 1);
                indptr_host.push(0u32);
                for i in 0..bs {
                    indptr_host.push(((i + 1) * max_num_blocks) as u32);
                }

                let (decode_plan_info, kv_len_arr_host) =
                    if let Some(params) = self.flashinfer_kv_params {
                        let mut kv_len_arr_host_bs = Vec::with_capacity(bs);
                        for i in 0..bs {
                            let num_pages = indptr_host[i + 1] - indptr_host[i];
                            if num_pages == 0 {
                                kv_len_arr_host_bs.push(0);
                            } else {
                                let full = (num_pages - 1) * params.page_size as u32;
                                kv_len_arr_host_bs.push(full + last_len_host[i]);
                            }
                        }
                        let kv_len_arr_host = kv_len_arr_host_bs.clone();
                        (
                            Some(crate::attention::flashinfer::decode_plan(
                                device,
                                params.kv_dtype,
                                params.out_dtype,
                                &indptr_host,
                                Some(&last_len_host[..bs]),
                                Some(kv_len_arr_host_bs.as_slice()),
                                bs,
                                params.num_qo_heads,
                                params.num_kv_heads,
                                params.head_dim,
                                params.page_size,
                                true,
                            )?),
                            Some(kv_len_arr_host),
                        )
                    } else {
                        (None, None)
                    };

                Some(crate::attention::FlashInferMetadata {
                    indptr: flashinfer_indptr.narrow(0, 0, bs + 1)?,
                    indptr_host,
                    indices: flashinfer_indices.narrow(0, 0, bs * max_num_blocks)?,
                    last_len: flashinfer_last_len.narrow(0, 0, bs)?,
                    last_len_host: Some(last_len_host[..bs].to_vec()),
                    kv_len_arr_host,
                    cu_seqlens_q_host: None,
                    total_num_rows: None,
                    batch_indices: None,
                    positions: None,
                    use_cuda_graph: true,
                    decode_plan_info,
                })
            };
            #[cfg(not(feature = "flashinfer"))]
            let flashinfer_metadata = None;
            let input_metadata = InputMetadata {
                is_prefill: false,
                sequence_ids: None,
                mamba_slot_mapping: Some(mamba_slot_mapping.narrow(0, 0, bs)?),
                slot_mapping: slot_mapping.narrow(0, 0, bs)?,
                block_tables: Some(block_tables.narrow(0, 0, bs)?),
                context_lens: Some(context_lens.narrow(0, 0, bs)?),
                cu_seqlens_q: None,
                cu_seqlens_k: None,
                max_seqlen_q: 0,
                max_seqlen_k: 0,
                max_context_len: self.max_model_len,
                disable_flash_attn: None,
                seqlens: None,
                flashinfer_metadata,
            };
            self.model.start_capture(bs)?;
            let out =
                self.model
                    .forward(&input_ids_bs, &positions_bs, kv_caches, &input_metadata)?;
            self.model.end_capture()?;
            outputs.insert(bs, out);
        }
        let _ = self.model.report_graph_pool_usage();

        tracing::warn!("Captured batches {:?}", outputs.keys());

        self.graph_vars = Some(GraphCaptureVars {
            input_ids,
            positions,
            mamba_slot_mapping,
            slot_mapping,
            context_lens,
            block_tables,
            #[cfg(feature = "flashinfer")]
            flashinfer_indptr,
            #[cfg(feature = "flashinfer")]
            flashinfer_indices,
            #[cfg(feature = "flashinfer")]
            flashinfer_last_len,
            outputs,
        });

        Ok(())
    }

    pub fn is_captured(&self, batch: usize) -> bool {
        self.graph_vars.is_some()
            && self
                .graph_vars
                .as_ref()
                .unwrap()
                .outputs
                .keys()
                .find(|&&x| x >= batch)
                .is_some()
    }

    pub fn is_exact_captured(&self, batch: usize) -> bool {
        self.graph_vars.is_some()
            && self
                .graph_vars
                .as_ref()
                .unwrap()
                .outputs
                .contains_key(&batch)
    }

    pub fn replay(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        if input_metadata.is_prefill {
            candle_core::bail!("Graph replay is not used for prefill!")
        }
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;
        let input_batch = input_ids.dim(0)?;
        let require_exact_batch = input_metadata.mamba_slot_mapping.is_some()
            || input_metadata.flashinfer_metadata.is_some();
        if let Some(graph_vars) = &self.graph_vars {
            let selected_batch = if require_exact_batch {
                graph_vars
                    .outputs
                    .keys()
                    .find(|&&x| x == input_batch)
                    .copied()
            } else {
                graph_vars
                    .outputs
                    .keys()
                    .find(|&&x| x >= input_batch)
                    .copied()
            };
            if let Some(batch) = selected_batch {
                tensor_zero_(&graph_vars.input_ids)?;
                tensor_copy_(&graph_vars.input_ids, &input_ids, 0)?;
                tensor_zero_(&graph_vars.positions)?;
                tensor_copy_(&graph_vars.positions, &positions, 0)?;

                if let Some(ms_mapping) = input_metadata.mamba_slot_mapping.as_ref() {
                    tensor_zero_(&graph_vars.mamba_slot_mapping)?;
                    tensor_copy_(&graph_vars.mamba_slot_mapping, ms_mapping, 0)?;
                } else {
                    tensor_zero_(&graph_vars.mamba_slot_mapping)?;
                }

                let s_mapping = input_metadata.slot_mapping.as_ref();
                tensor_zero_(&graph_vars.slot_mapping)?;
                tensor_copy_(&graph_vars.slot_mapping, s_mapping, 0)?;

                let c_lens = input_metadata.context_lens.as_ref().unwrap();
                tensor_zero_(&graph_vars.context_lens)?;
                tensor_copy_(&graph_vars.context_lens, c_lens, 0)?;

                let b_tables = input_metadata.block_tables.as_ref().unwrap();
                let padded_table = b_tables
                    .pad_with_zeros(1, 0, max_num_blocks - b_tables.dim(1)?)?
                    .contiguous()?;

                tensor_zero_(&graph_vars.block_tables)?;
                tensor_copy_(&graph_vars.block_tables, &padded_table, 0)?;

                #[cfg(feature = "flashinfer")]
                if let Some(fm) = &input_metadata.flashinfer_metadata {
                    let mut indptr_host = fm.indptr_host.clone();
                    if input_batch == batch {
                        tensor_zero_(&graph_vars.flashinfer_indptr)?;
                        tensor_copy_(&graph_vars.flashinfer_indptr, &fm.indptr, 0)?;
                    } else {
                        let last = *indptr_host.last().unwrap_or(&0);
                        for _ in (input_batch + 1)..=batch {
                            indptr_host.push(last);
                        }
                        let indptr_padded = Tensor::from_vec(
                            indptr_host.clone(),
                            (batch + 1,),
                            graph_vars.input_ids.device(),
                        )?;
                        tensor_copy_(&graph_vars.flashinfer_indptr, &indptr_padded, 0)?;
                    }

                    tensor_zero_(&graph_vars.flashinfer_last_len)?;
                    tensor_copy_(&graph_vars.flashinfer_last_len, &fm.last_len, 0)?;

                    tensor_zero_(&graph_vars.flashinfer_indices)?;
                    tensor_copy_(&graph_vars.flashinfer_indices, &fm.indices, 0)?;

                    if let Some(params) = self.flashinfer_kv_params {
                        let dev = self
                            .device
                            .as_ref()
                            .ok_or_else(|| candle_core::Error::msg("graph device is missing"))?;
                        let _ = crate::attention::flashinfer::decode_plan(
                            dev,
                            params.kv_dtype,
                            params.out_dtype,
                            &indptr_host,
                            fm.last_len_host.as_deref(),
                            fm.kv_len_arr_host.as_deref(),
                            batch,
                            params.num_qo_heads,
                            params.num_kv_heads,
                            params.head_dim,
                            params.page_size,
                            fm.use_cuda_graph,
                        )?;
                    }
                }

                let result = self.model.replay(batch);
                if result.is_err() {
                    eprintln!("Error when replaying graph {:?}", result);
                }

                graph_vars.outputs[&batch]
                    .narrow(0, 0, input_batch)?
                    .contiguous()
            } else {
                candle_core::bail!("Input batch {} is not captured!", input_batch)
            }
        } else {
            candle_core::bail!("Graph is not captured!")
        }
    }
}

unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

pub type ModelFn = dyn for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
    ) -> Result<Tensor>
    + Send
    + Sync;

pub type CudaGraphFn = Box<
    dyn for<'a> Fn(
            &'a Tensor,
            &'a Tensor,
            Option<&'a Vec<(Tensor, Tensor)>>,
            &'a InputMetadata,
        ) -> Result<Tensor>
        + Send
        + Sync,
>;
