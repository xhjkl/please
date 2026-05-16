use super::{
    ATTN_VALUES, EXPERTS, GATE_UP_VALUES, HIDDEN_SIZE, KV_VALUES, LM_HEAD_TOP1_BLOCK_SIZE,
    LOCAL_WINDOW_TOKENS, MAX_KV_CACHE_PROBE_TOKENS, MAX_PREFILL_PROBE_TOKENS,
    MXFP4_BYTES_PER_GROUP, MXFP4_GROUPS, Q_HEADS,
};

#[cfg(target_os = "macos")]
pub(crate) mod platform {
    use super::{
        ATTN_VALUES, EXPERTS, GATE_UP_VALUES, HIDDEN_SIZE, KV_VALUES, LM_HEAD_TOP1_BLOCK_SIZE,
        LOCAL_WINDOW_TOKENS, MAX_KV_CACHE_PROBE_TOKENS, MAX_PREFILL_PROBE_TOKENS,
        MXFP4_BYTES_PER_GROUP, MXFP4_GROUPS, Q_HEADS,
    };
    use crate::runtime_core::ExpertScore;
    use eyre::{Result, eyre};
    use objc2::ffi::NSUInteger;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLFunction, MTLLibrary, MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::mem::size_of_val;
    use std::ptr::NonNull;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    const THREADS_PER_GROUP: u64 = 256;

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {}

    type MetalDevice = ProtocolObject<dyn MTLDevice>;
    type MetalCommandQueue = ProtocolObject<dyn MTLCommandQueue>;
    type MetalCommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
    type MetalComputeCommandEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
    type MetalComputePipelineState = ProtocolObject<dyn MTLComputePipelineState>;
    type MetalFunction = ProtocolObject<dyn MTLFunction>;
    type MetalLibrary = ProtocolObject<dyn MTLLibrary>;
    type MetalBuffer = ProtocolObject<dyn MTLBuffer>;

    pub struct MetalBatch<'a> {
        context: &'a MetalContext,
        command_buffer: Retained<MetalCommandBuffer>,
    }

    pub struct MetalContext {
        device: Retained<MetalDevice>,
        queue: Retained<MetalCommandQueue>,
        profile_enabled: AtomicBool,
        gpu_time_ns: Mutex<u128>,
        partial_sum_squares: Retained<MetalComputePipelineState>,
        apply_rms_norm: Retained<MetalComputePipelineState>,
        apply_rms_norm_from_partials: Retained<MetalComputePipelineState>,
        rms_norm_batch: Retained<MetalComputePipelineState>,
        bf16_matvec: Retained<MetalComputePipelineState>,
        bf16_matvec_batch: Retained<MetalComputePipelineState>,
        bf16_matvec_logits: Retained<MetalComputePipelineState>,
        top1_logits_blocks: Retained<MetalComputePipelineState>,
        top1_logits_final: Retained<MetalComputePipelineState>,
        topk_logits: Retained<MetalComputePipelineState>,
        embedding_lookup_bf16: Retained<MetalComputePipelineState>,
        embedding_lookup_bf16_batch: Retained<MetalComputePipelineState>,
        rope_row: Retained<MetalComputePipelineState>,
        rope_batch: Retained<MetalComputePipelineState>,
        single_token_attention: Retained<MetalComputePipelineState>,
        sequence_attention: Retained<MetalComputePipelineState>,
        kv_cache_decode_attention: Retained<MetalComputePipelineState>,
        write_f32_slot: Retained<MetalComputePipelineState>,
        write_f32_slots_batch: Retained<MetalComputePipelineState>,
        read_f32_slot: Retained<MetalComputePipelineState>,
        vector_add: Retained<MetalComputePipelineState>,
        top4_softmax: Retained<MetalComputePipelineState>,
        mxfp4_matvec: Retained<MetalComputePipelineState>,
        mxfp4_top4_gate_swiglu: Retained<MetalComputePipelineState>,
        mxfp4_top4_down_weighted: Retained<MetalComputePipelineState>,
        swiglu: Retained<MetalComputePipelineState>,
        weighted_sum4: Retained<MetalComputePipelineState>,
    }

    #[derive(Clone)]
    pub struct Bf16MatrixBuffer {
        buffer: Retained<MetalBuffer>,
        rows: usize,
        cols: usize,
    }

    impl Bf16MatrixBuffer {
        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn cols(&self) -> usize {
            self.cols
        }
    }

    #[derive(Clone)]
    pub struct F32VectorBuffer {
        buffer: Retained<MetalBuffer>,
        len: usize,
    }

    #[derive(Clone)]
    pub struct U8Buffer {
        buffer: Retained<MetalBuffer>,
        len: usize,
    }

    #[derive(Clone)]
    pub struct U32Buffer {
        buffer: Retained<MetalBuffer>,
        len: usize,
    }

    const KERNEL_SOURCE: &str = include_str!("kernels.metal");

    impl MetalContext {
        pub fn new() -> Result<Self> {
            let device =
                MTLCreateSystemDefaultDevice().ok_or_else(|| eyre!("no Metal device found"))?;
            let queue = device.new_command_queue();
            let options = MTLCompileOptions::new();
            let source = NSString::from_str(KERNEL_SOURCE);
            let library = device
                .newLibraryWithSource_options_error(&source, Some(&options))
                .map_err(|error| eyre!("compile Metal kernels: {error:?}"))?;

            Ok(Self {
                profile_enabled: AtomicBool::new(false),
                gpu_time_ns: Mutex::new(0),
                partial_sum_squares: pipeline(&device, &library, "partial_sum_squares")?,
                apply_rms_norm: pipeline(&device, &library, "apply_rms_norm")?,
                apply_rms_norm_from_partials: pipeline(
                    &device,
                    &library,
                    "apply_rms_norm_from_partials",
                )?,
                rms_norm_batch: pipeline(&device, &library, "rms_norm_batch")?,
                bf16_matvec: pipeline(&device, &library, "bf16_matvec")?,
                bf16_matvec_batch: pipeline(&device, &library, "bf16_matvec_batch")?,
                bf16_matvec_logits: pipeline(&device, &library, "bf16_matvec_logits")?,
                top1_logits_blocks: pipeline(&device, &library, "top1_logits_blocks")?,
                top1_logits_final: pipeline(&device, &library, "top1_logits_final")?,
                topk_logits: pipeline(&device, &library, "topk_logits")?,
                embedding_lookup_bf16: pipeline(&device, &library, "embedding_lookup_bf16")?,
                embedding_lookup_bf16_batch: pipeline(
                    &device,
                    &library,
                    "embedding_lookup_bf16_batch",
                )?,
                rope_row: pipeline(&device, &library, "rope_row")?,
                rope_batch: pipeline(&device, &library, "rope_batch")?,
                single_token_attention: pipeline(&device, &library, "single_token_attention")?,
                sequence_attention: pipeline(&device, &library, "sequence_attention")?,
                kv_cache_decode_attention: pipeline(
                    &device,
                    &library,
                    "kv_cache_decode_attention",
                )?,
                write_f32_slot: pipeline(&device, &library, "write_f32_slot")?,
                write_f32_slots_batch: pipeline(&device, &library, "write_f32_slots_batch")?,
                read_f32_slot: pipeline(&device, &library, "read_f32_slot")?,
                vector_add: pipeline(&device, &library, "vector_add")?,
                top4_softmax: pipeline(&device, &library, "top4_softmax")?,
                mxfp4_matvec: pipeline(&device, &library, "mxfp4_matvec")?,
                mxfp4_top4_gate_swiglu: pipeline(&device, &library, "mxfp4_top4_gate_swiglu")?,
                mxfp4_top4_down_weighted: pipeline(&device, &library, "mxfp4_top4_down_weighted")?,
                swiglu: pipeline(&device, &library, "swiglu")?,
                weighted_sum4: pipeline(&device, &library, "weighted_sum4")?,
                device,
                queue,
            })
        }

        pub fn take_gpu_time_ns(&self) -> u128 {
            let mut gpu_time_ns = self.gpu_time_ns.lock().unwrap();
            let value = *gpu_time_ns;
            *gpu_time_ns = 0;
            value
        }

        pub fn set_profile_enabled(&self, enabled: bool) {
            self.profile_enabled.store(enabled, Ordering::Relaxed);
        }

        fn finish_command_buffer(&self, command_buffer: &MetalCommandBuffer) -> u128 {
            command_buffer.commit();
            command_buffer.wait_until_completed();
            if !self.profile_enabled.load(Ordering::Relaxed) {
                return 0;
            }
            let gpu_time_ns = command_buffer_gpu_time_ns(command_buffer);
            if gpu_time_ns > 0 {
                *self.gpu_time_ns.lock().unwrap() += gpu_time_ns;
            }
            gpu_time_ns
        }

        pub fn begin_batch(&self) -> MetalBatch<'_> {
            MetalBatch {
                context: self,
                command_buffer: self.queue.new_command_buffer(),
            }
        }

        pub fn alloc_f32_vector(&self, len: usize) -> Result<F32VectorBuffer> {
            Ok(F32VectorBuffer {
                buffer: self.device.new_buffer(
                    (len * std::mem::size_of::<f32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
                len,
            })
        }

        pub fn alloc_u32_buffer(&self, len: usize) -> Result<U32Buffer> {
            Ok(U32Buffer {
                buffer: self.device.new_buffer(
                    (len * std::mem::size_of::<u32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
                len,
            })
        }

        pub fn upload_u32_buffer(&self, values: &[u32]) -> Result<U32Buffer> {
            Ok(U32Buffer {
                buffer: buffer_with_data(&self.device, values),
                len: values.len(),
            })
        }

        pub fn read_f32_vector(&self, buffer: &F32VectorBuffer) -> Vec<f32> {
            read_buffer::<f32>(&buffer.buffer, buffer.len)
        }

        pub fn read_u32_buffer(&self, buffer: &U32Buffer) -> Vec<u32> {
            read_buffer::<u32>(&buffer.buffer, buffer.len)
        }

        pub fn rms_norm(&self, x: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
            if x.len() != weight.len() {
                return Err(eyre!(
                    "RMSNorm input has {} values but weight has {} values",
                    x.len(),
                    weight.len()
                ));
            }
            if x.is_empty() {
                return Ok(Vec::new());
            }

            let n = x.len() as u32;
            let groups = (x.len() as u64).div_ceil(THREADS_PER_GROUP);
            let x_buffer = buffer_with_data(&self.device, x);
            let weight_buffer = buffer_with_data(&self.device, weight);
            let partial_buffer = self.device.new_buffer(
                groups * std::mem::size_of::<f32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let n_buffer = buffer_with_data(&self.device, &[n]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.partial_sum_squares);
            encoder.set_buffer(0, Some(&x_buffer), 0);
            encoder.set_buffer(1, Some(&partial_buffer), 0);
            encoder.set_buffer(2, Some(&n_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(groups as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            let partials = read_buffer::<f32>(&partial_buffer, groups as usize);
            let sum_squares = partials.iter().map(|value| *value as f64).sum::<f64>();
            let mean_square = sum_squares / x.len() as f64;
            let scale = (mean_square + 1e-5).sqrt().recip() as f32;

            let scale_buffer = buffer_with_data(&self.device, &[scale]);
            let out_buffer = self
                .device
                .new_buffer(size_of_val(x) as u64, MTLResourceOptions::StorageModeShared);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.apply_rms_norm);
            encoder.set_buffer(0, Some(&x_buffer), 0);
            encoder.set_buffer(1, Some(&weight_buffer), 0);
            encoder.set_buffer(2, Some(&out_buffer), 0);
            encoder.set_buffer(3, Some(&scale_buffer), 0);
            encoder.set_buffer(4, Some(&n_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(x.len() as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, x.len()))
        }

        pub fn bf16_matvec(
            &self,
            weight: &[u16],
            rows: usize,
            cols: usize,
            input: &[f32],
            bias: &[f32],
        ) -> Result<Vec<f32>> {
            if rows.checked_mul(cols) != Some(weight.len()) {
                return Err(eyre!(
                    "BF16 matvec weight has {} values, expected rows * cols = {} * {}",
                    weight.len(),
                    rows,
                    cols
                ));
            }
            if input.len() != cols {
                return Err(eyre!(
                    "BF16 matvec input has {} values, expected {cols}",
                    input.len()
                ));
            }
            if bias.len() != rows {
                return Err(eyre!(
                    "BF16 matvec bias has {} values, expected {rows}",
                    bias.len()
                ));
            }
            if rows == 0 {
                return Ok(Vec::new());
            }

            let weight_buffer = buffer_with_data(&self.device, weight);
            let input_buffer = buffer_with_data(&self.device, input);
            let bias_buffer = buffer_with_data(&self.device, bias);
            let out_buffer = self.device.new_buffer(
                (rows * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let rows = rows as u32;
            let cols = cols as u32;
            let rows_buffer = buffer_with_data(&self.device, &[rows]);
            let cols_buffer = buffer_with_data(&self.device, &[cols]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.bf16_matvec);
            encoder.set_buffer(0, Some(&weight_buffer), 0);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(&bias_buffer), 0);
            encoder.set_buffer(3, Some(&out_buffer), 0);
            encoder.set_buffer(4, Some(&rows_buffer), 0);
            encoder.set_buffer(5, Some(&cols_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, rows as usize))
        }

        pub fn bf16_matrix_matvec(
            &self,
            weight: &Bf16MatrixBuffer,
            input: &[f32],
            bias: &F32VectorBuffer,
        ) -> Result<Vec<f32>> {
            let rows = weight.rows;
            let cols = weight.cols();
            if input.len() != cols {
                return Err(eyre!(
                    "BF16 resident matvec input has {} values, expected {cols}",
                    input.len()
                ));
            }
            if bias.len != rows {
                return Err(eyre!(
                    "BF16 resident matvec bias has {} values, expected {rows}",
                    bias.len
                ));
            }
            if rows == 0 {
                return Ok(Vec::new());
            }

            let input_buffer = buffer_with_data(&self.device, input);
            let out_buffer = self.device.new_buffer(
                (rows * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let rows = rows as u32;
            let cols = cols as u32;
            let rows_buffer = buffer_with_data(&self.device, &[rows]);
            let cols_buffer = buffer_with_data(&self.device, &[cols]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.bf16_matvec);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&out_buffer), 0);
            encoder.set_buffer(4, Some(&rows_buffer), 0);
            encoder.set_buffer(5, Some(&cols_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, rows as usize))
        }

        pub fn upload_f32_vector(&self, values: &[f32]) -> Result<F32VectorBuffer> {
            Ok(F32VectorBuffer {
                buffer: buffer_with_data(&self.device, values),
                len: values.len(),
            })
        }

        pub fn upload_u8_buffer(&self, values: &[u8]) -> Result<U8Buffer> {
            Ok(U8Buffer {
                buffer: buffer_with_data(&self.device, values),
                len: values.len(),
            })
        }

        pub fn upload_bf16_matrix(
            &self,
            weight: &[u16],
            rows: usize,
            cols: usize,
        ) -> Result<Bf16MatrixBuffer> {
            if rows.checked_mul(cols) != Some(weight.len()) {
                return Err(eyre!(
                    "BF16 matrix has {} values, expected rows * cols = {} * {}",
                    weight.len(),
                    rows,
                    cols
                ));
            }
            Ok(Bf16MatrixBuffer {
                buffer: buffer_with_data(&self.device, weight),
                rows,
                cols,
            })
        }

        pub fn bf16_matrix_topk(
            &self,
            weight: &Bf16MatrixBuffer,
            input: &[f32],
            k: usize,
        ) -> Result<Vec<(usize, f32)>> {
            let rows = weight.rows;
            let cols = weight.cols;
            if input.len() != cols {
                return Err(eyre!(
                    "BF16 matvec top-k input has {} values, expected {cols}",
                    input.len()
                ));
            }
            if rows == 0 {
                return Err(eyre!("BF16 matvec top-k needs at least one row"));
            }
            if k == 0 || k > 8 {
                return Err(eyre!("BF16 matvec top-k supports k in 1..=8, got {k}"));
            }
            if k > rows {
                return Err(eyre!("BF16 matvec top-k k {k} exceeds rows {rows}"));
            }

            let input_buffer = buffer_with_data(&self.device, input);
            let logits_buffer = self.device.new_buffer(
                (rows * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let rows = rows as u32;
            let cols = cols as u32;
            let k = k as u32;
            let rows_buffer = buffer_with_data(&self.device, &[rows]);
            let cols_buffer = buffer_with_data(&self.device, &[cols]);
            let k_buffer = buffer_with_data(&self.device, &[k]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.bf16_matvec_logits);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(&logits_buffer), 0);
            encoder.set_buffer(3, Some(&rows_buffer), 0);
            encoder.set_buffer(4, Some(&cols_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            let indices_buffer = self.device.new_buffer(
                (k as usize * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let values_buffer = self.device.new_buffer(
                (k as usize * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.topk_logits);
            encoder.set_buffer(0, Some(&logits_buffer), 0);
            encoder.set_buffer(1, Some(&indices_buffer), 0);
            encoder.set_buffer(2, Some(&values_buffer), 0);
            encoder.set_buffer(3, Some(&rows_buffer), 0);
            encoder.set_buffer(4, Some(&k_buffer), 0);
            encoder.dispatch_threads(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            let indices = read_buffer::<u32>(&indices_buffer, k as usize);
            let values = read_buffer::<f32>(&values_buffer, k as usize);
            Ok(indices
                .into_iter()
                .zip(values)
                .map(|(index, value)| (index as usize, value))
                .collect())
        }

        pub fn rope_row(&self, row: &[f32], heads: usize, position: usize) -> Result<Vec<f32>> {
            let expected = heads
                .checked_mul(64)
                .ok_or_else(|| eyre!("RoPE row expected length overflow"))?;
            if row.len() != expected {
                return Err(eyre!(
                    "RoPE row has {} values, expected {expected}",
                    row.len()
                ));
            }
            if row.is_empty() {
                return Ok(Vec::new());
            }

            let input_buffer = buffer_with_data(&self.device, row);
            let out_buffer = self.device.new_buffer(
                size_of_val(row) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let heads = heads as u32;
            let position = position as u32;
            let heads_buffer = buffer_with_data(&self.device, &[heads]);
            let position_buffer = buffer_with_data(&self.device, &[position]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.rope_row);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&out_buffer), 0);
            encoder.set_buffer(2, Some(&heads_buffer), 0);
            encoder.set_buffer(3, Some(&position_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(row.len() as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, row.len()))
        }

        pub fn single_token_attention(
            &self,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            sinks: &[f32],
        ) -> Result<Vec<f32>> {
            if q.len() != 64 * 64 {
                return Err(eyre!("attention q has {} values, expected 4096", q.len()));
            }
            if k.len() != 8 * 64 {
                return Err(eyre!("attention k has {} values, expected 512", k.len()));
            }
            if v.len() != 8 * 64 {
                return Err(eyre!("attention v has {} values, expected 512", v.len()));
            }
            if sinks.len() != 64 {
                return Err(eyre!(
                    "attention sinks has {} values, expected 64",
                    sinks.len()
                ));
            }

            let q_buffer = buffer_with_data(&self.device, q);
            let k_buffer = buffer_with_data(&self.device, k);
            let v_buffer = buffer_with_data(&self.device, v);
            let sinks_buffer = buffer_with_data(&self.device, sinks);
            let out_buffer = self.device.new_buffer(
                (q.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.single_token_attention);
            encoder.set_buffer(0, Some(&q_buffer), 0);
            encoder.set_buffer(1, Some(&k_buffer), 0);
            encoder.set_buffer(2, Some(&v_buffer), 0);
            encoder.set_buffer(3, Some(&sinks_buffer), 0);
            encoder.set_buffer(4, Some(&out_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(64, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, q.len()))
        }

        pub fn sequence_attention(
            &self,
            layer: usize,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            sinks: &[f32],
            seq_len: usize,
        ) -> Result<Vec<f32>> {
            if seq_len == 0 {
                return Err(eyre!("sequence attention needs at least one token"));
            }
            if seq_len > MAX_PREFILL_PROBE_TOKENS {
                return Err(eyre!(
                    "sequence attention supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {seq_len}"
                ));
            }

            let q_len = seq_len
                .checked_mul(ATTN_VALUES)
                .ok_or_else(|| eyre!("sequence attention q length overflow"))?;
            let kv_len = seq_len
                .checked_mul(KV_VALUES)
                .ok_or_else(|| eyre!("sequence attention kv length overflow"))?;
            if q.len() != q_len {
                return Err(eyre!(
                    "sequence attention q has {} values, expected {q_len}",
                    q.len()
                ));
            }
            if k.len() != kv_len {
                return Err(eyre!(
                    "sequence attention k has {} values, expected {kv_len}",
                    k.len()
                ));
            }
            if v.len() != kv_len {
                return Err(eyre!(
                    "sequence attention v has {} values, expected {kv_len}",
                    v.len()
                ));
            }
            if sinks.len() != 64 {
                return Err(eyre!(
                    "sequence attention sinks has {} values, expected 64",
                    sinks.len()
                ));
            }

            let q_buffer = buffer_with_data(&self.device, q);
            let k_buffer = buffer_with_data(&self.device, k);
            let v_buffer = buffer_with_data(&self.device, v);
            let sinks_buffer = buffer_with_data(&self.device, sinks);
            let out_buffer = self.device.new_buffer(
                (q.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let seq_len = seq_len as u32;
            let layer = layer as u32;
            let seq_len_buffer = buffer_with_data(&self.device, &[seq_len]);
            let layer_buffer = buffer_with_data(&self.device, &[layer]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.sequence_attention);
            encoder.set_buffer(0, Some(&q_buffer), 0);
            encoder.set_buffer(1, Some(&k_buffer), 0);
            encoder.set_buffer(2, Some(&v_buffer), 0);
            encoder.set_buffer(3, Some(&sinks_buffer), 0);
            encoder.set_buffer(4, Some(&out_buffer), 0);
            encoder.set_buffer(5, Some(&seq_len_buffer), 0);
            encoder.set_buffer(6, Some(&layer_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(64, seq_len as NSUInteger, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, q.len()))
        }

        pub fn kv_cache_decode_attention(
            &self,
            layer: usize,
            query_position: usize,
            cache_start_position: usize,
            q: &[f32],
            k_cache: &[f32],
            v_cache: &[f32],
            sinks: &[f32],
        ) -> Result<Vec<f32>> {
            if q.len() != ATTN_VALUES {
                return Err(eyre!(
                    "KV-cache decode q has {} values, expected {ATTN_VALUES}",
                    q.len()
                ));
            }
            if k_cache.len() != v_cache.len() {
                return Err(eyre!(
                    "KV-cache K/V length mismatch: k={}, v={}",
                    k_cache.len(),
                    v_cache.len()
                ));
            }
            if k_cache.is_empty() || k_cache.len() % KV_VALUES != 0 {
                return Err(eyre!(
                    "KV-cache has {} K values, expected a non-empty multiple of {KV_VALUES}",
                    k_cache.len()
                ));
            }
            if sinks.len() != 64 {
                return Err(eyre!(
                    "KV-cache decode sinks has {} values, expected 64",
                    sinks.len()
                ));
            }
            if cache_start_position > query_position {
                return Err(eyre!(
                    "KV-cache start position {cache_start_position} exceeds query position {query_position}"
                ));
            }

            let cache_len = k_cache.len() / KV_VALUES;
            let expected_cache_len = query_position - cache_start_position + 1;
            if cache_len != expected_cache_len {
                return Err(eyre!(
                    "KV-cache has {cache_len} positions, expected {expected_cache_len} for positions {cache_start_position}..={query_position}"
                ));
            }

            let mut effective_key_start = cache_start_position;
            if layer % 2 == 0 && query_position + 1 > LOCAL_WINDOW_TOKENS {
                effective_key_start =
                    effective_key_start.max(query_position + 1 - LOCAL_WINDOW_TOKENS);
            }
            let key_count = query_position + 1 - effective_key_start;
            if key_count > MAX_KV_CACHE_PROBE_TOKENS {
                return Err(eyre!(
                    "KV-cache decode probe supports at most {MAX_KV_CACHE_PROBE_TOKENS} keys, got {key_count}"
                ));
            }

            let q_buffer = buffer_with_data(&self.device, q);
            let k_buffer = buffer_with_data(&self.device, k_cache);
            let v_buffer = buffer_with_data(&self.device, v_cache);
            let sinks_buffer = buffer_with_data(&self.device, sinks);
            let out_buffer = self.device.new_buffer(
                (q.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let layer = layer as u32;
            let query_position = query_position as u32;
            let cache_start_position = cache_start_position as u32;
            let cache_len = cache_len as u32;
            let layer_buffer = buffer_with_data(&self.device, &[layer]);
            let query_position_buffer = buffer_with_data(&self.device, &[query_position]);
            let cache_start_position_buffer =
                buffer_with_data(&self.device, &[cache_start_position]);
            let cache_len_buffer = buffer_with_data(&self.device, &[cache_len]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.kv_cache_decode_attention);
            encoder.set_buffer(0, Some(&q_buffer), 0);
            encoder.set_buffer(1, Some(&k_buffer), 0);
            encoder.set_buffer(2, Some(&v_buffer), 0);
            encoder.set_buffer(3, Some(&sinks_buffer), 0);
            encoder.set_buffer(4, Some(&out_buffer), 0);
            encoder.set_buffer(5, Some(&layer_buffer), 0);
            encoder.set_buffer(6, Some(&query_position_buffer), 0);
            encoder.set_buffer(7, Some(&cache_start_position_buffer), 0);
            encoder.set_buffer(8, Some(&cache_len_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(64, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, q.len()))
        }

        pub fn vector_add(&self, left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
            if left.len() != right.len() {
                return Err(eyre!(
                    "vector add length mismatch: left {}, right {}",
                    left.len(),
                    right.len()
                ));
            }
            if left.is_empty() {
                return Ok(Vec::new());
            }

            let left_buffer = buffer_with_data(&self.device, left);
            let right_buffer = buffer_with_data(&self.device, right);
            let out_buffer = self.device.new_buffer(
                (left.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let n = left.len() as u32;
            let n_buffer = buffer_with_data(&self.device, &[n]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.vector_add);
            encoder.set_buffer(0, Some(&left_buffer), 0);
            encoder.set_buffer(1, Some(&right_buffer), 0);
            encoder.set_buffer(2, Some(&out_buffer), 0);
            encoder.set_buffer(3, Some(&n_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(left.len() as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, left.len()))
        }

        pub fn top4_softmax(&self, logits: &[f32]) -> Result<Vec<ExpertScore>> {
            if logits.len() < 4 {
                return Err(eyre!(
                    "top4_softmax needs at least 4 logits, got {}",
                    logits.len()
                ));
            }

            let logits_buffer = buffer_with_data(&self.device, logits);
            let indices_buffer = self.device.new_buffer(
                (4 * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let selected_logits_buffer = self.device.new_buffer(
                (4 * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let weights_buffer = self.device.new_buffer(
                (4 * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let n = logits.len() as u32;
            let n_buffer = buffer_with_data(&self.device, &[n]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.top4_softmax);
            encoder.set_buffer(0, Some(&logits_buffer), 0);
            encoder.set_buffer(1, Some(&indices_buffer), 0);
            encoder.set_buffer(2, Some(&selected_logits_buffer), 0);
            encoder.set_buffer(3, Some(&weights_buffer), 0);
            encoder.set_buffer(4, Some(&n_buffer), 0);
            encoder.dispatch_threads(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            let indices = read_buffer::<u32>(&indices_buffer, 4);
            let selected_logits = read_buffer::<f32>(&selected_logits_buffer, 4);
            let weights = read_buffer::<f32>(&weights_buffer, 4);
            Ok(indices
                .into_iter()
                .zip(selected_logits)
                .zip(weights)
                .map(|((index, logit), weight)| ExpertScore {
                    index: index as usize,
                    logit,
                    weight,
                })
                .collect())
        }

        pub fn mxfp4_matvec(
            &self,
            blocks: &[u8],
            scales: &[u8],
            rows: usize,
            input: &[f32],
            bias: &[f32],
        ) -> Result<Vec<f32>> {
            if input.len() % 32 != 0 {
                return Err(eyre!(
                    "MXFP4 input has {} values, expected a multiple of 32",
                    input.len()
                ));
            }
            let groups = input.len() / 32;
            let expected_blocks = rows
                .checked_mul(groups)
                .and_then(|value| value.checked_mul(16))
                .ok_or_else(|| eyre!("MXFP4 block length overflow"))?;
            let expected_scales = rows
                .checked_mul(groups)
                .ok_or_else(|| eyre!("MXFP4 scale length overflow"))?;
            if blocks.len() != expected_blocks {
                return Err(eyre!(
                    "MXFP4 blocks has {} bytes, expected {expected_blocks}",
                    blocks.len()
                ));
            }
            if scales.len() != expected_scales {
                return Err(eyre!(
                    "MXFP4 scales has {} bytes, expected {expected_scales}",
                    scales.len()
                ));
            }
            if bias.len() != rows {
                return Err(eyre!(
                    "MXFP4 bias has {} values, expected {rows}",
                    bias.len()
                ));
            }
            if rows == 0 {
                return Ok(Vec::new());
            }

            let blocks_buffer = buffer_with_data(&self.device, blocks);
            let scales_buffer = buffer_with_data(&self.device, scales);
            let input_buffer = buffer_with_data(&self.device, input);
            let bias_buffer = buffer_with_data(&self.device, bias);
            let out_buffer = self.device.new_buffer(
                (rows * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let rows = rows as u32;
            let groups = groups as u32;
            let rows_buffer = buffer_with_data(&self.device, &[rows]);
            let groups_buffer = buffer_with_data(&self.device, &[groups]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.mxfp4_matvec);
            encoder.set_buffer(0, Some(&blocks_buffer), 0);
            encoder.set_buffer(1, Some(&scales_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&bias_buffer), 0);
            encoder.set_buffer(4, Some(&out_buffer), 0);
            encoder.set_buffer(5, Some(&rows_buffer), 0);
            encoder.set_buffer(6, Some(&groups_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, rows as usize))
        }

        pub fn mxfp4_matvec_resident(
            &self,
            blocks: &U8Buffer,
            scales: &U8Buffer,
            rows: usize,
            input: &[f32],
            bias: &F32VectorBuffer,
        ) -> Result<Vec<f32>> {
            if input.len() % 32 != 0 {
                return Err(eyre!(
                    "MXFP4 resident input has {} values, expected a multiple of 32",
                    input.len()
                ));
            }
            let groups = input.len() / 32;
            let expected_blocks = rows
                .checked_mul(groups)
                .and_then(|value| value.checked_mul(16))
                .ok_or_else(|| eyre!("MXFP4 resident block length overflow"))?;
            let expected_scales = rows
                .checked_mul(groups)
                .ok_or_else(|| eyre!("MXFP4 resident scale length overflow"))?;
            if blocks.len != expected_blocks {
                return Err(eyre!(
                    "MXFP4 resident blocks has {} bytes, expected {expected_blocks}",
                    blocks.len
                ));
            }
            if scales.len != expected_scales {
                return Err(eyre!(
                    "MXFP4 resident scales has {} bytes, expected {expected_scales}",
                    scales.len
                ));
            }
            if bias.len != rows {
                return Err(eyre!(
                    "MXFP4 resident bias has {} values, expected {rows}",
                    bias.len
                ));
            }
            if rows == 0 {
                return Ok(Vec::new());
            }

            let input_buffer = buffer_with_data(&self.device, input);
            let out_buffer = self.device.new_buffer(
                (rows * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let rows = rows as u32;
            let groups = groups as u32;
            let rows_buffer = buffer_with_data(&self.device, &[rows]);
            let groups_buffer = buffer_with_data(&self.device, &[groups]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.mxfp4_matvec);
            encoder.set_buffer(0, Some(&blocks.buffer), 0);
            encoder.set_buffer(1, Some(&scales.buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&bias.buffer), 0);
            encoder.set_buffer(4, Some(&out_buffer), 0);
            encoder.set_buffer(5, Some(&rows_buffer), 0);
            encoder.set_buffer(6, Some(&groups_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, rows as usize))
        }

        pub fn swiglu(&self, values: &[f32]) -> Result<Vec<f32>> {
            if values.len() % 2 != 0 {
                return Err(eyre!(
                    "SwiGLU input has {} values, expected an even length",
                    values.len()
                ));
            }
            if values.is_empty() {
                return Ok(Vec::new());
            }

            let input_buffer = buffer_with_data(&self.device, values);
            let out_len = values.len() / 2;
            let out_buffer = self.device.new_buffer(
                (out_len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_len = out_len as u32;
            let n_buffer = buffer_with_data(&self.device, &[out_len]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.swiglu);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&out_buffer), 0);
            encoder.set_buffer(2, Some(&n_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(out_len as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, out_len as usize))
        }

        pub fn weighted_sum4(&self, vectors: [&[f32]; 4], weights: [f32; 4]) -> Result<Vec<f32>> {
            let n = vectors[0].len();
            for (index, vector) in vectors.iter().enumerate() {
                if vector.len() != n {
                    return Err(eyre!(
                        "weighted_sum4 vector {index} has {} values, expected {n}",
                        vector.len()
                    ));
                }
            }
            if n == 0 {
                return Ok(Vec::new());
            }

            let mut packed = Vec::with_capacity(n * 4);
            for vector in vectors {
                packed.extend_from_slice(vector);
            }
            let vectors_buffer = buffer_with_data(&self.device, &packed);
            let weights_buffer = buffer_with_data(&self.device, &weights);
            let out_buffer = self.device.new_buffer(
                (n * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let n = n as u32;
            let n_buffer = buffer_with_data(&self.device, &[n]);

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.weighted_sum4);
            encoder.set_buffer(0, Some(&vectors_buffer), 0);
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&out_buffer), 0);
            encoder.set_buffer(3, Some(&n_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(n as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            self.finish_command_buffer(&command_buffer);

            Ok(read_buffer::<f32>(&out_buffer, n as usize))
        }
    }

    impl<'a> MetalBatch<'a> {
        pub fn finish(self) -> u128 {
            self.context.finish_command_buffer(&self.command_buffer)
        }

        pub fn embedding_lookup_bf16_into(
            &self,
            weight: &Bf16MatrixBuffer,
            token: usize,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if token >= weight.rows {
                return Err(eyre!(
                    "embedding token {token} exceeds embedding rows {}",
                    weight.rows
                ));
            }
            if out.len != weight.cols {
                return Err(eyre!(
                    "embedding output has {} values, expected {}",
                    out.len,
                    weight.cols
                ));
            }

            let token = token as u32;
            let cols = weight.cols as u32;
            let token_buffer = buffer_with_data(&self.context.device, &[token]);
            let cols_buffer = buffer_with_data(&self.context.device, &[cols]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.embedding_lookup_bf16);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&out.buffer), 0);
            encoder.set_buffer(2, Some(&token_buffer), 0);
            encoder.set_buffer(3, Some(&cols_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(weight.cols as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn embedding_lookup_bf16_batch_into(
            &self,
            weight: &Bf16MatrixBuffer,
            tokens: &U32Buffer,
            token_count: usize,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if token_count == 0 {
                return Ok(());
            }
            if tokens.len < token_count {
                return Err(eyre!(
                    "batched embedding has {} token ids, expected at least {token_count}",
                    tokens.len
                ));
            }
            let expected = token_count
                .checked_mul(weight.cols)
                .ok_or_else(|| eyre!("batched embedding output length overflow"))?;
            if out.len < expected {
                return Err(eyre!(
                    "batched embedding output has {} values, expected at least {expected}",
                    out.len
                ));
            }

            let cols = weight.cols as u32;
            let token_count = token_count as u32;
            let cols_buffer = buffer_with_data(&self.context.device, &[cols]);
            let token_count_buffer = buffer_with_data(&self.context.device, &[token_count]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.embedding_lookup_bf16_batch);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&tokens.buffer), 0);
            encoder.set_buffer(2, Some(&out.buffer), 0);
            encoder.set_buffer(3, Some(&cols_buffer), 0);
            encoder.set_buffer(4, Some(&token_count_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(expected as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn rms_norm_into(
            &self,
            input: &F32VectorBuffer,
            weight: &F32VectorBuffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if input.len != weight.len || input.len != out.len {
                return Err(eyre!(
                    "RMSNorm buffer length mismatch: input {}, weight {}, out {}",
                    input.len,
                    weight.len,
                    out.len
                ));
            }
            if input.len == 0 {
                return Ok(());
            }

            let n = input.len as u32;
            let groups = (input.len as u64).div_ceil(THREADS_PER_GROUP);
            let partial_buffer = self.context.device.new_buffer(
                groups * std::mem::size_of::<f32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let n_buffer = buffer_with_data(&self.context.device, &[n]);
            let groups_buffer = buffer_with_data(&self.context.device, &[groups as u32]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.partial_sum_squares);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&partial_buffer), 0);
            encoder.set_buffer(2, Some(&n_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(groups as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.apply_rms_norm_from_partials);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&weight.buffer), 0);
            encoder.set_buffer(2, Some(&partial_buffer), 0);
            encoder.set_buffer(3, Some(&out.buffer), 0);
            encoder.set_buffer(4, Some(&n_buffer), 0);
            encoder.set_buffer(5, Some(&groups_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(input.len as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn rms_norm_batch_into(
            &self,
            input: &F32VectorBuffer,
            weight: &F32VectorBuffer,
            out: &F32VectorBuffer,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            if rows == 0 {
                return Ok(());
            }
            if weight.len != cols {
                return Err(eyre!(
                    "batched RMSNorm weight has {} values, expected {cols}",
                    weight.len
                ));
            }
            let expected = rows
                .checked_mul(cols)
                .ok_or_else(|| eyre!("batched RMSNorm length overflow"))?;
            if input.len < expected || out.len < expected {
                return Err(eyre!(
                    "batched RMSNorm buffer length mismatch: input {}, out {}, expected at least {expected}",
                    input.len,
                    out.len
                ));
            }

            let rows = rows as u32;
            let cols = cols as u32;
            let rows_buffer = buffer_with_data(&self.context.device, &[rows]);
            let cols_buffer = buffer_with_data(&self.context.device, &[cols]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.rms_norm_batch);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&weight.buffer), 0);
            encoder.set_buffer(2, Some(&out.buffer), 0);
            encoder.set_buffer(3, Some(&rows_buffer), 0);
            encoder.set_buffer(4, Some(&cols_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn bf16_matrix_matvec_into(
            &self,
            weight: &Bf16MatrixBuffer,
            input: &F32VectorBuffer,
            bias: &F32VectorBuffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if input.len != weight.cols {
                return Err(eyre!(
                    "BF16 resident matvec input has {} values, expected {}",
                    input.len,
                    weight.cols
                ));
            }
            if bias.len != weight.rows || out.len != weight.rows {
                return Err(eyre!(
                    "BF16 resident matvec row mismatch: bias {}, out {}, rows {}",
                    bias.len,
                    out.len,
                    weight.rows
                ));
            }

            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let rows_buffer = buffer_with_data(&self.context.device, &[rows]);
            let cols_buffer = buffer_with_data(&self.context.device, &[cols]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.bf16_matvec);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input.buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&out.buffer), 0);
            encoder.set_buffer(4, Some(&rows_buffer), 0);
            encoder.set_buffer(5, Some(&cols_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn bf16_matrix_matvec_batch_into(
            &self,
            weight: &Bf16MatrixBuffer,
            input: &F32VectorBuffer,
            bias: &F32VectorBuffer,
            out: &F32VectorBuffer,
            batch_rows: usize,
        ) -> Result<()> {
            if batch_rows == 0 {
                return Ok(());
            }
            let input_expected = batch_rows
                .checked_mul(weight.cols)
                .ok_or_else(|| eyre!("batched BF16 matvec input length overflow"))?;
            let out_expected = batch_rows
                .checked_mul(weight.rows)
                .ok_or_else(|| eyre!("batched BF16 matvec output length overflow"))?;
            if input.len < input_expected {
                return Err(eyre!(
                    "batched BF16 matvec input has {} values, expected at least {input_expected}",
                    input.len
                ));
            }
            if bias.len != weight.rows || out.len < out_expected {
                return Err(eyre!(
                    "batched BF16 matvec shape mismatch: bias {}, out {}, rows {}, expected output at least {out_expected}",
                    bias.len,
                    out.len,
                    weight.rows
                ));
            }

            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let batch_rows = batch_rows as u32;
            let rows_buffer = buffer_with_data(&self.context.device, &[rows]);
            let cols_buffer = buffer_with_data(&self.context.device, &[cols]);
            let batch_rows_buffer = buffer_with_data(&self.context.device, &[batch_rows]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.bf16_matvec_batch);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input.buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&out.buffer), 0);
            encoder.set_buffer(4, Some(&rows_buffer), 0);
            encoder.set_buffer(5, Some(&cols_buffer), 0);
            encoder.set_buffer(6, Some(&batch_rows_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, batch_rows as NSUInteger, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn bf16_matrix_topk_into(
            &self,
            weight: &Bf16MatrixBuffer,
            input: &F32VectorBuffer,
            logits: &F32VectorBuffer,
            indices: &U32Buffer,
            values: &F32VectorBuffer,
            k: usize,
        ) -> Result<()> {
            if input.len != weight.cols {
                return Err(eyre!(
                    "BF16 top-k input has {} values, expected {}",
                    input.len,
                    weight.cols
                ));
            }
            if logits.len != weight.rows {
                return Err(eyre!(
                    "BF16 top-k logits scratch has {} values, expected {}",
                    logits.len,
                    weight.rows
                ));
            }
            if k == 0 || k > 8 || indices.len < k || values.len < k {
                return Err(eyre!(
                    "BF16 top-k needs k in 1..=8 with output room, got {k}"
                ));
            }

            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let k = k as u32;
            let rows_buffer = buffer_with_data(&self.context.device, &[rows]);
            let cols_buffer = buffer_with_data(&self.context.device, &[cols]);
            let k_buffer = buffer_with_data(&self.context.device, &[k]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.bf16_matvec_logits);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input.buffer), 0);
            encoder.set_buffer(2, Some(&logits.buffer), 0);
            encoder.set_buffer(3, Some(&rows_buffer), 0);
            encoder.set_buffer(4, Some(&cols_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.topk_logits);
            encoder.set_buffer(0, Some(&logits.buffer), 0);
            encoder.set_buffer(1, Some(&indices.buffer), 0);
            encoder.set_buffer(2, Some(&values.buffer), 0);
            encoder.set_buffer(3, Some(&rows_buffer), 0);
            encoder.set_buffer(4, Some(&k_buffer), 0);
            encoder.dispatch_threads(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
            encoder.end_encoding();
            Ok(())
        }

        pub fn bf16_matrix_top1_into(
            &self,
            weight: &Bf16MatrixBuffer,
            input: &F32VectorBuffer,
            logits: &F32VectorBuffer,
            block_indices: &U32Buffer,
            block_values: &F32VectorBuffer,
            out_index: &U32Buffer,
            out_value: &F32VectorBuffer,
        ) -> Result<()> {
            if input.len != weight.cols {
                return Err(eyre!(
                    "BF16 top-1 input has {} values, expected {}",
                    input.len,
                    weight.cols
                ));
            }
            if logits.len != weight.rows {
                return Err(eyre!(
                    "BF16 top-1 logits scratch has {} values, expected {}",
                    logits.len,
                    weight.rows
                ));
            }
            let blocks = weight.rows.div_ceil(LM_HEAD_TOP1_BLOCK_SIZE);
            if block_indices.len < blocks || block_values.len < blocks {
                return Err(eyre!(
                    "BF16 top-1 block scratch has indices {}/values {}, expected at least {blocks}",
                    block_indices.len,
                    block_values.len
                ));
            }
            if out_index.len < 1 || out_value.len < 1 {
                return Err(eyre!("BF16 top-1 output buffers need one slot"));
            }

            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let blocks = blocks as u32;
            let rows_buffer = buffer_with_data(&self.context.device, &[rows]);
            let cols_buffer = buffer_with_data(&self.context.device, &[cols]);
            let blocks_buffer = buffer_with_data(&self.context.device, &[blocks]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.bf16_matvec_logits);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input.buffer), 0);
            encoder.set_buffer(2, Some(&logits.buffer), 0);
            encoder.set_buffer(3, Some(&rows_buffer), 0);
            encoder.set_buffer(4, Some(&cols_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.top1_logits_blocks);
            encoder.set_buffer(0, Some(&logits.buffer), 0);
            encoder.set_buffer(1, Some(&block_indices.buffer), 0);
            encoder.set_buffer(2, Some(&block_values.buffer), 0);
            encoder.set_buffer(3, Some(&rows_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(blocks as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.top1_logits_final);
            encoder.set_buffer(0, Some(&block_indices.buffer), 0);
            encoder.set_buffer(1, Some(&block_values.buffer), 0);
            encoder.set_buffer(2, Some(&out_index.buffer), 0);
            encoder.set_buffer(3, Some(&out_value.buffer), 0);
            encoder.set_buffer(4, Some(&blocks_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn rope_row_into(
            &self,
            input: &F32VectorBuffer,
            out: &F32VectorBuffer,
            heads: usize,
            position: usize,
        ) -> Result<()> {
            let expected = heads
                .checked_mul(64)
                .ok_or_else(|| eyre!("RoPE row expected length overflow"))?;
            if input.len != expected || out.len != expected {
                return Err(eyre!(
                    "RoPE length mismatch: input {}, out {}, expected {expected}",
                    input.len,
                    out.len
                ));
            }

            let heads = heads as u32;
            let position = position as u32;
            let heads_buffer = buffer_with_data(&self.context.device, &[heads]);
            let position_buffer = buffer_with_data(&self.context.device, &[position]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.rope_row);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&out.buffer), 0);
            encoder.set_buffer(2, Some(&heads_buffer), 0);
            encoder.set_buffer(3, Some(&position_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(expected as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn rope_batch_into(
            &self,
            input: &F32VectorBuffer,
            out: &F32VectorBuffer,
            heads: usize,
            start_position: usize,
            rows: usize,
        ) -> Result<()> {
            if rows == 0 {
                return Ok(());
            }
            let width = heads
                .checked_mul(64)
                .ok_or_else(|| eyre!("batched RoPE row width overflow"))?;
            let expected = rows
                .checked_mul(width)
                .ok_or_else(|| eyre!("batched RoPE length overflow"))?;
            if input.len < expected || out.len < expected {
                return Err(eyre!(
                    "batched RoPE length mismatch: input {}, out {}, expected at least {expected}",
                    input.len,
                    out.len
                ));
            }

            let heads = heads as u32;
            let start_position = start_position as u32;
            let rows = rows as u32;
            let heads_buffer = buffer_with_data(&self.context.device, &[heads]);
            let start_position_buffer = buffer_with_data(&self.context.device, &[start_position]);
            let rows_buffer = buffer_with_data(&self.context.device, &[rows]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.rope_batch);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&out.buffer), 0);
            encoder.set_buffer(2, Some(&heads_buffer), 0);
            encoder.set_buffer(3, Some(&start_position_buffer), 0);
            encoder.set_buffer(4, Some(&rows_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(expected as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn write_f32_slot_into(
            &self,
            input: &F32VectorBuffer,
            output: &F32VectorBuffer,
            slot: usize,
            width: usize,
        ) -> Result<()> {
            if input.len != width {
                return Err(eyre!(
                    "slot write input has {} values, expected {width}",
                    input.len
                ));
            }
            let required = slot
                .checked_add(1)
                .and_then(|slots| slots.checked_mul(width))
                .ok_or_else(|| eyre!("slot write length overflow"))?;
            if output.len < required {
                return Err(eyre!(
                    "slot write output has {} values, needs at least {required}",
                    output.len
                ));
            }

            let slot = slot as u32;
            let width = width as u32;
            let slot_buffer = buffer_with_data(&self.context.device, &[slot]);
            let width_buffer = buffer_with_data(&self.context.device, &[width]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.write_f32_slot);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            encoder.set_buffer(2, Some(&slot_buffer), 0);
            encoder.set_buffer(3, Some(&width_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(width as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn write_f32_slots_batch_into(
            &self,
            input: &F32VectorBuffer,
            output: &F32VectorBuffer,
            start_slot: usize,
            slots: usize,
            width: usize,
        ) -> Result<()> {
            if slots == 0 {
                return Ok(());
            }
            let input_expected = slots
                .checked_mul(width)
                .ok_or_else(|| eyre!("batched slot write input length overflow"))?;
            if input.len < input_expected {
                return Err(eyre!(
                    "batched slot write input has {} values, expected at least {input_expected}",
                    input.len
                ));
            }
            let output_expected = start_slot
                .checked_add(slots)
                .and_then(|slot_count| slot_count.checked_mul(width))
                .ok_or_else(|| eyre!("batched slot write output length overflow"))?;
            if output.len < output_expected {
                return Err(eyre!(
                    "batched slot write output has {} values, needs at least {output_expected}",
                    output.len
                ));
            }

            let start_slot = start_slot as u32;
            let slots = slots as u32;
            let width = width as u32;
            let start_slot_buffer = buffer_with_data(&self.context.device, &[start_slot]);
            let slots_buffer = buffer_with_data(&self.context.device, &[slots]);
            let width_buffer = buffer_with_data(&self.context.device, &[width]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.write_f32_slots_batch);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            encoder.set_buffer(2, Some(&start_slot_buffer), 0);
            encoder.set_buffer(3, Some(&slots_buffer), 0);
            encoder.set_buffer(4, Some(&width_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(input_expected as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn read_f32_slot_into(
            &self,
            input: &F32VectorBuffer,
            slot: usize,
            width: usize,
            output: &F32VectorBuffer,
        ) -> Result<()> {
            if output.len != width {
                return Err(eyre!(
                    "slot read output has {} values, expected {width}",
                    output.len
                ));
            }
            let required = slot
                .checked_add(1)
                .and_then(|slots| slots.checked_mul(width))
                .ok_or_else(|| eyre!("slot read length overflow"))?;
            if input.len < required {
                return Err(eyre!(
                    "slot read input has {} values, needs at least {required}",
                    input.len
                ));
            }

            let slot = slot as u32;
            let width = width as u32;
            let slot_buffer = buffer_with_data(&self.context.device, &[slot]);
            let width_buffer = buffer_with_data(&self.context.device, &[width]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.read_f32_slot);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            encoder.set_buffer(2, Some(&slot_buffer), 0);
            encoder.set_buffer(3, Some(&width_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(width as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn sequence_attention_into(
            &self,
            layer: usize,
            q: &F32VectorBuffer,
            k: &F32VectorBuffer,
            v: &F32VectorBuffer,
            sinks: &F32VectorBuffer,
            out: &F32VectorBuffer,
            seq_len: usize,
        ) -> Result<()> {
            if seq_len == 0 {
                return Ok(());
            }
            let q_len = seq_len
                .checked_mul(ATTN_VALUES)
                .ok_or_else(|| eyre!("resident sequence attention q length overflow"))?;
            let kv_len = seq_len
                .checked_mul(KV_VALUES)
                .ok_or_else(|| eyre!("resident sequence attention kv length overflow"))?;
            if q.len < q_len || out.len < q_len {
                return Err(eyre!(
                    "resident sequence attention q/out length mismatch: q {}, out {}, expected at least {q_len}",
                    q.len,
                    out.len
                ));
            }
            if k.len < kv_len || v.len < kv_len {
                return Err(eyre!(
                    "resident sequence attention K/V length mismatch: k {}, v {}, expected at least {kv_len}",
                    k.len,
                    v.len
                ));
            }
            if sinks.len != Q_HEADS {
                return Err(eyre!(
                    "resident sequence attention sinks has {} values, expected {Q_HEADS}",
                    sinks.len
                ));
            }

            let seq_len = seq_len as u32;
            let layer = layer as u32;
            let seq_len_buffer = buffer_with_data(&self.context.device, &[seq_len]);
            let layer_buffer = buffer_with_data(&self.context.device, &[layer]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.sequence_attention);
            encoder.set_buffer(0, Some(&q.buffer), 0);
            encoder.set_buffer(1, Some(&k.buffer), 0);
            encoder.set_buffer(2, Some(&v.buffer), 0);
            encoder.set_buffer(3, Some(&sinks.buffer), 0);
            encoder.set_buffer(4, Some(&out.buffer), 0);
            encoder.set_buffer(5, Some(&seq_len_buffer), 0);
            encoder.set_buffer(6, Some(&layer_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(Q_HEADS as NSUInteger, seq_len as NSUInteger, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn kv_cache_decode_attention_into(
            &self,
            layer: usize,
            query_position: usize,
            cache_start_position: usize,
            cache_len: usize,
            q: &F32VectorBuffer,
            k_cache: &F32VectorBuffer,
            v_cache: &F32VectorBuffer,
            sinks: &F32VectorBuffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if q.len != ATTN_VALUES || out.len != ATTN_VALUES {
                return Err(eyre!(
                    "KV-cache decode q/out length mismatch: q {}, out {}, expected {ATTN_VALUES}",
                    q.len,
                    out.len
                ));
            }
            if k_cache.len != v_cache.len || k_cache.len < cache_len * KV_VALUES {
                return Err(eyre!(
                    "KV-cache resident K/V length mismatch: k {}, v {}, cache_len {cache_len}",
                    k_cache.len,
                    v_cache.len
                ));
            }
            if sinks.len != Q_HEADS {
                return Err(eyre!(
                    "KV-cache decode sinks has {} values, expected {Q_HEADS}",
                    sinks.len
                ));
            }
            if cache_start_position > query_position {
                return Err(eyre!(
                    "KV-cache start position {cache_start_position} exceeds query position {query_position}"
                ));
            }
            let effective_key_start = if layer % 2 == 0 && query_position + 1 > LOCAL_WINDOW_TOKENS
            {
                cache_start_position.max(query_position + 1 - LOCAL_WINDOW_TOKENS)
            } else {
                cache_start_position
            };
            let key_count = query_position + 1 - effective_key_start;
            let _ = key_count;

            let layer = layer as u32;
            let query_position = query_position as u32;
            let cache_start_position = cache_start_position as u32;
            let cache_len = cache_len as u32;
            let layer_buffer = buffer_with_data(&self.context.device, &[layer]);
            let query_position_buffer = buffer_with_data(&self.context.device, &[query_position]);
            let cache_start_position_buffer =
                buffer_with_data(&self.context.device, &[cache_start_position]);
            let cache_len_buffer = buffer_with_data(&self.context.device, &[cache_len]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.kv_cache_decode_attention);
            encoder.set_buffer(0, Some(&q.buffer), 0);
            encoder.set_buffer(1, Some(&k_cache.buffer), 0);
            encoder.set_buffer(2, Some(&v_cache.buffer), 0);
            encoder.set_buffer(3, Some(&sinks.buffer), 0);
            encoder.set_buffer(4, Some(&out.buffer), 0);
            encoder.set_buffer(5, Some(&layer_buffer), 0);
            encoder.set_buffer(6, Some(&query_position_buffer), 0);
            encoder.set_buffer(7, Some(&cache_start_position_buffer), 0);
            encoder.set_buffer(8, Some(&cache_len_buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(Q_HEADS as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn vector_add_into(
            &self,
            left: &F32VectorBuffer,
            right: &F32VectorBuffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if left.len != right.len || left.len != out.len {
                return Err(eyre!(
                    "vector add length mismatch: left {}, right {}, out {}",
                    left.len,
                    right.len,
                    out.len
                ));
            }

            let n = left.len as u32;
            let n_buffer = buffer_with_data(&self.context.device, &[n]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.vector_add);
            encoder.set_buffer(0, Some(&left.buffer), 0);
            encoder.set_buffer(1, Some(&right.buffer), 0);
            encoder.set_buffer(2, Some(&out.buffer), 0);
            encoder.set_buffer(3, Some(&n_buffer), 0);
            encoder.dispatch_threads(
                mtl_size(left.len as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        pub fn top4_softmax_into(
            &self,
            logits: &F32VectorBuffer,
            indices: &U32Buffer,
            selected_logits: &F32VectorBuffer,
            weights: &F32VectorBuffer,
        ) -> Result<()> {
            if logits.len < 4 || indices.len < 4 || selected_logits.len < 4 || weights.len < 4 {
                return Err(eyre!(
                    "top4_softmax needs logits>=4 and output room for 4 values"
                ));
            }

            let n = logits.len as u32;
            let n_buffer = buffer_with_data(&self.context.device, &[n]);

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.top4_softmax);
            encoder.set_buffer(0, Some(&logits.buffer), 0);
            encoder.set_buffer(1, Some(&indices.buffer), 0);
            encoder.set_buffer(2, Some(&selected_logits.buffer), 0);
            encoder.set_buffer(3, Some(&weights.buffer), 0);
            encoder.set_buffer(4, Some(&n_buffer), 0);
            encoder.dispatch_threads(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
            encoder.end_encoding();
            Ok(())
        }

        pub fn mxfp4_top4_gate_swiglu_into(
            &self,
            blocks: &U8Buffer,
            scales: &U8Buffer,
            bias: &Bf16MatrixBuffer,
            input: &F32VectorBuffer,
            top_indices: &U32Buffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            let expected_blocks = EXPERTS
                .checked_mul(GATE_UP_VALUES)
                .and_then(|value| value.checked_mul(MXFP4_GROUPS))
                .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
                .ok_or_else(|| eyre!("MXFP4 gate-up slab block length overflow"))?;
            let expected_scales = EXPERTS
                .checked_mul(GATE_UP_VALUES)
                .and_then(|value| value.checked_mul(MXFP4_GROUPS))
                .ok_or_else(|| eyre!("MXFP4 gate-up slab scale length overflow"))?;
            if blocks.len != expected_blocks || scales.len != expected_scales {
                return Err(eyre!(
                    "MXFP4 gate-up slab length mismatch: blocks {}, scales {}, expected {expected_blocks}/{expected_scales}",
                    blocks.len,
                    scales.len
                ));
            }
            if bias.rows != EXPERTS || bias.cols != GATE_UP_VALUES {
                return Err(eyre!(
                    "MXFP4 gate-up bias shape is {}x{}, expected {EXPERTS}x{GATE_UP_VALUES}",
                    bias.rows,
                    bias.cols
                ));
            }
            if input.len != HIDDEN_SIZE || top_indices.len < 4 || out.len != 4 * HIDDEN_SIZE {
                return Err(eyre!(
                    "MXFP4 gate-up fused shape mismatch: input {}, top_indices {}, out {}",
                    input.len,
                    top_indices.len,
                    out.len
                ));
            }

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.mxfp4_top4_gate_swiglu);
            encoder.set_buffer(0, Some(&blocks.buffer), 0);
            encoder.set_buffer(1, Some(&scales.buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&input.buffer), 0);
            encoder.set_buffer(4, Some(&top_indices.buffer), 0);
            encoder.set_buffer(5, Some(&out.buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(HIDDEN_SIZE as NSUInteger, 4, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn mxfp4_top4_down_weighted_into(
            &self,
            blocks: &U8Buffer,
            scales: &U8Buffer,
            bias: &Bf16MatrixBuffer,
            expert_acts: &F32VectorBuffer,
            top_indices: &U32Buffer,
            top_weights: &F32VectorBuffer,
            residual: &F32VectorBuffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            let expected_blocks = EXPERTS
                .checked_mul(HIDDEN_SIZE)
                .and_then(|value| value.checked_mul(MXFP4_GROUPS))
                .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
                .ok_or_else(|| eyre!("MXFP4 down slab block length overflow"))?;
            let expected_scales = EXPERTS
                .checked_mul(HIDDEN_SIZE)
                .and_then(|value| value.checked_mul(MXFP4_GROUPS))
                .ok_or_else(|| eyre!("MXFP4 down slab scale length overflow"))?;
            if blocks.len != expected_blocks || scales.len != expected_scales {
                return Err(eyre!(
                    "MXFP4 down slab length mismatch: blocks {}, scales {}, expected {expected_blocks}/{expected_scales}",
                    blocks.len,
                    scales.len
                ));
            }
            if bias.rows != EXPERTS || bias.cols != HIDDEN_SIZE {
                return Err(eyre!(
                    "MXFP4 down bias shape is {}x{}, expected {EXPERTS}x{HIDDEN_SIZE}",
                    bias.rows,
                    bias.cols
                ));
            }
            if expert_acts.len != 4 * HIDDEN_SIZE
                || top_indices.len < 4
                || top_weights.len < 4
                || residual.len != HIDDEN_SIZE
                || out.len != HIDDEN_SIZE
            {
                return Err(eyre!(
                    "MXFP4 down fused shape mismatch: acts {}, top_indices {}, top_weights {}, residual {}, out {}",
                    expert_acts.len,
                    top_indices.len,
                    top_weights.len,
                    residual.len,
                    out.len
                ));
            }

            let encoder = self.command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.mxfp4_top4_down_weighted);
            encoder.set_buffer(0, Some(&blocks.buffer), 0);
            encoder.set_buffer(1, Some(&scales.buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&expert_acts.buffer), 0);
            encoder.set_buffer(4, Some(&top_indices.buffer), 0);
            encoder.set_buffer(5, Some(&top_weights.buffer), 0);
            encoder.set_buffer(6, Some(&residual.buffer), 0);
            encoder.set_buffer(7, Some(&out.buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(HIDDEN_SIZE as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            encoder.end_encoding();
            Ok(())
        }
    }

    trait MetalDeviceExt {
        fn new_command_queue(&self) -> Retained<MetalCommandQueue>;
        fn new_buffer(&self, length: u64, options: MTLResourceOptions) -> Retained<MetalBuffer>;
        fn new_buffer_with_data<T>(
            &self,
            values: &[T],
            options: MTLResourceOptions,
        ) -> Retained<MetalBuffer>;
    }

    impl MetalDeviceExt for MetalDevice {
        fn new_command_queue(&self) -> Retained<MetalCommandQueue> {
            self.newCommandQueue()
                .expect("Metal command queue allocation failed")
        }

        fn new_buffer(&self, length: u64, options: MTLResourceOptions) -> Retained<MetalBuffer> {
            self.newBufferWithLength_options(length as NSUInteger, options)
                .expect("Metal buffer allocation failed")
        }

        fn new_buffer_with_data<T>(
            &self,
            values: &[T],
            options: MTLResourceOptions,
        ) -> Retained<MetalBuffer> {
            let pointer = NonNull::new(values.as_ptr().cast_mut().cast::<c_void>())
                .expect("slice pointers are never null");
            unsafe {
                self.newBufferWithBytes_length_options(
                    pointer,
                    size_of_val(values) as NSUInteger,
                    options,
                )
            }
            .expect("Metal buffer allocation failed")
        }
    }

    trait MetalCommandQueueExt {
        fn new_command_buffer(&self) -> Retained<MetalCommandBuffer>;
    }

    impl MetalCommandQueueExt for MetalCommandQueue {
        fn new_command_buffer(&self) -> Retained<MetalCommandBuffer> {
            self.commandBuffer()
                .expect("Metal command buffer allocation failed")
        }
    }

    trait MetalCommandBufferExt {
        fn new_compute_command_encoder(&self) -> Retained<MetalComputeCommandEncoder>;
        fn wait_until_completed(&self);
    }

    impl MetalCommandBufferExt for MetalCommandBuffer {
        fn new_compute_command_encoder(&self) -> Retained<MetalComputeCommandEncoder> {
            self.computeCommandEncoder()
                .expect("Metal compute encoder allocation failed")
        }

        fn wait_until_completed(&self) {
            self.waitUntilCompleted();
        }
    }

    trait MetalComputeCommandEncoderExt {
        fn set_compute_pipeline_state(&self, state: &Retained<MetalComputePipelineState>);
        fn set_buffer(&self, index: u64, buffer: Option<&Retained<MetalBuffer>>, offset: u64);
        fn dispatch_thread_groups(
            &self,
            threadgroups_per_grid: MTLSize,
            threads_per_threadgroup: MTLSize,
        );
        fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize);
        fn end_encoding(&self);
    }

    impl MetalComputeCommandEncoderExt for MetalComputeCommandEncoder {
        fn set_compute_pipeline_state(&self, state: &Retained<MetalComputePipelineState>) {
            self.setComputePipelineState(state);
        }

        fn set_buffer(&self, index: u64, buffer: Option<&Retained<MetalBuffer>>, offset: u64) {
            unsafe {
                self.setBuffer_offset_atIndex(
                    buffer.map(|buffer| &**buffer),
                    offset as NSUInteger,
                    index as NSUInteger,
                );
            }
        }

        fn dispatch_thread_groups(
            &self,
            threadgroups_per_grid: MTLSize,
            threads_per_threadgroup: MTLSize,
        ) {
            self.dispatchThreadgroups_threadsPerThreadgroup(
                threadgroups_per_grid,
                threads_per_threadgroup,
            );
        }

        fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
            self.dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup);
        }

        fn end_encoding(&self) {
            self.endEncoding();
        }
    }

    fn pipeline(
        device: &MetalDevice,
        library: &MetalLibrary,
        name: &str,
    ) -> Result<Retained<MetalComputePipelineState>> {
        let function = metal_function(library, name)?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| eyre!("create Metal pipeline {name}: {error:?}"))
    }

    fn metal_function(library: &MetalLibrary, name: &str) -> Result<Retained<MetalFunction>> {
        let name = NSString::from_str(name);
        let function = library
            .newFunctionWithName(&name)
            .ok_or_else(|| eyre!("load Metal function {name}"))?;
        Ok(function)
    }

    fn buffer_with_data<T>(device: &MetalDevice, values: &[T]) -> Retained<MetalBuffer> {
        device.new_buffer_with_data(values, MTLResourceOptions::StorageModeShared)
    }

    fn read_buffer<T: Copy>(buffer: &MetalBuffer, len: usize) -> Vec<T> {
        let values =
            unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<T>(), len) };
        values.to_vec()
    }

    fn mtl_size(width: NSUInteger, height: NSUInteger, depth: NSUInteger) -> MTLSize {
        MTLSize {
            width,
            height,
            depth,
        }
    }

    fn command_buffer_gpu_time_ns(command_buffer: &MetalCommandBuffer) -> u128 {
        let start = command_buffer.GPUStartTime();
        let end = command_buffer.GPUEndTime();
        if !start.is_finite() || !end.is_finite() || end <= start {
            return 0;
        }
        ((end - start) * 1_000_000_000.0) as u128
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) mod platform {
    use crate::runtime_core::ExpertScore;
    use eyre::{Result, eyre};
    use std::marker::PhantomData;

    pub struct MetalBatch<'a> {
        _marker: PhantomData<&'a ()>,
    }
    pub struct MetalContext;
    #[derive(Clone)]
    pub struct Bf16MatrixBuffer;
    #[derive(Clone)]
    pub struct F32VectorBuffer;
    #[derive(Clone)]
    pub struct U8Buffer;
    #[derive(Clone)]
    pub struct U32Buffer;

    impl Bf16MatrixBuffer {
        pub fn rows(&self) -> usize {
            0
        }

        pub fn cols(&self) -> usize {
            0
        }
    }

    impl MetalContext {
        pub fn new() -> Result<Self> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn take_gpu_time_ns(&self) -> u128 {
            0
        }

        pub fn set_profile_enabled(&self, _enabled: bool) {}

        pub fn begin_batch(&self) -> MetalBatch<'_> {
            MetalBatch {
                _marker: PhantomData,
            }
        }

        pub fn alloc_f32_vector(&self, _len: usize) -> Result<F32VectorBuffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn alloc_u32_buffer(&self, _len: usize) -> Result<U32Buffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn upload_u32_buffer(&self, _values: &[u32]) -> Result<U32Buffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn read_f32_vector(&self, _buffer: &F32VectorBuffer) -> Vec<f32> {
            Vec::new()
        }

        pub fn read_u32_buffer(&self, _buffer: &U32Buffer) -> Vec<u32> {
            Vec::new()
        }

        pub fn rms_norm(&self, _x: &[f32], _weight: &[f32]) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn bf16_matvec(
            &self,
            _weight: &[u16],
            _rows: usize,
            _cols: usize,
            _input: &[f32],
            _bias: &[f32],
        ) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn bf16_matrix_matvec(
            &self,
            _weight: &Bf16MatrixBuffer,
            _input: &[f32],
            _bias: &F32VectorBuffer,
        ) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn upload_f32_vector(&self, _values: &[f32]) -> Result<F32VectorBuffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn upload_u8_buffer(&self, _values: &[u8]) -> Result<U8Buffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn upload_bf16_matrix(
            &self,
            _weight: &[u16],
            _rows: usize,
            _cols: usize,
        ) -> Result<Bf16MatrixBuffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn bf16_matrix_topk(
            &self,
            _weight: &Bf16MatrixBuffer,
            _input: &[f32],
            _k: usize,
        ) -> Result<Vec<(usize, f32)>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn rope_row(&self, _row: &[f32], _heads: usize, _position: usize) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn single_token_attention(
            &self,
            _q: &[f32],
            _k: &[f32],
            _v: &[f32],
            _sinks: &[f32],
        ) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn sequence_attention(
            &self,
            _layer: usize,
            _q: &[f32],
            _k: &[f32],
            _v: &[f32],
            _sinks: &[f32],
            _seq_len: usize,
        ) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn kv_cache_decode_attention(
            &self,
            _layer: usize,
            _query_position: usize,
            _cache_start_position: usize,
            _q: &[f32],
            _k_cache: &[f32],
            _v_cache: &[f32],
            _sinks: &[f32],
        ) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn vector_add(&self, _left: &[f32], _right: &[f32]) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn top4_softmax(&self, _logits: &[f32]) -> Result<Vec<ExpertScore>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn mxfp4_matvec(
            &self,
            _blocks: &[u8],
            _scales: &[u8],
            _rows: usize,
            _input: &[f32],
            _bias: &[f32],
        ) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn mxfp4_matvec_resident(
            &self,
            _blocks: &U8Buffer,
            _scales: &U8Buffer,
            _rows: usize,
            _input: &[f32],
            _bias: &F32VectorBuffer,
        ) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn swiglu(&self, _values: &[f32]) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn weighted_sum4(&self, _vectors: [&[f32]; 4], _weights: [f32; 4]) -> Result<Vec<f32>> {
            Err(eyre!("Metal backend is only available on macOS"))
        }
    }

    impl<'a> MetalBatch<'a> {
        pub fn finish(self) -> u128 {
            0
        }

        pub fn embedding_lookup_bf16_into(
            &self,
            _weight: &Bf16MatrixBuffer,
            _token: usize,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn embedding_lookup_bf16_batch_into(
            &self,
            _weight: &Bf16MatrixBuffer,
            _tokens: &U32Buffer,
            _token_count: usize,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn rms_norm_into(
            &self,
            _input: &F32VectorBuffer,
            _weight: &F32VectorBuffer,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn rms_norm_batch_into(
            &self,
            _input: &F32VectorBuffer,
            _weight: &F32VectorBuffer,
            _out: &F32VectorBuffer,
            _rows: usize,
            _cols: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn bf16_matrix_matvec_into(
            &self,
            _weight: &Bf16MatrixBuffer,
            _input: &F32VectorBuffer,
            _bias: &F32VectorBuffer,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn bf16_matrix_matvec_batch_into(
            &self,
            _weight: &Bf16MatrixBuffer,
            _input: &F32VectorBuffer,
            _bias: &F32VectorBuffer,
            _out: &F32VectorBuffer,
            _batch_rows: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn bf16_matrix_topk_into(
            &self,
            _weight: &Bf16MatrixBuffer,
            _input: &F32VectorBuffer,
            _logits: &F32VectorBuffer,
            _indices: &U32Buffer,
            _values: &F32VectorBuffer,
            _k: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn bf16_matrix_top1_into(
            &self,
            _weight: &Bf16MatrixBuffer,
            _input: &F32VectorBuffer,
            _logits: &F32VectorBuffer,
            _block_indices: &U32Buffer,
            _block_values: &F32VectorBuffer,
            _out_index: &U32Buffer,
            _out_value: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn rope_row_into(
            &self,
            _input: &F32VectorBuffer,
            _out: &F32VectorBuffer,
            _heads: usize,
            _position: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn rope_batch_into(
            &self,
            _input: &F32VectorBuffer,
            _out: &F32VectorBuffer,
            _heads: usize,
            _start_position: usize,
            _rows: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn write_f32_slot_into(
            &self,
            _input: &F32VectorBuffer,
            _output: &F32VectorBuffer,
            _slot: usize,
            _width: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn write_f32_slots_batch_into(
            &self,
            _input: &F32VectorBuffer,
            _output: &F32VectorBuffer,
            _start_slot: usize,
            _slots: usize,
            _width: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn read_f32_slot_into(
            &self,
            _input: &F32VectorBuffer,
            _slot: usize,
            _width: usize,
            _output: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn sequence_attention_into(
            &self,
            _layer: usize,
            _q: &F32VectorBuffer,
            _k: &F32VectorBuffer,
            _v: &F32VectorBuffer,
            _sinks: &F32VectorBuffer,
            _out: &F32VectorBuffer,
            _seq_len: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn kv_cache_decode_attention_into(
            &self,
            _layer: usize,
            _query_position: usize,
            _cache_start_position: usize,
            _cache_len: usize,
            _q: &F32VectorBuffer,
            _k_cache: &F32VectorBuffer,
            _v_cache: &F32VectorBuffer,
            _sinks: &F32VectorBuffer,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn vector_add_into(
            &self,
            _left: &F32VectorBuffer,
            _right: &F32VectorBuffer,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn top4_softmax_into(
            &self,
            _logits: &F32VectorBuffer,
            _indices: &U32Buffer,
            _selected_logits: &F32VectorBuffer,
            _weights: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn mxfp4_top4_gate_swiglu_into(
            &self,
            _blocks: &U8Buffer,
            _scales: &U8Buffer,
            _bias: &Bf16MatrixBuffer,
            _input: &F32VectorBuffer,
            _top_indices: &U32Buffer,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn mxfp4_top4_down_weighted_into(
            &self,
            _blocks: &U8Buffer,
            _scales: &U8Buffer,
            _bias: &Bf16MatrixBuffer,
            _expert_acts: &F32VectorBuffer,
            _top_indices: &U32Buffer,
            _top_weights: &F32VectorBuffer,
            _residual: &F32VectorBuffer,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }
    }
}
