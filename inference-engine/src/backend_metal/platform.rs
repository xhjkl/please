use super::profile::GpuStage;
use super::{
    ATTN_VALUES, EXPERTS, HIDDEN_SIZE, KV_VALUES, LM_HEAD_TOP1_BLOCK_SIZE, LOCAL_WINDOW_TOKENS,
    Q_HEADS,
};

pub(crate) use imp::*;

#[cfg(target_os = "macos")]
mod imp {
    use super::GpuStage;
    use super::{
        ATTN_VALUES, EXPERTS, HIDDEN_SIZE, KV_VALUES, LM_HEAD_TOP1_BLOCK_SIZE, LOCAL_WINDOW_TOKENS,
        Q_HEADS,
    };
    use eyre::{Result, eyre};
    use objc2::ffi::NSUInteger;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    #[cfg(feature = "profile")]
    use objc2_foundation::NSRange;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLFunction, MTLLibrary, MTLResourceOptions, MTLSize,
    };
    #[cfg(feature = "profile")]
    use objc2_metal::{
        MTLComputePassDescriptor, MTLCounterResultTimestamp, MTLCounterSampleBuffer,
        MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLStorageMode,
    };
    use std::cell::{Cell, RefCell};
    #[cfg(feature = "profile")]
    use std::collections::HashMap;
    use std::ffi::c_void;
    #[cfg(feature = "profile")]
    use std::mem::size_of;
    use std::mem::size_of_val;
    use std::ptr::NonNull;
    #[cfg(feature = "profile")]
    use std::slice;
    #[cfg(feature = "profile")]
    use std::sync::Mutex;

    const THREADS_PER_GROUP: u64 = 256;
    const Q8_MV_THREADS_PER_GROUP: u64 = 128;
    const DECODE_ATTENTION_THREADS_PER_GROUP: u64 = 64;
    #[cfg(feature = "profile")]
    const MAX_COUNTER_SAMPLES_PER_BATCH: usize = 1024;

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
        #[cfg(not(feature = "profile"))]
        encoder: RefCell<Option<Retained<MetalComputeCommandEncoder>>>,
        compute_encoders: Cell<usize>,
        dispatches: Cell<usize>,
        scalar_param_buffers: Cell<usize>,
        #[cfg(feature = "profile")]
        label: Option<String>,
        #[cfg(feature = "profile")]
        stage: Cell<GpuStage>,
        #[cfg(feature = "profile")]
        counter_samples: RefCell<Option<CounterSamples>>,
    }

    pub struct BatchTiming {
        pub gpu_ns: u128,
        pub counters: BatchCounters,
        #[cfg(feature = "profile")]
        pub gpu_stages: Vec<(GpuStage, u128)>,
    }

    #[derive(Debug, Clone, Copy, Default)]
    pub struct BatchCounters {
        pub command_buffers: usize,
        pub compute_encoders: usize,
        pub dispatches: usize,
        pub scalar_param_buffers: usize,
    }

    #[cfg(feature = "profile")]
    struct CounterSamples {
        sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
        stages: Vec<GpuStage>,
        sample_limit: usize,
        timestamp_frequency: u64,
        mode: CounterSamplingMode,
    }

    #[cfg(feature = "profile")]
    #[derive(Debug, Clone, Copy)]
    enum CounterSamplingMode {
        EncoderBoundary,
        DispatchBoundary,
    }

    pub struct MetalContext {
        device: Retained<MetalDevice>,
        queue: Retained<MetalCommandQueue>,
        #[cfg(feature = "profile")]
        gpu_time_ns: Mutex<u128>,
        #[cfg(feature = "profile")]
        counter_sampling_status: Mutex<String>,
        partial_sum_squares: Retained<MetalComputePipelineState>,
        apply_rms_norm_from_partials: Retained<MetalComputePipelineState>,
        rms_norm_batch: Retained<MetalComputePipelineState>,
        f32_matvec: Retained<MetalComputePipelineState>,
        f32_matvec_batch: Retained<MetalComputePipelineState>,
        q8_0_matvec_logits_pair: Retained<MetalComputePipelineState>,
        q8_0_matvec_add_pair: Retained<MetalComputePipelineState>,
        q8_0_qkv_matvec_pair: Retained<MetalComputePipelineState>,
        q8_0_matvec_add_batch: Retained<MetalComputePipelineState>,
        q8_0_qkv_matvec_batch: Retained<MetalComputePipelineState>,
        top1_logits_blocks: Retained<MetalComputePipelineState>,
        top1_logits_final: Retained<MetalComputePipelineState>,
        embedding_lookup_q8_0: Retained<MetalComputePipelineState>,
        embedding_lookup_q8_0_batch: Retained<MetalComputePipelineState>,
        rope_batch: Retained<MetalComputePipelineState>,
        qk_rope_write_cache: Retained<MetalComputePipelineState>,
        sequence_attention: Retained<MetalComputePipelineState>,
        suffix_sequence_attention: Retained<MetalComputePipelineState>,
        kv_cache_decode_attention: Retained<MetalComputePipelineState>,
        write_f32_slots_batch: Retained<MetalComputePipelineState>,
        copy_f32_slot: Retained<MetalComputePipelineState>,
        top4_softmax: Retained<MetalComputePipelineState>,
        top4_softmax_batch: Retained<MetalComputePipelineState>,
        mxfp4_gguf_top4_gate_swiglu: Retained<MetalComputePipelineState>,
        mxfp4_gguf_top4_down_slots: Retained<MetalComputePipelineState>,
        weighted_sum4_residual: Retained<MetalComputePipelineState>,
        mxfp4_gguf_top4_gate_swiglu_batch: Retained<MetalComputePipelineState>,
        mxfp4_gguf_top4_down_weighted_batch: Retained<MetalComputePipelineState>,
    }

    impl Q8_0MatrixBuffer {
        pub fn rows(&self) -> usize {
            self.rows
        }
    }

    #[derive(Clone)]
    pub struct F32VectorBuffer {
        buffer: Retained<MetalBuffer>,
        len: usize,
    }

    #[derive(Clone)]
    pub struct F32MatrixBuffer {
        buffer: Retained<MetalBuffer>,
        rows: usize,
        cols: usize,
    }

    #[derive(Clone)]
    pub struct Q8_0MatrixBuffer {
        buffer: Retained<MetalBuffer>,
        rows: usize,
        cols: usize,
    }

    #[derive(Clone)]
    pub struct Mxfp4ExpertTensorBuffer {
        buffer: Retained<MetalBuffer>,
        experts: usize,
        rows: usize,
        cols: usize,
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
                #[cfg(feature = "profile")]
                gpu_time_ns: Mutex::new(0),
                #[cfg(feature = "profile")]
                counter_sampling_status: Mutex::new("not attempted".to_string()),
                partial_sum_squares: pipeline(&device, &library, "partial_sum_squares")?,
                apply_rms_norm_from_partials: pipeline(
                    &device,
                    &library,
                    "apply_rms_norm_from_partials",
                )?,
                rms_norm_batch: pipeline(&device, &library, "rms_norm_batch")?,
                f32_matvec: pipeline(&device, &library, "f32_matvec")?,
                f32_matvec_batch: pipeline(&device, &library, "f32_matvec_batch")?,
                q8_0_matvec_logits_pair: pipeline(&device, &library, "q8_0_matvec_logits_pair")?,
                q8_0_matvec_add_pair: pipeline(&device, &library, "q8_0_matvec_add_pair")?,
                q8_0_qkv_matvec_pair: pipeline(&device, &library, "q8_0_qkv_matvec_pair")?,
                q8_0_matvec_add_batch: pipeline(&device, &library, "q8_0_matvec_add_batch")?,
                q8_0_qkv_matvec_batch: pipeline(&device, &library, "q8_0_qkv_matvec_batch")?,
                top1_logits_blocks: pipeline(&device, &library, "top1_logits_blocks")?,
                top1_logits_final: pipeline(&device, &library, "top1_logits_final")?,
                embedding_lookup_q8_0: pipeline(&device, &library, "embedding_lookup_q8_0")?,
                embedding_lookup_q8_0_batch: pipeline(
                    &device,
                    &library,
                    "embedding_lookup_q8_0_batch",
                )?,
                rope_batch: pipeline(&device, &library, "rope_batch")?,
                qk_rope_write_cache: pipeline(&device, &library, "qk_rope_write_cache")?,
                sequence_attention: pipeline(&device, &library, "sequence_attention")?,
                suffix_sequence_attention: pipeline(
                    &device,
                    &library,
                    "suffix_sequence_attention",
                )?,
                kv_cache_decode_attention: pipeline(
                    &device,
                    &library,
                    "kv_cache_decode_attention",
                )?,
                write_f32_slots_batch: pipeline(&device, &library, "write_f32_slots_batch")?,
                copy_f32_slot: pipeline(&device, &library, "copy_f32_slot")?,
                top4_softmax: pipeline(&device, &library, "top4_softmax")?,
                top4_softmax_batch: pipeline(&device, &library, "top4_softmax_batch")?,
                mxfp4_gguf_top4_gate_swiglu: pipeline(
                    &device,
                    &library,
                    "mxfp4_gguf_top4_gate_swiglu",
                )?,
                mxfp4_gguf_top4_down_slots: pipeline(
                    &device,
                    &library,
                    "mxfp4_gguf_top4_down_slots",
                )?,
                weighted_sum4_residual: pipeline(&device, &library, "weighted_sum4_residual")?,
                mxfp4_gguf_top4_gate_swiglu_batch: pipeline(
                    &device,
                    &library,
                    "mxfp4_gguf_top4_gate_swiglu_batch",
                )?,
                mxfp4_gguf_top4_down_weighted_batch: pipeline(
                    &device,
                    &library,
                    "mxfp4_gguf_top4_down_weighted_batch",
                )?,
                device,
                queue,
            })
        }

        #[cfg(feature = "profile")]
        pub fn take_gpu_time_ns(&self) -> u128 {
            let mut gpu_time_ns = self.gpu_time_ns.lock().unwrap();
            let value = *gpu_time_ns;
            *gpu_time_ns = 0;
            value
        }

        #[cfg(feature = "profile")]
        pub fn counter_sampling_summary(&self) -> String {
            self.counter_sampling_status.lock().unwrap().clone()
        }

        #[cfg(feature = "profile")]
        fn set_counter_sampling_status(&self, status: impl Into<String>) {
            *self.counter_sampling_status.lock().unwrap() = status.into();
        }

        #[cfg(feature = "profile")]
        fn detect_counter_sampling(&self) -> &'static str {
            if !self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
                && !self
                    .device
                    .supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary)
            {
                return "unsupported by this Metal device";
            }
            let Some(counter_sets) = self.device.counterSets() else {
                return "counter sets unavailable";
            };
            let timestamp_name = NSString::from_str("timestamp");
            if counter_sets
                .iter()
                .all(|counter_set| &*counter_set.name() != &*timestamp_name)
            {
                return "timestamp counter set unavailable";
            }
            if self.device.queryTimestampFrequency() == 0 {
                return "timestamp frequency unavailable";
            }
            if self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
            {
                "compute-pass encoder-boundary timestamps"
            } else {
                "compute-dispatch-boundary timestamps"
            }
        }

        fn finish_command_buffer(&self, command_buffer: &MetalCommandBuffer) -> u128 {
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_time_ns = command_buffer_gpu_time_ns(command_buffer);
            #[cfg(feature = "profile")]
            {
                if gpu_time_ns > 0 {
                    *self.gpu_time_ns.lock().unwrap() += gpu_time_ns;
                }
            }
            gpu_time_ns
        }

        pub fn begin_labeled_batch(&self, label: &str) -> MetalBatch<'_> {
            let command_buffer = self.queue.new_command_buffer();
            #[cfg(feature = "profile")]
            command_buffer.setLabel(Some(&NSString::from_str(label)));
            #[cfg(not(feature = "profile"))]
            let _ = label;
            #[cfg(feature = "profile")]
            let counter_samples = self.counter_samples(label);
            MetalBatch {
                context: self,
                command_buffer,
                #[cfg(not(feature = "profile"))]
                encoder: RefCell::new(None),
                compute_encoders: Cell::new(0),
                dispatches: Cell::new(0),
                scalar_param_buffers: Cell::new(0),
                #[cfg(feature = "profile")]
                label: Some(label.to_owned()),
                #[cfg(feature = "profile")]
                stage: Cell::new(GpuStage::Other),
                #[cfg(feature = "profile")]
                counter_samples: RefCell::new(counter_samples),
            }
        }

        #[cfg(feature = "profile")]
        fn counter_samples(&self, label: &str) -> Option<CounterSamples> {
            let status = self.detect_counter_sampling();
            self.set_counter_sampling_status(status);
            if status != "compute-pass encoder-boundary timestamps"
                && status != "compute-dispatch-boundary timestamps"
            {
                return None;
            }
            let mode = if self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
            {
                CounterSamplingMode::EncoderBoundary
            } else if self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary)
            {
                CounterSamplingMode::DispatchBoundary
            } else {
                return None;
            };
            let counter_sets = self.device.counterSets()?;
            let timestamp_name = NSString::from_str("timestamp");
            let timestamp_counter = counter_sets
                .iter()
                .find(|counter_set| &*counter_set.name() == &*timestamp_name)?;
            let descriptor = MTLCounterSampleBufferDescriptor::new();
            descriptor.setCounterSet(Some(&timestamp_counter));
            descriptor.setStorageMode(MTLStorageMode::Shared);
            descriptor.setLabel(&NSString::from_str(label));
            unsafe {
                descriptor.setSampleCount(MAX_COUNTER_SAMPLES_PER_BATCH as NSUInteger);
            }
            let sample_buffer = self
                .device
                .newCounterSampleBufferWithDescriptor_error(&descriptor)
                .inspect_err(|error| {
                    self.set_counter_sampling_status(format!(
                        "sample buffer creation failed: {error:?}"
                    ))
                })
                .ok()?;
            let timestamp_frequency = self.device.queryTimestampFrequency();
            if timestamp_frequency == 0 {
                self.set_counter_sampling_status("timestamp frequency unavailable");
                return None;
            }
            Some(CounterSamples {
                sample_buffer,
                stages: Vec::new(),
                sample_limit: MAX_COUNTER_SAMPLES_PER_BATCH,
                timestamp_frequency,
                mode,
            })
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

        pub fn write_u32_buffer(&self, buffer: &U32Buffer, values: &[u32]) -> Result<()> {
            if values.len() > buffer.len {
                return Err(eyre!(
                    "u32 buffer write has {} values, capacity {}",
                    values.len(),
                    buffer.len
                ));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.buffer.contents().as_ptr().cast::<u32>(),
                    values.len(),
                );
            }
            Ok(())
        }

        pub fn read_u32_array<const N: usize>(&self, buffer: &U32Buffer) -> Result<[u32; N]> {
            if buffer.len < N {
                return Err(eyre!(
                    "u32 buffer read needs {N} values, capacity {}",
                    buffer.len
                ));
            }
            Ok(read_buffer_array::<u32, N>(&buffer.buffer))
        }

        pub fn upload_f32_vector_bytes(
            &self,
            values: &[u8],
            len: usize,
        ) -> Result<F32VectorBuffer> {
            let expected = len
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| eyre!("f32 vector byte shape overflow"))?;
            if values.len() != expected {
                return Err(eyre!(
                    "f32 vector has {} bytes, expected {len} * 4",
                    values.len()
                ));
            }
            Ok(F32VectorBuffer {
                buffer: buffer_with_data(&self.device, values),
                len,
            })
        }

        pub fn upload_f32_matrix_bytes(
            &self,
            values: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<F32MatrixBuffer> {
            let expected = rows
                .checked_mul(cols)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| eyre!("f32 matrix byte shape overflow"))?;
            if values.len() != expected {
                return Err(eyre!(
                    "f32 matrix has {} bytes, expected {rows} * {cols} * 4",
                    values.len()
                ));
            }
            Ok(F32MatrixBuffer {
                buffer: buffer_with_data(&self.device, values),
                rows,
                cols,
            })
        }

        pub fn upload_q8_0_matrix_bytes(
            &self,
            values: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<Q8_0MatrixBuffer> {
            if cols % 32 != 0 {
                return Err(eyre!("q8_0 matrix cols {cols} is not divisible by 32"));
            }
            let expected = rows
                .checked_mul(cols / 32)
                .and_then(|blocks| blocks.checked_mul(2 + 32))
                .ok_or_else(|| eyre!("q8_0 matrix byte shape overflow"))?;
            if values.len() != expected {
                return Err(eyre!(
                    "q8_0 matrix has {} bytes, expected {rows} * ({cols} / 32) * 34",
                    values.len()
                ));
            }
            Ok(Q8_0MatrixBuffer {
                buffer: buffer_with_data(&self.device, values),
                rows,
                cols,
            })
        }

        pub fn upload_mxfp4_expert_tensor_bytes(
            &self,
            values: &[u8],
            experts: usize,
            rows: usize,
            cols: usize,
        ) -> Result<Mxfp4ExpertTensorBuffer> {
            if cols % 32 != 0 {
                return Err(eyre!("MXFP4 expert cols {cols} is not divisible by 32"));
            }
            let expected = experts
                .checked_mul(rows)
                .and_then(|values| values.checked_mul(cols / 32))
                .and_then(|groups| groups.checked_mul(1 + 16))
                .ok_or_else(|| eyre!("MXFP4 expert tensor byte shape overflow"))?;
            if values.len() != expected {
                return Err(eyre!(
                    "MXFP4 expert tensor has {} bytes, expected {experts} * {rows} * ({cols} / 32) * 17",
                    values.len()
                ));
            }
            Ok(Mxfp4ExpertTensorBuffer {
                buffer: buffer_with_data(&self.device, values),
                experts,
                rows,
                cols,
            })
        }
    }

    impl<'a> MetalBatch<'a> {
        pub fn finish(self) -> BatchTiming {
            #[cfg(feature = "profile")]
            self.sample_stage_boundary(GpuStage::Other);
            #[cfg(not(feature = "profile"))]
            if let Some(encoder) = self.encoder.borrow_mut().take() {
                encoder.end_encoding();
            }
            let gpu_ns = self.context.finish_command_buffer(&self.command_buffer);
            #[cfg(feature = "profile")]
            let gpu_stages = self
                .counter_samples
                .into_inner()
                .map(CounterSamples::resolve)
                .unwrap_or_default();
            BatchTiming {
                gpu_ns,
                counters: BatchCounters {
                    command_buffers: 1,
                    compute_encoders: self.compute_encoders.get(),
                    dispatches: self.dispatches.get(),
                    scalar_param_buffers: self.scalar_param_buffers.get(),
                },
                #[cfg(feature = "profile")]
                gpu_stages,
            }
        }

        pub fn set_stage(&self, stage: GpuStage) {
            #[cfg(feature = "profile")]
            self.stage.set(stage);
            #[cfg(not(feature = "profile"))]
            let _ = stage;
        }

        fn encoder(&self, label: &str) -> Retained<MetalComputeCommandEncoder> {
            #[cfg(not(feature = "profile"))]
            {
                let _ = label;
                let mut encoder = self.encoder.borrow_mut();
                if let Some(encoder) = &*encoder {
                    return encoder.clone();
                }
                self.compute_encoders
                    .set(self.compute_encoders.get().saturating_add(1));
                let created = self.command_buffer.new_compute_command_encoder();
                *encoder = Some(created.clone());
                return created;
            }

            #[cfg(feature = "profile")]
            {
                self.compute_encoders
                    .set(self.compute_encoders.get().saturating_add(1));
                let sample_indices = self.reserve_encoder_boundary_samples(self.stage.get());
                let sampled_by_encoder = sample_indices.is_some();
                let encoder = if let Some((start, end)) = sample_indices {
                    self.command_buffer
                        .new_compute_command_encoder_with_samples(
                            self.counter_sample_buffer()
                                .as_deref()
                                .expect("reserved sample indices require a sample buffer"),
                            start,
                            end,
                        )
                } else {
                    self.command_buffer.new_compute_command_encoder()
                };
                if let Some(batch_label) = &self.label {
                    let label = format!("{batch_label}.{label}");
                    encoder.setLabel(Some(&NSString::from_str(&label)));
                }
                if !sampled_by_encoder {
                    self.sample_encoder_boundary(&encoder, self.stage.get());
                }
                encoder
            }
        }

        fn set_scalar<T>(&self, encoder: &MetalComputeCommandEncoder, index: u64, values: &[T]) {
            encoder.set_bytes(index, values);
        }

        fn end_encoder(&self, encoder: Retained<MetalComputeCommandEncoder>) {
            self.dispatches.set(self.dispatches.get().saturating_add(1));
            #[cfg(feature = "profile")]
            {
                if self.uses_dispatch_boundary_samples() {
                    self.sample_encoder_boundary(&encoder, GpuStage::Other);
                }
                encoder.end_encoding();
            }
            #[cfg(not(feature = "profile"))]
            let _ = encoder;
        }

        #[cfg(feature = "profile")]
        fn sample_stage_boundary(&self, stage: GpuStage) {
            let encoder = self.command_buffer.new_compute_command_encoder();
            if let Some(batch_label) = &self.label {
                let label = format!("{batch_label}.profile_boundary");
                encoder.setLabel(Some(&NSString::from_str(&label)));
            }
            self.sample_encoder_boundary(&encoder, stage);
            encoder.end_encoding();
        }

        #[cfg(feature = "profile")]
        fn sample_encoder_boundary(&self, encoder: &MetalComputeCommandEncoder, stage: GpuStage) {
            let mut counter_samples = self.counter_samples.borrow_mut();
            let Some(counter_samples) = &mut *counter_samples else {
                return;
            };
            counter_samples.sample(encoder, stage);
        }

        #[cfg(feature = "profile")]
        fn reserve_encoder_boundary_samples(&self, stage: GpuStage) -> Option<(usize, usize)> {
            let mut counter_samples = self.counter_samples.borrow_mut();
            let counter_samples = counter_samples.as_mut()?;
            counter_samples.reserve_encoder_boundary(stage)
        }

        #[cfg(feature = "profile")]
        fn uses_dispatch_boundary_samples(&self) -> bool {
            let counter_samples = self.counter_samples.borrow();
            let Some(counter_samples) = &*counter_samples else {
                return false;
            };
            counter_samples.uses_dispatch_boundary()
        }

        #[cfg(feature = "profile")]
        fn counter_sample_buffer(
            &self,
        ) -> Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>> {
            let counter_samples = self.counter_samples.borrow();
            counter_samples
                .as_ref()
                .map(|counter_samples| counter_samples.sample_buffer.clone())
        }

        pub fn embedding_lookup_q8_0_into(
            &self,
            weight: &Q8_0MatrixBuffer,
            token: usize,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if token >= weight.rows {
                return Err(eyre!(
                    "q8_0 embedding token {token} exceeds embedding rows {}",
                    weight.rows
                ));
            }
            if out.len != weight.cols {
                return Err(eyre!(
                    "q8_0 embedding output has {} values, expected {}",
                    out.len,
                    weight.cols
                ));
            }

            let token = token as u32;
            let cols = weight.cols as u32;

            let encoder = self.encoder("embedding_lookup_q8_0");
            encoder.set_compute_pipeline_state(&self.context.embedding_lookup_q8_0);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 2, &[token]);
            self.set_scalar(&encoder, 3, &[cols]);
            encoder.dispatch_threads(
                mtl_size(weight.cols as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn embedding_lookup_q8_0_batch_into(
            &self,
            weight: &Q8_0MatrixBuffer,
            tokens: &U32Buffer,
            token_count: usize,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if token_count == 0 {
                return Ok(());
            }
            if tokens.len < token_count {
                return Err(eyre!(
                    "batched q8_0 embedding has {} token ids, expected at least {token_count}",
                    tokens.len
                ));
            }
            let expected = token_count
                .checked_mul(weight.cols)
                .ok_or_else(|| eyre!("batched q8_0 embedding output length overflow"))?;
            if out.len < expected {
                return Err(eyre!(
                    "batched q8_0 embedding output has {} values, expected at least {expected}",
                    out.len
                ));
            }

            let cols = weight.cols as u32;
            let token_count = token_count as u32;

            let encoder = self.encoder("embedding_lookup_q8_0_batch");
            encoder.set_compute_pipeline_state(&self.context.embedding_lookup_q8_0_batch);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&tokens.buffer), 0);
            encoder.set_buffer(2, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 3, &[cols]);
            self.set_scalar(&encoder, 4, &[token_count]);
            encoder.dispatch_threads(
                mtl_size(expected as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn rms_norm_with_partials_into(
            &self,
            input: &F32VectorBuffer,
            weight: &F32VectorBuffer,
            partials: &F32VectorBuffer,
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
            if partials.len < groups as usize {
                return Err(eyre!(
                    "RMSNorm partial scratch has {} values, expected at least {groups}",
                    partials.len
                ));
            }
            let groups = groups as u32;

            let encoder = self.encoder("partial_sum_squares");
            encoder.set_compute_pipeline_state(&self.context.partial_sum_squares);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&partials.buffer), 0);
            self.set_scalar(&encoder, 2, &[n]);
            encoder.dispatch_thread_groups(
                mtl_size(groups as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);

            let encoder = self.encoder("apply_rms_norm_from_partials");
            encoder.set_compute_pipeline_state(&self.context.apply_rms_norm_from_partials);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&weight.buffer), 0);
            encoder.set_buffer(2, Some(&partials.buffer), 0);
            encoder.set_buffer(3, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 4, &[n]);
            self.set_scalar(&encoder, 5, &[groups]);
            encoder.dispatch_threads(
                mtl_size(input.len as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
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

            let encoder = self.encoder("rms_norm_batch");
            encoder.set_compute_pipeline_state(&self.context.rms_norm_batch);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&weight.buffer), 0);
            encoder.set_buffer(2, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 3, &[rows]);
            self.set_scalar(&encoder, 4, &[cols]);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn f32_matrix_matvec_into(
            &self,
            weight: &F32MatrixBuffer,
            input: &F32VectorBuffer,
            bias: &F32VectorBuffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if input.len != weight.cols {
                return Err(eyre!(
                    "f32 matvec input has {} values, expected {}",
                    input.len,
                    weight.cols
                ));
            }
            if bias.len != weight.rows || out.len != weight.rows {
                return Err(eyre!(
                    "f32 matvec row mismatch: bias {}, out {}, rows {}",
                    bias.len,
                    out.len,
                    weight.rows
                ));
            }

            let rows = weight.rows as u32;
            let cols = weight.cols as u32;

            let encoder = self.encoder("f32_matvec");
            encoder.set_compute_pipeline_state(&self.context.f32_matvec);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input.buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 4, &[rows]);
            self.set_scalar(&encoder, 5, &[cols]);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn f32_matrix_matvec_batch_into(
            &self,
            weight: &F32MatrixBuffer,
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
                .ok_or_else(|| eyre!("batched f32 matvec input length overflow"))?;
            let out_expected = batch_rows
                .checked_mul(weight.rows)
                .ok_or_else(|| eyre!("batched f32 matvec output length overflow"))?;
            if input.len < input_expected {
                return Err(eyre!(
                    "batched f32 matvec input has {} values, expected at least {input_expected}",
                    input.len
                ));
            }
            if bias.len != weight.rows || out.len < out_expected {
                return Err(eyre!(
                    "batched f32 matvec shape mismatch: bias {}, out {}, rows {}, expected output at least {out_expected}",
                    bias.len,
                    out.len,
                    weight.rows
                ));
            }

            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let batch_rows = batch_rows as u32;

            let encoder = self.encoder("f32_matrix_matvec_batch");
            encoder.set_compute_pipeline_state(&self.context.f32_matvec_batch);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input.buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 4, &[rows]);
            self.set_scalar(&encoder, 5, &[cols]);
            self.set_scalar(&encoder, 6, &[batch_rows]);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, batch_rows as NSUInteger, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn q8_0_matrix_top1_into(
            &self,
            weight: &Q8_0MatrixBuffer,
            input: &F32VectorBuffer,
            logits: &F32VectorBuffer,
            block_indices: &U32Buffer,
            block_values: &F32VectorBuffer,
            out_index: &U32Buffer,
            out_value: &F32VectorBuffer,
            sample_result: &U32Buffer,
        ) -> Result<()> {
            if input.len != weight.cols {
                return Err(eyre!(
                    "q8_0 top-1 input has {} values, expected {}",
                    input.len,
                    weight.cols
                ));
            }
            if logits.len != weight.rows {
                return Err(eyre!(
                    "q8_0 top-1 logits scratch has {} values, expected {}",
                    logits.len,
                    weight.rows
                ));
            }
            let blocks = weight.rows.div_ceil(LM_HEAD_TOP1_BLOCK_SIZE);
            if block_indices.len < blocks || block_values.len < blocks {
                return Err(eyre!(
                    "q8_0 top-1 block scratch has indices {}/values {}, expected at least {blocks}",
                    block_indices.len,
                    block_values.len
                ));
            }
            if out_index.len < 1 || out_value.len < 1 {
                return Err(eyre!("q8_0 top-1 output buffers need one slot"));
            }
            if sample_result.len < 4 {
                return Err(eyre!("q8_0 top-1 sample result needs four u32 slots"));
            }

            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let blocks = blocks as u32;

            let encoder = self.encoder("q8_0_matvec_logits_pair");
            encoder.set_compute_pipeline_state(&self.context.q8_0_matvec_logits_pair);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input.buffer), 0);
            encoder.set_buffer(2, Some(&logits.buffer), 0);
            self.set_scalar(&encoder, 3, &[rows]);
            self.set_scalar(&encoder, 4, &[cols]);
            encoder.dispatch_thread_groups(
                mtl_size(rows.div_ceil(2) as NSUInteger, 1, 1),
                mtl_size(Q8_MV_THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);

            let encoder = self.encoder("top1_logits_blocks");
            encoder.set_compute_pipeline_state(&self.context.top1_logits_blocks);
            encoder.set_buffer(0, Some(&logits.buffer), 0);
            encoder.set_buffer(1, Some(&block_indices.buffer), 0);
            encoder.set_buffer(2, Some(&block_values.buffer), 0);
            self.set_scalar(&encoder, 3, &[rows]);
            encoder.dispatch_thread_groups(
                mtl_size(blocks as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);

            let encoder = self.encoder("top1_logits_final");
            encoder.set_compute_pipeline_state(&self.context.top1_logits_final);
            encoder.set_buffer(0, Some(&block_indices.buffer), 0);
            encoder.set_buffer(1, Some(&block_values.buffer), 0);
            encoder.set_buffer(2, Some(&out_index.buffer), 0);
            encoder.set_buffer(3, Some(&out_value.buffer), 0);
            encoder.set_buffer(4, Some(&sample_result.buffer), 0);
            self.set_scalar(&encoder, 5, &[blocks]);
            encoder.dispatch_threads(
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn q8_0_matrix_matvec_add_into(
            &self,
            weight: &Q8_0MatrixBuffer,
            input: &F32VectorBuffer,
            bias: &F32VectorBuffer,
            residual: &F32VectorBuffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if input.len != weight.cols {
                return Err(eyre!(
                    "q8_0 matvec-add input has {} values, expected {}",
                    input.len,
                    weight.cols
                ));
            }
            if bias.len != weight.rows || residual.len != weight.rows || out.len != weight.rows {
                return Err(eyre!(
                    "q8_0 matvec-add row mismatch: bias {}, residual {}, out {}, rows {}",
                    bias.len,
                    residual.len,
                    out.len,
                    weight.rows
                ));
            }

            let rows = weight.rows as u32;
            let cols = weight.cols as u32;

            let encoder = self.encoder("q8_0_matvec_add_pair");
            encoder.set_compute_pipeline_state(&self.context.q8_0_matvec_add_pair);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input.buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&residual.buffer), 0);
            encoder.set_buffer(4, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 5, &[rows]);
            self.set_scalar(&encoder, 6, &[cols]);
            encoder.dispatch_thread_groups(
                mtl_size(rows.div_ceil(2) as NSUInteger, 1, 1),
                mtl_size(Q8_MV_THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn q8_0_matrix_matvec_add_batch_into(
            &self,
            weight: &Q8_0MatrixBuffer,
            input: &F32VectorBuffer,
            bias: &F32VectorBuffer,
            residual: &F32VectorBuffer,
            out: &F32VectorBuffer,
            batch_rows: usize,
        ) -> Result<()> {
            if batch_rows == 0 {
                return Ok(());
            }
            let input_expected = batch_rows
                .checked_mul(weight.cols)
                .ok_or_else(|| eyre!("batched q8_0 matvec-add input length overflow"))?;
            let out_expected = batch_rows
                .checked_mul(weight.rows)
                .ok_or_else(|| eyre!("batched q8_0 matvec-add output length overflow"))?;
            if input.len < input_expected {
                return Err(eyre!(
                    "batched q8_0 matvec-add input has {} values, expected at least {input_expected}",
                    input.len
                ));
            }
            if bias.len != weight.rows || residual.len < out_expected || out.len < out_expected {
                return Err(eyre!(
                    "batched q8_0 matvec-add shape mismatch: bias {}, residual {}, out {}, rows {}, expected output at least {out_expected}",
                    bias.len,
                    residual.len,
                    out.len,
                    weight.rows
                ));
            }

            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let batch_rows = batch_rows as u32;

            let encoder = self.encoder("q8_0_matvec_add_batch");
            encoder.set_compute_pipeline_state(&self.context.q8_0_matvec_add_batch);
            encoder.set_buffer(0, Some(&weight.buffer), 0);
            encoder.set_buffer(1, Some(&input.buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&residual.buffer), 0);
            encoder.set_buffer(4, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 5, &[rows]);
            self.set_scalar(&encoder, 6, &[cols]);
            self.set_scalar(&encoder, 7, &[batch_rows]);
            encoder.dispatch_thread_groups(
                mtl_size(rows as NSUInteger, batch_rows as NSUInteger, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn q8_0_qkv_matvec_into(
            &self,
            q_weight: &Q8_0MatrixBuffer,
            k_weight: &Q8_0MatrixBuffer,
            v_weight: &Q8_0MatrixBuffer,
            input: &F32VectorBuffer,
            q_bias: &F32VectorBuffer,
            k_bias: &F32VectorBuffer,
            v_bias: &F32VectorBuffer,
            q_out: &F32VectorBuffer,
            k_out: &F32VectorBuffer,
            v_out: &F32VectorBuffer,
        ) -> Result<()> {
            if q_weight.cols != input.len
                || k_weight.cols != input.len
                || v_weight.cols != input.len
            {
                return Err(eyre!(
                    "q8_0 QKV input length mismatch: input {}, q cols {}, k cols {}, v cols {}",
                    input.len,
                    q_weight.cols,
                    k_weight.cols,
                    v_weight.cols
                ));
            }
            if k_weight.rows != v_weight.rows {
                return Err(eyre!(
                    "q8_0 QKV expects matching K/V rows, got {} and {}",
                    k_weight.rows,
                    v_weight.rows
                ));
            }
            if q_bias.len != q_weight.rows || q_out.len != q_weight.rows {
                return Err(eyre!(
                    "q8_0 QKV q row mismatch: bias {}, out {}, rows {}",
                    q_bias.len,
                    q_out.len,
                    q_weight.rows
                ));
            }
            if k_bias.len != k_weight.rows || k_out.len != k_weight.rows {
                return Err(eyre!(
                    "q8_0 QKV k row mismatch: bias {}, out {}, rows {}",
                    k_bias.len,
                    k_out.len,
                    k_weight.rows
                ));
            }
            if v_bias.len != v_weight.rows || v_out.len != v_weight.rows {
                return Err(eyre!(
                    "q8_0 QKV v row mismatch: bias {}, out {}, rows {}",
                    v_bias.len,
                    v_out.len,
                    v_weight.rows
                ));
            }

            let q_rows = q_weight.rows as u32;
            let kv_rows = k_weight.rows as u32;
            let cols = input.len as u32;
            let total_rows = q_weight.rows + k_weight.rows + v_weight.rows;

            let encoder = self.encoder("q8_0_qkv_matvec_pair");
            encoder.set_compute_pipeline_state(&self.context.q8_0_qkv_matvec_pair);
            encoder.set_buffer(0, Some(&q_weight.buffer), 0);
            encoder.set_buffer(1, Some(&k_weight.buffer), 0);
            encoder.set_buffer(2, Some(&v_weight.buffer), 0);
            encoder.set_buffer(3, Some(&input.buffer), 0);
            encoder.set_buffer(4, Some(&q_bias.buffer), 0);
            encoder.set_buffer(5, Some(&k_bias.buffer), 0);
            encoder.set_buffer(6, Some(&v_bias.buffer), 0);
            encoder.set_buffer(7, Some(&q_out.buffer), 0);
            encoder.set_buffer(8, Some(&k_out.buffer), 0);
            encoder.set_buffer(9, Some(&v_out.buffer), 0);
            self.set_scalar(&encoder, 10, &[q_rows]);
            self.set_scalar(&encoder, 11, &[kv_rows]);
            self.set_scalar(&encoder, 12, &[cols]);
            encoder.dispatch_thread_groups(
                mtl_size(total_rows.div_ceil(2) as NSUInteger, 1, 1),
                mtl_size(Q8_MV_THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn q8_0_qkv_matvec_batch_into(
            &self,
            q_weight: &Q8_0MatrixBuffer,
            k_weight: &Q8_0MatrixBuffer,
            v_weight: &Q8_0MatrixBuffer,
            input: &F32VectorBuffer,
            q_bias: &F32VectorBuffer,
            k_bias: &F32VectorBuffer,
            v_bias: &F32VectorBuffer,
            q_out: &F32VectorBuffer,
            k_out: &F32VectorBuffer,
            v_out: &F32VectorBuffer,
            batch_rows: usize,
        ) -> Result<()> {
            if batch_rows == 0 {
                return Ok(());
            }
            let input_expected = batch_rows
                .checked_mul(q_weight.cols)
                .ok_or_else(|| eyre!("batched q8_0 QKV input length overflow"))?;
            if q_weight.cols != k_weight.cols || q_weight.cols != v_weight.cols {
                return Err(eyre!(
                    "batched q8_0 QKV expects matching input widths, got q/k/v {}/{}/{}",
                    q_weight.cols,
                    k_weight.cols,
                    v_weight.cols
                ));
            }
            if input.len < input_expected {
                return Err(eyre!(
                    "batched q8_0 QKV input has {} values, expected at least {input_expected}",
                    input.len
                ));
            }
            if k_weight.rows != v_weight.rows {
                return Err(eyre!(
                    "batched q8_0 QKV expects matching K/V rows, got {} and {}",
                    k_weight.rows,
                    v_weight.rows
                ));
            }
            let q_expected = batch_rows
                .checked_mul(q_weight.rows)
                .ok_or_else(|| eyre!("batched q8_0 Q output length overflow"))?;
            let kv_expected = batch_rows
                .checked_mul(k_weight.rows)
                .ok_or_else(|| eyre!("batched q8_0 KV output length overflow"))?;
            if q_bias.len != q_weight.rows || q_out.len < q_expected {
                return Err(eyre!(
                    "batched q8_0 Q row mismatch: bias {}, out {}, rows {}, expected output at least {q_expected}",
                    q_bias.len,
                    q_out.len,
                    q_weight.rows
                ));
            }
            if k_bias.len != k_weight.rows || k_out.len < kv_expected {
                return Err(eyre!(
                    "batched q8_0 K row mismatch: bias {}, out {}, rows {}, expected output at least {kv_expected}",
                    k_bias.len,
                    k_out.len,
                    k_weight.rows
                ));
            }
            if v_bias.len != v_weight.rows || v_out.len < kv_expected {
                return Err(eyre!(
                    "batched q8_0 V row mismatch: bias {}, out {}, rows {}, expected output at least {kv_expected}",
                    v_bias.len,
                    v_out.len,
                    v_weight.rows
                ));
            }

            let q_rows = q_weight.rows as u32;
            let kv_rows = k_weight.rows as u32;
            let cols = q_weight.cols as u32;
            let batch_rows = batch_rows as u32;
            let total_rows = q_weight.rows + k_weight.rows + v_weight.rows;

            let encoder = self.encoder("q8_0_qkv_matvec_batch");
            encoder.set_compute_pipeline_state(&self.context.q8_0_qkv_matvec_batch);
            encoder.set_buffer(0, Some(&q_weight.buffer), 0);
            encoder.set_buffer(1, Some(&k_weight.buffer), 0);
            encoder.set_buffer(2, Some(&v_weight.buffer), 0);
            encoder.set_buffer(3, Some(&input.buffer), 0);
            encoder.set_buffer(4, Some(&q_bias.buffer), 0);
            encoder.set_buffer(5, Some(&k_bias.buffer), 0);
            encoder.set_buffer(6, Some(&v_bias.buffer), 0);
            encoder.set_buffer(7, Some(&q_out.buffer), 0);
            encoder.set_buffer(8, Some(&k_out.buffer), 0);
            encoder.set_buffer(9, Some(&v_out.buffer), 0);
            self.set_scalar(&encoder, 10, &[q_rows]);
            self.set_scalar(&encoder, 11, &[kv_rows]);
            self.set_scalar(&encoder, 12, &[cols]);
            self.set_scalar(&encoder, 13, &[batch_rows]);
            encoder.dispatch_thread_groups(
                mtl_size(total_rows as NSUInteger, batch_rows as NSUInteger, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
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

            let encoder = self.encoder("rope_batch");
            encoder.set_compute_pipeline_state(&self.context.rope_batch);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 2, &[heads]);
            self.set_scalar(&encoder, 3, &[start_position]);
            self.set_scalar(&encoder, 4, &[rows]);
            encoder.dispatch_threads(
                mtl_size(expected as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn qk_rope_write_cache_into(
            &self,
            q: &F32VectorBuffer,
            k: &F32VectorBuffer,
            v: &F32VectorBuffer,
            q_out: &F32VectorBuffer,
            k_cache: &F32VectorBuffer,
            v_cache: &F32VectorBuffer,
            position: usize,
        ) -> Result<()> {
            if q.len != ATTN_VALUES || q_out.len != ATTN_VALUES {
                return Err(eyre!(
                    "Q RoPE/cache input has q {}, q_out {}, expected {ATTN_VALUES}",
                    q.len,
                    q_out.len
                ));
            }
            if k.len != KV_VALUES || v.len != KV_VALUES {
                return Err(eyre!(
                    "K/V RoPE/cache input has k {}, v {}, expected {KV_VALUES}",
                    k.len,
                    v.len
                ));
            }
            let required = position
                .checked_add(1)
                .and_then(|slots| slots.checked_mul(KV_VALUES))
                .ok_or_else(|| eyre!("QK RoPE/cache length overflow"))?;
            if k_cache.len < required || v_cache.len < required {
                return Err(eyre!(
                    "QK RoPE/cache output has k {}, v {}, needs at least {required}",
                    k_cache.len,
                    v_cache.len
                ));
            }

            let position = position as u32;
            let encoder = self.encoder("qk_rope_write_cache");
            encoder.set_compute_pipeline_state(&self.context.qk_rope_write_cache);
            encoder.set_buffer(0, Some(&q.buffer), 0);
            encoder.set_buffer(1, Some(&k.buffer), 0);
            encoder.set_buffer(2, Some(&v.buffer), 0);
            encoder.set_buffer(3, Some(&q_out.buffer), 0);
            encoder.set_buffer(4, Some(&k_cache.buffer), 0);
            encoder.set_buffer(5, Some(&v_cache.buffer), 0);
            self.set_scalar(&encoder, 6, &[position]);
            encoder.dispatch_threads(
                mtl_size(ATTN_VALUES as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
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

            let encoder = self.encoder("write_f32_slots_batch");
            encoder.set_compute_pipeline_state(&self.context.write_f32_slots_batch);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            self.set_scalar(&encoder, 2, &[start_slot]);
            self.set_scalar(&encoder, 3, &[slots]);
            self.set_scalar(&encoder, 4, &[width]);
            encoder.dispatch_threads(
                mtl_size(input_expected as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn copy_f32_slot_into(
            &self,
            input: &F32VectorBuffer,
            slot: usize,
            width: usize,
            output: &F32VectorBuffer,
        ) -> Result<()> {
            if output.len != width {
                return Err(eyre!(
                    "slot copy output has {} values, expected {width}",
                    output.len
                ));
            }
            let required = slot
                .checked_add(1)
                .and_then(|slots| slots.checked_mul(width))
                .ok_or_else(|| eyre!("slot copy length overflow"))?;
            if input.len < required {
                return Err(eyre!(
                    "slot copy input has {} values, needs at least {required}",
                    input.len
                ));
            }

            let slot = slot as u32;
            let width = width as u32;

            let encoder = self.encoder("copy_f32_slot");
            encoder.set_compute_pipeline_state(&self.context.copy_f32_slot);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            self.set_scalar(&encoder, 2, &[slot]);
            self.set_scalar(&encoder, 3, &[width]);
            encoder.dispatch_threads(
                mtl_size(width as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
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

            let encoder = self.encoder("sequence_attention");
            encoder.set_compute_pipeline_state(&self.context.sequence_attention);
            encoder.set_buffer(0, Some(&q.buffer), 0);
            encoder.set_buffer(1, Some(&k.buffer), 0);
            encoder.set_buffer(2, Some(&v.buffer), 0);
            encoder.set_buffer(3, Some(&sinks.buffer), 0);
            encoder.set_buffer(4, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 5, &[seq_len]);
            self.set_scalar(&encoder, 6, &[layer]);
            encoder.dispatch_thread_groups(
                mtl_size(Q_HEADS as NSUInteger, seq_len as NSUInteger, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn suffix_sequence_attention_into(
            &self,
            layer: usize,
            start_position: usize,
            suffix_len: usize,
            q: &F32VectorBuffer,
            k_cache: &F32VectorBuffer,
            v_cache: &F32VectorBuffer,
            sinks: &F32VectorBuffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if suffix_len == 0 {
                return Ok(());
            }
            let q_len = suffix_len
                .checked_mul(ATTN_VALUES)
                .ok_or_else(|| eyre!("resident suffix attention q length overflow"))?;
            let kv_len = start_position
                .checked_add(suffix_len)
                .and_then(|len| len.checked_mul(KV_VALUES))
                .ok_or_else(|| eyre!("resident suffix attention KV length overflow"))?;
            if q.len < q_len || out.len < q_len {
                return Err(eyre!(
                    "resident suffix attention q/out length mismatch: q {}, out {}, expected at least {q_len}",
                    q.len,
                    out.len
                ));
            }
            if k_cache.len < kv_len || v_cache.len < kv_len {
                return Err(eyre!(
                    "resident suffix attention K/V length mismatch: k {}, v {}, expected at least {kv_len}",
                    k_cache.len,
                    v_cache.len
                ));
            }
            if sinks.len != Q_HEADS {
                return Err(eyre!(
                    "resident suffix attention sinks has {} values, expected {Q_HEADS}",
                    sinks.len
                ));
            }

            let layer = layer as u32;
            let start_position = start_position as u32;
            let suffix_len = suffix_len as u32;

            let encoder = self.encoder("suffix_sequence_attention");
            encoder.set_compute_pipeline_state(&self.context.suffix_sequence_attention);
            encoder.set_buffer(0, Some(&q.buffer), 0);
            encoder.set_buffer(1, Some(&k_cache.buffer), 0);
            encoder.set_buffer(2, Some(&v_cache.buffer), 0);
            encoder.set_buffer(3, Some(&sinks.buffer), 0);
            encoder.set_buffer(4, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 5, &[start_position]);
            self.set_scalar(&encoder, 6, &[suffix_len]);
            self.set_scalar(&encoder, 7, &[layer]);
            encoder.dispatch_thread_groups(
                mtl_size(Q_HEADS as NSUInteger, suffix_len as NSUInteger, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
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

            let encoder = self.encoder("kv_cache_decode_attention");
            encoder.set_compute_pipeline_state(&self.context.kv_cache_decode_attention);
            encoder.set_buffer(0, Some(&q.buffer), 0);
            encoder.set_buffer(1, Some(&k_cache.buffer), 0);
            encoder.set_buffer(2, Some(&v_cache.buffer), 0);
            encoder.set_buffer(3, Some(&sinks.buffer), 0);
            encoder.set_buffer(4, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 5, &[layer]);
            self.set_scalar(&encoder, 6, &[query_position]);
            self.set_scalar(&encoder, 7, &[cache_start_position]);
            self.set_scalar(&encoder, 8, &[cache_len]);
            encoder.dispatch_thread_groups(
                mtl_size(Q_HEADS as NSUInteger, 1, 1),
                mtl_size(DECODE_ATTENTION_THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
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

            let encoder = self.encoder("top4_softmax");
            encoder.set_compute_pipeline_state(&self.context.top4_softmax);
            encoder.set_buffer(0, Some(&logits.buffer), 0);
            encoder.set_buffer(1, Some(&indices.buffer), 0);
            encoder.set_buffer(2, Some(&selected_logits.buffer), 0);
            encoder.set_buffer(3, Some(&weights.buffer), 0);
            self.set_scalar(&encoder, 4, &[n]);
            encoder.dispatch_threads(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn top4_softmax_batch_into(
            &self,
            logits: &F32VectorBuffer,
            indices: &U32Buffer,
            selected_logits: &F32VectorBuffer,
            weights: &F32VectorBuffer,
            rows: usize,
            experts: usize,
        ) -> Result<()> {
            if rows == 0 {
                return Ok(());
            }
            if experts < 4 {
                return Err(eyre!(
                    "batched top4_softmax needs at least 4 experts, got {experts}"
                ));
            }
            let logits_expected = rows
                .checked_mul(experts)
                .ok_or_else(|| eyre!("batched top4 logits length overflow"))?;
            let output_expected = rows
                .checked_mul(4)
                .ok_or_else(|| eyre!("batched top4 output length overflow"))?;
            if logits.len < logits_expected
                || indices.len < output_expected
                || selected_logits.len < output_expected
                || weights.len < output_expected
            {
                return Err(eyre!(
                    "batched top4 shape mismatch: logits {}, indices {}, selected {}, weights {}, expected logits at least {logits_expected} and outputs at least {output_expected}",
                    logits.len,
                    indices.len,
                    selected_logits.len,
                    weights.len
                ));
            }

            let rows = rows as u32;
            let experts = experts as u32;

            let encoder = self.encoder("top4_softmax_batch");
            encoder.set_compute_pipeline_state(&self.context.top4_softmax_batch);
            encoder.set_buffer(0, Some(&logits.buffer), 0);
            encoder.set_buffer(1, Some(&indices.buffer), 0);
            encoder.set_buffer(2, Some(&selected_logits.buffer), 0);
            encoder.set_buffer(3, Some(&weights.buffer), 0);
            self.set_scalar(&encoder, 4, &[rows]);
            self.set_scalar(&encoder, 5, &[experts]);
            encoder.dispatch_threads(mtl_size(rows as NSUInteger, 1, 1), mtl_size(1, 1, 1));
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn mxfp4_gguf_top4_gate_swiglu_into(
            &self,
            gate: &Mxfp4ExpertTensorBuffer,
            up: &Mxfp4ExpertTensorBuffer,
            gate_bias: &F32MatrixBuffer,
            up_bias: &F32MatrixBuffer,
            input: &F32VectorBuffer,
            top_indices: &U32Buffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if gate.experts != EXPERTS || up.experts != EXPERTS {
                return Err(eyre!(
                    "GGUF MXFP4 gate/up expects {EXPERTS} experts, got {}/{}",
                    gate.experts,
                    up.experts
                ));
            }
            if gate.rows != HIDDEN_SIZE
                || up.rows != HIDDEN_SIZE
                || gate.cols != HIDDEN_SIZE
                || up.cols != HIDDEN_SIZE
            {
                return Err(eyre!(
                    "GGUF MXFP4 gate/up shape mismatch: gate {}x{}, up {}x{}, expected {HIDDEN_SIZE}x{HIDDEN_SIZE}",
                    gate.rows,
                    gate.cols,
                    up.rows,
                    up.cols
                ));
            }
            if gate_bias.rows != EXPERTS
                || up_bias.rows != EXPERTS
                || gate_bias.cols != HIDDEN_SIZE
                || up_bias.cols != HIDDEN_SIZE
            {
                return Err(eyre!(
                    "GGUF MXFP4 gate/up bias shape mismatch: gate {}x{}, up {}x{}, expected {EXPERTS}x{HIDDEN_SIZE}",
                    gate_bias.rows,
                    gate_bias.cols,
                    up_bias.rows,
                    up_bias.cols
                ));
            }
            if input.len != HIDDEN_SIZE || top_indices.len < 4 || out.len != 4 * HIDDEN_SIZE {
                return Err(eyre!(
                    "GGUF MXFP4 gate-up fused shape mismatch: input {}, top_indices {}, out {}",
                    input.len,
                    top_indices.len,
                    out.len
                ));
            }

            let encoder = self.encoder("mxfp4_gguf_top4_gate_swiglu");
            encoder.set_compute_pipeline_state(&self.context.mxfp4_gguf_top4_gate_swiglu);
            encoder.set_buffer(0, Some(&gate.buffer), 0);
            encoder.set_buffer(1, Some(&up.buffer), 0);
            encoder.set_buffer(2, Some(&gate_bias.buffer), 0);
            encoder.set_buffer(3, Some(&up_bias.buffer), 0);
            encoder.set_buffer(4, Some(&input.buffer), 0);
            encoder.set_buffer(5, Some(&top_indices.buffer), 0);
            encoder.set_buffer(6, Some(&out.buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(HIDDEN_SIZE.div_ceil(4) as NSUInteger, 4, 1),
                mtl_size(64, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn mxfp4_gguf_top4_down_slots_into(
            &self,
            down: &Mxfp4ExpertTensorBuffer,
            down_bias: &F32MatrixBuffer,
            expert_acts: &F32VectorBuffer,
            top_indices: &U32Buffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if down.experts != EXPERTS || down.rows != HIDDEN_SIZE || down.cols != HIDDEN_SIZE {
                return Err(eyre!(
                    "GGUF MXFP4 down-slots shape is experts={}, rows={}, cols={}, expected {EXPERTS}x{HIDDEN_SIZE}x{HIDDEN_SIZE}",
                    down.experts,
                    down.rows,
                    down.cols
                ));
            }
            if down_bias.rows != EXPERTS || down_bias.cols != HIDDEN_SIZE {
                return Err(eyre!(
                    "GGUF MXFP4 down-slots bias shape is {}x{}, expected {EXPERTS}x{HIDDEN_SIZE}",
                    down_bias.rows,
                    down_bias.cols
                ));
            }
            if expert_acts.len != 4 * HIDDEN_SIZE
                || top_indices.len < 4
                || out.len != 4 * HIDDEN_SIZE
            {
                return Err(eyre!(
                    "GGUF MXFP4 down-slots shape mismatch: expert_acts {}, top_indices {}, out {}",
                    expert_acts.len,
                    top_indices.len,
                    out.len
                ));
            }

            let encoder = self.encoder("mxfp4_gguf_top4_down_slots");
            encoder.set_compute_pipeline_state(&self.context.mxfp4_gguf_top4_down_slots);
            encoder.set_buffer(0, Some(&down.buffer), 0);
            encoder.set_buffer(1, Some(&down_bias.buffer), 0);
            encoder.set_buffer(2, Some(&expert_acts.buffer), 0);
            encoder.set_buffer(3, Some(&top_indices.buffer), 0);
            encoder.set_buffer(4, Some(&out.buffer), 0);
            encoder.dispatch_thread_groups(
                mtl_size(HIDDEN_SIZE.div_ceil(4) as NSUInteger, 4, 1),
                mtl_size(64, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        pub fn weighted_sum4_residual_into(
            &self,
            vectors: &F32VectorBuffer,
            weights: &F32VectorBuffer,
            residual: &F32VectorBuffer,
            out: &F32VectorBuffer,
        ) -> Result<()> {
            if vectors.len != 4 * HIDDEN_SIZE
                || weights.len < 4
                || residual.len != HIDDEN_SIZE
                || out.len != HIDDEN_SIZE
            {
                return Err(eyre!(
                    "weighted_sum4_residual shape mismatch: vectors {}, weights {}, residual {}, out {}",
                    vectors.len,
                    weights.len,
                    residual.len,
                    out.len
                ));
            }

            let encoder = self.encoder("weighted_sum4_residual");
            encoder.set_compute_pipeline_state(&self.context.weighted_sum4_residual);
            encoder.set_buffer(0, Some(&vectors.buffer), 0);
            encoder.set_buffer(1, Some(&weights.buffer), 0);
            encoder.set_buffer(2, Some(&residual.buffer), 0);
            encoder.set_buffer(3, Some(&out.buffer), 0);
            encoder.dispatch_threads(
                mtl_size(HIDDEN_SIZE as NSUInteger, 1, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn mxfp4_gguf_top4_gate_swiglu_batch_into(
            &self,
            gate: &Mxfp4ExpertTensorBuffer,
            up: &Mxfp4ExpertTensorBuffer,
            gate_bias: &F32MatrixBuffer,
            up_bias: &F32MatrixBuffer,
            input: &F32VectorBuffer,
            top_indices: &U32Buffer,
            out: &F32VectorBuffer,
            row_offset: usize,
            rows: usize,
        ) -> Result<()> {
            if gate.experts != EXPERTS || up.experts != EXPERTS {
                return Err(eyre!(
                    "batched GGUF MXFP4 gate/up expects {EXPERTS} experts, got {}/{}",
                    gate.experts,
                    up.experts
                ));
            }
            if gate.rows != HIDDEN_SIZE
                || up.rows != HIDDEN_SIZE
                || gate.cols != HIDDEN_SIZE
                || up.cols != HIDDEN_SIZE
            {
                return Err(eyre!(
                    "batched GGUF MXFP4 gate/up shape mismatch: gate {}x{}, up {}x{}, expected {HIDDEN_SIZE}x{HIDDEN_SIZE}",
                    gate.rows,
                    gate.cols,
                    up.rows,
                    up.cols
                ));
            }
            if gate_bias.rows != EXPERTS
                || up_bias.rows != EXPERTS
                || gate_bias.cols != HIDDEN_SIZE
                || up_bias.cols != HIDDEN_SIZE
            {
                return Err(eyre!(
                    "batched GGUF MXFP4 gate/up bias shape mismatch: gate {}x{}, up {}x{}, expected {EXPERTS}x{HIDDEN_SIZE}",
                    gate_bias.rows,
                    gate_bias.cols,
                    up_bias.rows,
                    up_bias.cols
                ));
            }
            let input_expected = row_offset
                .checked_add(rows)
                .and_then(|rows| rows.checked_mul(HIDDEN_SIZE))
                .ok_or_else(|| eyre!("batched GGUF MXFP4 gate-up input length overflow"))?;
            let top_expected = row_offset
                .checked_add(rows)
                .and_then(|rows| rows.checked_mul(4))
                .ok_or_else(|| eyre!("batched GGUF MXFP4 gate-up top-index length overflow"))?;
            let out_expected = rows
                .checked_mul(4)
                .and_then(|values| values.checked_mul(HIDDEN_SIZE))
                .ok_or_else(|| eyre!("batched GGUF MXFP4 gate-up output length overflow"))?;
            if input.len < input_expected
                || top_indices.len < top_expected
                || out.len < out_expected
            {
                return Err(eyre!(
                    "batched GGUF MXFP4 gate-up shape mismatch: input {}, top_indices {}, out {}, expected input/top/out at least {input_expected}/{top_expected}/{out_expected}",
                    input.len,
                    top_indices.len,
                    out.len
                ));
            }
            if rows == 0 {
                return Ok(());
            }

            let row_offset = row_offset as u32;
            let rows = rows as u32;

            let encoder = self.encoder("mxfp4_gguf_top4_gate_swiglu_batch");
            encoder.set_compute_pipeline_state(&self.context.mxfp4_gguf_top4_gate_swiglu_batch);
            encoder.set_buffer(0, Some(&gate.buffer), 0);
            encoder.set_buffer(1, Some(&up.buffer), 0);
            encoder.set_buffer(2, Some(&gate_bias.buffer), 0);
            encoder.set_buffer(3, Some(&up_bias.buffer), 0);
            encoder.set_buffer(4, Some(&input.buffer), 0);
            encoder.set_buffer(5, Some(&top_indices.buffer), 0);
            encoder.set_buffer(6, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 7, &[row_offset]);
            self.set_scalar(&encoder, 8, &[rows]);
            encoder.dispatch_thread_groups(
                mtl_size(HIDDEN_SIZE as NSUInteger, 4, rows as NSUInteger),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn mxfp4_gguf_top4_down_weighted_batch_into(
            &self,
            down: &Mxfp4ExpertTensorBuffer,
            down_bias: &F32MatrixBuffer,
            expert_acts: &F32VectorBuffer,
            top_indices: &U32Buffer,
            top_weights: &F32VectorBuffer,
            residual: &F32VectorBuffer,
            out: &F32VectorBuffer,
            row_offset: usize,
            rows: usize,
        ) -> Result<()> {
            if down.experts != EXPERTS || down.rows != HIDDEN_SIZE || down.cols != HIDDEN_SIZE {
                return Err(eyre!(
                    "batched GGUF MXFP4 down shape is experts={}, rows={}, cols={}, expected {EXPERTS}x{HIDDEN_SIZE}x{HIDDEN_SIZE}",
                    down.experts,
                    down.rows,
                    down.cols
                ));
            }
            if down_bias.rows != EXPERTS || down_bias.cols != HIDDEN_SIZE {
                return Err(eyre!(
                    "batched GGUF MXFP4 down bias shape is {}x{}, expected {EXPERTS}x{HIDDEN_SIZE}",
                    down_bias.rows,
                    down_bias.cols
                ));
            }
            let hidden_expected = row_offset
                .checked_add(rows)
                .and_then(|rows| rows.checked_mul(HIDDEN_SIZE))
                .ok_or_else(|| eyre!("batched GGUF MXFP4 down hidden length overflow"))?;
            let top_expected = row_offset
                .checked_add(rows)
                .and_then(|rows| rows.checked_mul(4))
                .ok_or_else(|| eyre!("batched GGUF MXFP4 down top length overflow"))?;
            let acts_expected = rows
                .checked_mul(4)
                .and_then(|values| values.checked_mul(HIDDEN_SIZE))
                .ok_or_else(|| eyre!("batched GGUF MXFP4 down activation length overflow"))?;
            if expert_acts.len < acts_expected
                || top_indices.len < top_expected
                || top_weights.len < top_expected
                || residual.len < hidden_expected
                || out.len < hidden_expected
            {
                return Err(eyre!(
                    "batched GGUF MXFP4 down shape mismatch: acts {}, top_indices {}, top_weights {}, residual {}, out {}, expected acts/top/hidden at least {acts_expected}/{top_expected}/{hidden_expected}",
                    expert_acts.len,
                    top_indices.len,
                    top_weights.len,
                    residual.len,
                    out.len
                ));
            }
            if rows == 0 {
                return Ok(());
            }

            let row_offset = row_offset as u32;
            let rows = rows as u32;

            let encoder = self.encoder("mxfp4_gguf_top4_down_weighted_batch");
            encoder.set_compute_pipeline_state(&self.context.mxfp4_gguf_top4_down_weighted_batch);
            encoder.set_buffer(0, Some(&down.buffer), 0);
            encoder.set_buffer(1, Some(&down_bias.buffer), 0);
            encoder.set_buffer(2, Some(&expert_acts.buffer), 0);
            encoder.set_buffer(3, Some(&top_indices.buffer), 0);
            encoder.set_buffer(4, Some(&top_weights.buffer), 0);
            encoder.set_buffer(5, Some(&residual.buffer), 0);
            encoder.set_buffer(6, Some(&out.buffer), 0);
            self.set_scalar(&encoder, 7, &[row_offset]);
            self.set_scalar(&encoder, 8, &[rows]);
            encoder.dispatch_thread_groups(
                mtl_size(HIDDEN_SIZE as NSUInteger, rows as NSUInteger, 1),
                mtl_size(THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            self.end_encoder(encoder);
            Ok(())
        }
    }

    #[cfg(feature = "profile")]
    impl CounterSamples {
        fn reserve_encoder_boundary(&mut self, stage: GpuStage) -> Option<(usize, usize)> {
            if !matches!(self.mode, CounterSamplingMode::EncoderBoundary) {
                return None;
            }
            if self.stages.len().saturating_add(2) > self.sample_limit {
                return None;
            }
            let start = self.stages.len();
            self.stages.push(stage);
            let end = self.stages.len();
            self.stages.push(GpuStage::Other);
            Some((start, end))
        }

        fn uses_dispatch_boundary(&self) -> bool {
            matches!(self.mode, CounterSamplingMode::DispatchBoundary)
        }

        fn sample(&mut self, encoder: &MetalComputeCommandEncoder, stage: GpuStage) {
            if !self.uses_dispatch_boundary() {
                return;
            }
            if self.stages.len() >= self.sample_limit {
                return;
            }
            unsafe {
                encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(
                    &self.sample_buffer,
                    self.stages.len() as NSUInteger,
                    true,
                );
            }
            self.stages.push(stage);
        }

        fn resolve(self) -> Vec<(GpuStage, u128)> {
            if self.stages.len() < 2 {
                return Vec::new();
            }
            let Some(data) = (unsafe {
                self.sample_buffer
                    .resolveCounterRange(NSRange::new(0, self.stages.len()))
            }) else {
                return Vec::new();
            };
            let bytes = unsafe { data.as_bytes_unchecked() };
            let timestamp_count = bytes.len() / size_of::<MTLCounterResultTimestamp>();
            if timestamp_count < 2 {
                return Vec::new();
            }
            let timestamp_count = timestamp_count.min(self.stages.len());
            let timestamps = unsafe {
                slice::from_raw_parts(
                    bytes.as_ptr().cast::<MTLCounterResultTimestamp>(),
                    timestamp_count,
                )
            };
            let mut values = HashMap::<usize, u128>::new();
            for index in 0..timestamp_count.saturating_sub(1) {
                let stage = self.stages[index];
                if stage == GpuStage::Other {
                    continue;
                }
                let start = timestamps[index].timestamp;
                let end = timestamps[index + 1].timestamp;
                let ticks = end.saturating_sub(start) as u128;
                let ns = ticks.saturating_mul(1_000_000_000) / self.timestamp_frequency as u128;
                *values.entry(stage.index()).or_default() = values
                    .get(&stage.index())
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(ns);
            }
            let mut values = values
                .into_iter()
                .map(|(index, ns)| (GpuStage::ALL[index], ns))
                .collect::<Vec<_>>();
            values.sort_by_key(|(stage, _)| stage.index());
            values
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
        #[cfg(feature = "profile")]
        fn new_compute_command_encoder_with_samples(
            &self,
            sample_buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>,
            start_sample: usize,
            end_sample: usize,
        ) -> Retained<MetalComputeCommandEncoder>;
        fn wait_until_completed(&self);
    }

    impl MetalCommandBufferExt for MetalCommandBuffer {
        fn new_compute_command_encoder(&self) -> Retained<MetalComputeCommandEncoder> {
            self.computeCommandEncoder()
                .expect("Metal compute encoder allocation failed")
        }

        #[cfg(feature = "profile")]
        fn new_compute_command_encoder_with_samples(
            &self,
            sample_buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>,
            start_sample: usize,
            end_sample: usize,
        ) -> Retained<MetalComputeCommandEncoder> {
            let descriptor = MTLComputePassDescriptor::computePassDescriptor();
            let attachments = descriptor.sampleBufferAttachments();
            let attachment = unsafe { attachments.objectAtIndexedSubscript(0) };
            attachment.setSampleBuffer(Some(sample_buffer));
            unsafe {
                attachment.setStartOfEncoderSampleIndex(start_sample as NSUInteger);
                attachment.setEndOfEncoderSampleIndex(end_sample as NSUInteger);
            }
            self.computeCommandEncoderWithDescriptor(&descriptor)
                .expect("Metal sampled compute encoder allocation failed")
        }

        fn wait_until_completed(&self) {
            self.waitUntilCompleted();
        }
    }

    trait MetalComputeCommandEncoderExt {
        fn set_compute_pipeline_state(&self, state: &Retained<MetalComputePipelineState>);
        fn set_buffer(&self, index: u64, buffer: Option<&Retained<MetalBuffer>>, offset: u64);
        fn set_bytes<T>(&self, index: u64, values: &[T]);
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

        fn set_bytes<T>(&self, index: u64, values: &[T]) {
            let bytes = NonNull::new(values.as_ptr().cast_mut().cast::<c_void>())
                .expect("setBytes value slice must be non-null");
            unsafe {
                self.setBytes_length_atIndex(
                    bytes,
                    size_of_val(values) as NSUInteger,
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

    fn read_buffer_array<T: Copy + Default, const N: usize>(buffer: &MetalBuffer) -> [T; N] {
        let values =
            unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<T>(), N) };
        let mut out = [T::default(); N];
        out.copy_from_slice(values);
        out
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
mod imp {
    use super::GpuStage;
    use eyre::{Result, eyre};
    use std::marker::PhantomData;

    pub struct MetalBatch<'a> {
        _marker: PhantomData<&'a ()>,
    }
    pub struct BatchTiming {
        pub gpu_ns: u128,
        pub counters: BatchCounters,
        #[cfg(feature = "profile")]
        pub gpu_stages: Vec<(GpuStage, u128)>,
    }
    #[derive(Debug, Clone, Copy, Default)]
    pub struct BatchCounters {
        pub command_buffers: usize,
        pub compute_encoders: usize,
        pub dispatches: usize,
        pub scalar_param_buffers: usize,
    }
    pub struct MetalContext;
    #[derive(Clone)]
    pub struct F32VectorBuffer;
    #[derive(Clone)]
    pub struct F32MatrixBuffer;
    #[derive(Clone)]
    pub struct Q8_0MatrixBuffer;
    #[derive(Clone)]
    pub struct Mxfp4ExpertTensorBuffer;
    #[derive(Clone)]
    pub struct U32Buffer;

    impl F32VectorBuffer {
        pub fn len(&self) -> usize {
            0
        }
    }

    impl Q8_0MatrixBuffer {
        pub fn rows(&self) -> usize {
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

        #[cfg(feature = "profile")]
        pub fn counter_sampling_summary(&self) -> String {
            "Metal backend is only available on macOS".to_string()
        }

        pub fn begin_labeled_batch(&self, _label: &str) -> MetalBatch<'_> {
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

        pub fn write_u32_buffer(&self, _buffer: &U32Buffer, _values: &[u32]) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn read_u32_array<const N: usize>(&self, _buffer: &U32Buffer) -> Result<[u32; N]> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn upload_f32_vector_bytes(
            &self,
            _values: &[u8],
            _len: usize,
        ) -> Result<F32VectorBuffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn upload_f32_matrix_bytes(
            &self,
            _values: &[u8],
            _rows: usize,
            _cols: usize,
        ) -> Result<F32MatrixBuffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn upload_q8_0_matrix_bytes(
            &self,
            _values: &[u8],
            _rows: usize,
            _cols: usize,
        ) -> Result<Q8_0MatrixBuffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn upload_mxfp4_expert_tensor_bytes(
            &self,
            _values: &[u8],
            _experts: usize,
            _rows: usize,
            _cols: usize,
        ) -> Result<Mxfp4ExpertTensorBuffer> {
            Err(eyre!("Metal backend is only available on macOS"))
        }
    }

    impl<'a> MetalBatch<'a> {
        pub fn finish(self) -> BatchTiming {
            BatchTiming {
                gpu_ns: 0,
                counters: BatchCounters::default(),
                #[cfg(feature = "profile")]
                gpu_stages: Vec::new(),
            }
        }

        pub fn set_stage(&self, stage: GpuStage) {
            let _ = stage;
        }

        pub fn embedding_lookup_q8_0_into(
            &self,
            _weight: &Q8_0MatrixBuffer,
            _token: usize,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn embedding_lookup_q8_0_batch_into(
            &self,
            _weight: &Q8_0MatrixBuffer,
            _tokens: &U32Buffer,
            _token_count: usize,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn rms_norm_with_partials_into(
            &self,
            _input: &F32VectorBuffer,
            _weight: &F32VectorBuffer,
            _partials: &F32VectorBuffer,
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

        pub fn f32_matrix_matvec_into(
            &self,
            _weight: &F32MatrixBuffer,
            _input: &F32VectorBuffer,
            _bias: &F32VectorBuffer,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn f32_matrix_matvec_batch_into(
            &self,
            _weight: &F32MatrixBuffer,
            _input: &F32VectorBuffer,
            _bias: &F32VectorBuffer,
            _out: &F32VectorBuffer,
            _batch_rows: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn q8_0_matrix_top1_into(
            &self,
            _weight: &Q8_0MatrixBuffer,
            _input: &F32VectorBuffer,
            _logits: &F32VectorBuffer,
            _block_indices: &U32Buffer,
            _block_values: &F32VectorBuffer,
            _out_index: &U32Buffer,
            _out_value: &F32VectorBuffer,
            _sample_result: &U32Buffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn q8_0_matrix_matvec_add_into(
            &self,
            _weight: &Q8_0MatrixBuffer,
            _input: &F32VectorBuffer,
            _bias: &F32VectorBuffer,
            _residual: &F32VectorBuffer,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn q8_0_matrix_matvec_add_batch_into(
            &self,
            _weight: &Q8_0MatrixBuffer,
            _input: &F32VectorBuffer,
            _bias: &F32VectorBuffer,
            _residual: &F32VectorBuffer,
            _out: &F32VectorBuffer,
            _batch_rows: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn q8_0_qkv_matvec_into(
            &self,
            _q_weight: &Q8_0MatrixBuffer,
            _k_weight: &Q8_0MatrixBuffer,
            _v_weight: &Q8_0MatrixBuffer,
            _input: &F32VectorBuffer,
            _q_bias: &F32VectorBuffer,
            _k_bias: &F32VectorBuffer,
            _v_bias: &F32VectorBuffer,
            _q_out: &F32VectorBuffer,
            _k_out: &F32VectorBuffer,
            _v_out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn q8_0_qkv_matvec_batch_into(
            &self,
            _q_weight: &Q8_0MatrixBuffer,
            _k_weight: &Q8_0MatrixBuffer,
            _v_weight: &Q8_0MatrixBuffer,
            _input: &F32VectorBuffer,
            _q_bias: &F32VectorBuffer,
            _k_bias: &F32VectorBuffer,
            _v_bias: &F32VectorBuffer,
            _q_out: &F32VectorBuffer,
            _k_out: &F32VectorBuffer,
            _v_out: &F32VectorBuffer,
            _batch_rows: usize,
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

        pub fn qk_rope_write_cache_into(
            &self,
            _q: &F32VectorBuffer,
            _k: &F32VectorBuffer,
            _v: &F32VectorBuffer,
            _q_out: &F32VectorBuffer,
            _k_cache: &F32VectorBuffer,
            _v_cache: &F32VectorBuffer,
            _position: usize,
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

        pub fn copy_f32_slot_into(
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
        pub fn suffix_sequence_attention_into(
            &self,
            _layer: usize,
            _start_position: usize,
            _suffix_len: usize,
            _q: &F32VectorBuffer,
            _k_cache: &F32VectorBuffer,
            _v_cache: &F32VectorBuffer,
            _sinks: &F32VectorBuffer,
            _out: &F32VectorBuffer,
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

        pub fn top4_softmax_into(
            &self,
            _logits: &F32VectorBuffer,
            _indices: &U32Buffer,
            _selected_logits: &F32VectorBuffer,
            _weights: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn top4_softmax_batch_into(
            &self,
            _logits: &F32VectorBuffer,
            _indices: &U32Buffer,
            _selected_logits: &F32VectorBuffer,
            _weights: &F32VectorBuffer,
            _rows: usize,
            _experts: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        pub fn mxfp4_gguf_top4_gate_swiglu_into(
            &self,
            _gate: &Mxfp4ExpertTensorBuffer,
            _up: &Mxfp4ExpertTensorBuffer,
            _gate_bias: &F32MatrixBuffer,
            _up_bias: &F32MatrixBuffer,
            _input: &F32VectorBuffer,
            _top_indices: &U32Buffer,
            _out: &F32VectorBuffer,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn mxfp4_gguf_top4_gate_swiglu_batch_into(
            &self,
            _gate: &Mxfp4ExpertTensorBuffer,
            _up: &Mxfp4ExpertTensorBuffer,
            _gate_bias: &F32MatrixBuffer,
            _up_bias: &F32MatrixBuffer,
            _input: &F32VectorBuffer,
            _top_indices: &U32Buffer,
            _out: &F32VectorBuffer,
            _row_offset: usize,
            _rows: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn mxfp4_gguf_top4_down_weighted_batch_into(
            &self,
            _down: &Mxfp4ExpertTensorBuffer,
            _down_bias: &F32MatrixBuffer,
            _expert_acts: &F32VectorBuffer,
            _top_indices: &U32Buffer,
            _top_weights: &F32VectorBuffer,
            _residual: &F32VectorBuffer,
            _out: &F32VectorBuffer,
            _row_offset: usize,
            _rows: usize,
        ) -> Result<()> {
            Err(eyre!("Metal backend is only available on macOS"))
        }
    }
}
