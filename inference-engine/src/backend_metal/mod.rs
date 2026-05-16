use eyre::{Result, eyre};

use crate::model_store::{self, SafeTensorMap, SourceModelReport};
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex};
#[cfg(feature = "profile")]
use std::time::{Duration, Instant};

const LAYERS: usize = 24;
const Q_HEADS: usize = 64;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const ATTN_VALUES: usize = Q_HEADS * HEAD_DIM;
const KV_VALUES: usize = KV_HEADS * HEAD_DIM;
const HIDDEN_SIZE: usize = 2880;
const GATE_UP_VALUES: usize = HIDDEN_SIZE * 2;
const EXPERTS: usize = 32;
const MXFP4_GROUPS: usize = HIDDEN_SIZE / 32;
const MXFP4_BYTES_PER_GROUP: usize = 16;
const LOCAL_WINDOW_TOKENS: usize = 128;
const MAX_RESIDENT_CONTEXT_TOKENS: usize = 4096;
const MAX_PREFILL_PROBE_TOKENS: usize = 128;
const MAX_KV_CACHE_PROBE_TOKENS: usize = 256;
const LM_HEAD_TOP1_BLOCK_SIZE: usize = 256;

mod generation;
pub mod probes;
mod profile;
mod runtime;
mod weights;

pub use generation::MetalEngine;
pub(crate) use probes::{decode_token_text, decode_tokens_text, metal_sampler_description};
#[cfg(feature = "profile")]
pub use profile::{MetalProfileRecord, MetalProfileReport};
use profile::{ProfileDelta, StageMarker, TokenStage, stage_marker};
#[cfg(feature = "profile")]
use profile::{ProfileState, StageProfileState};
pub(crate) use runtime::platform;
use weights::{ExpertsCarouselSlabs, WeightCache, mxfp4_slab_blocks_len, mxfp4_slab_scales_len};

pub struct MetalRuntime {
    platform: platform::MetalContext,
    lm_head: Option<platform::Bf16MatrixBuffer>,
    weights: Option<Arc<WeightCache>>,
    #[cfg(feature = "profile")]
    profile: Arc<Mutex<ProfileState>>,
    gpu_bf16_matrices: Mutex<HashMap<String, platform::Bf16MatrixBuffer>>,
    gpu_bf16_vectors: Mutex<HashMap<String, platform::F32VectorBuffer>>,
    gpu_bf16_rows: Mutex<HashMap<(String, usize), platform::F32VectorBuffer>>,
    gpu_u8_slices: Mutex<HashMap<(String, usize, usize), platform::U8Buffer>>,
}

impl MetalRuntime {
    pub fn new() -> Result<Self> {
        Ok(Self {
            platform: platform::MetalContext::new()?,
            lm_head: None,
            weights: None,
            #[cfg(feature = "profile")]
            profile: Arc::new(Mutex::new(ProfileState::default())),
            gpu_bf16_matrices: Mutex::new(HashMap::new()),
            gpu_bf16_vectors: Mutex::new(HashMap::new()),
            gpu_bf16_rows: Mutex::new(HashMap::new()),
            gpu_u8_slices: Mutex::new(HashMap::new()),
        })
    }

    pub fn with_lm_head(report: &SourceModelReport) -> Result<Self> {
        let platform = platform::MetalContext::new()?;
        let weight = model_store::read_bf16_matrix(report, "lm_head.weight")?;
        let lm_head = platform.upload_bf16_matrix(&weight.values, weight.rows, weight.cols)?;
        Ok(Self {
            platform,
            lm_head: Some(lm_head),
            weights: Some(Arc::new(WeightCache::default())),
            #[cfg(feature = "profile")]
            profile: Arc::new(Mutex::new(ProfileState::default())),
            gpu_bf16_matrices: Mutex::new(HashMap::new()),
            gpu_bf16_vectors: Mutex::new(HashMap::new()),
            gpu_bf16_rows: Mutex::new(HashMap::new()),
            gpu_u8_slices: Mutex::new(HashMap::new()),
        })
    }

    pub fn with_lm_head_map(source: &SafeTensorMap) -> Result<Self> {
        let platform = platform::MetalContext::new()?;
        let weight = source.bf16_matrix_bytes("lm_head.weight")?;
        let lm_head = platform.upload_bf16_matrix_bytes(weight.bytes, weight.rows, weight.cols)?;
        Ok(Self {
            platform,
            lm_head: Some(lm_head),
            weights: Some(Arc::new(WeightCache::default())),
            #[cfg(feature = "profile")]
            profile: Arc::new(Mutex::new(ProfileState::default())),
            gpu_bf16_matrices: Mutex::new(HashMap::new()),
            gpu_bf16_vectors: Mutex::new(HashMap::new()),
            gpu_bf16_rows: Mutex::new(HashMap::new()),
            gpu_u8_slices: Mutex::new(HashMap::new()),
        })
    }

    fn lm_head_topk(&self, final_hidden: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        let Some(lm_head) = &self.lm_head else {
            return Err(eyre!(
                "Metal lm_head weight is not cached; construct MetalRuntime::with_lm_head"
            ));
        };
        let stats = ProfileDelta {
            command_buffers: 2,
            upload_bytes: final_hidden.len() * size_of::<f32>() + 3 * size_of::<u32>(),
            readback_bytes: k * (size_of::<u32>() + size_of::<f32>()),
            ..ProfileDelta::default()
        };
        self.profile_op("op.lm_head_topk", stats, || {
            self.platform.bf16_matrix_topk(lm_head, final_hidden, k)
        })
    }

    #[cfg(feature = "profile")]
    fn reset_profile(&self) {
        let _ = self.platform.take_gpu_time_ns();
        let mut profile = self.profile.lock().unwrap();
        profile.records.clear();
        profile.stage_profile = None;
    }

    #[cfg(feature = "profile")]
    pub fn profile_report(&self) -> MetalProfileReport {
        let profile = self.profile.lock().unwrap();
        let records = profile.records.values().cloned().collect();
        let stage_profile = profile
            .stage_profile
            .as_ref()
            .map(StageProfileState::report);
        MetalProfileReport {
            records,
            stage_profile,
        }
    }

    #[cfg(feature = "profile")]
    fn profile_op<T>(
        &self,
        name: &str,
        stats: ProfileDelta,
        run: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let _ = self.platform.take_gpu_time_ns();
        let started = Instant::now();
        let result = run();
        let mut stats = stats;
        stats.wall = started.elapsed();
        stats.gpu_ns = self.platform.take_gpu_time_ns();
        self.record_profile(name, stats);
        result
    }

    #[cfg(not(feature = "profile"))]
    fn profile_op<T>(
        &self,
        _name: &str,
        _stats: ProfileDelta,
        run: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        run()
    }

    #[cfg(feature = "profile")]
    fn record_profile(&self, name: &str, delta: ProfileDelta) {
        let mut profile = self.profile.lock().unwrap();
        let record =
            profile
                .records
                .entry(name.to_string())
                .or_insert_with(|| MetalProfileRecord {
                    name: name.to_string(),
                    ..MetalProfileRecord::default()
                });
        record.calls += usize::from(delta.wall > Duration::ZERO);
        record.wall_ns += delta.wall.as_nanos();
        record.gpu_ns += delta.gpu_ns;
        record.command_buffers += delta.command_buffers;
        record.upload_bytes += delta.upload_bytes;
        record.readback_bytes += delta.readback_bytes;
        record.cache_hits += delta.cache_hits;
        record.cache_misses += delta.cache_misses;
    }

    #[cfg(not(feature = "profile"))]
    #[inline(always)]
    fn record_profile(&self, _name: &str, _delta: ProfileDelta) {}

    #[cfg(feature = "profile")]
    fn reset_stage_profile(&self, ring_capacity: usize) {
        let mut profile = self.profile.lock().unwrap();
        profile.stage_profile = Some(StageProfileState::new(ring_capacity.max(1)));
    }

    #[cfg(feature = "profile")]
    fn record_token_stage(&self, token_position: usize, stage: TokenStage, ns: u128) {
        if ns == 0 {
            return;
        }
        let mut profile = self.profile.lock().unwrap();
        let Some(stage_profile) = &mut profile.stage_profile else {
            return;
        };
        stage_profile.record(token_position, stage, ns);
    }

    fn bf16_matrix_buffer_from_map(
        &self,
        source: &SafeTensorMap,
        tensor_name: &str,
        op_name: &str,
    ) -> Result<platform::Bf16MatrixBuffer> {
        let mut cache = self.gpu_bf16_matrices.lock().unwrap();
        if !cache.contains_key(tensor_name) {
            let weight = source.bf16_matrix_bytes(tensor_name)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: weight.bytes.len(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let weight = if uses_tiled4_bf16_linear(op_name) {
                self.platform.upload_bf16_matrix_tiled4_bytes(
                    weight.bytes,
                    weight.rows,
                    weight.cols,
                )?
            } else {
                self.platform
                    .upload_bf16_matrix_bytes(weight.bytes, weight.rows, weight.cols)?
            };
            cache.insert(tensor_name.to_string(), weight);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        cache
            .get(tensor_name)
            .cloned()
            .ok_or_else(|| eyre!("cached BF16 matrix is missing for {tensor_name}"))
    }

    fn bf16_vector_buffer_from_map(
        &self,
        source: &SafeTensorMap,
        tensor_name: &str,
        op_name: &str,
    ) -> Result<platform::F32VectorBuffer> {
        let mut cache = self.gpu_bf16_vectors.lock().unwrap();
        if !cache.contains_key(tensor_name) {
            let weight = if let Some(weights) = &self.weights {
                weights.bf16_vector_from_map(source, tensor_name)?
            } else {
                Arc::new(source.read_bf16_vector(tensor_name)?)
            };
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: weight.len() * size_of::<f32>(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let weight = self.platform.upload_f32_vector(&weight)?;
            cache.insert(tensor_name.to_string(), weight);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        cache
            .get(tensor_name)
            .cloned()
            .ok_or_else(|| eyre!("cached BF16 vector is missing for {tensor_name}"))
    }

    fn u8_slice_buffer_from_map(
        &self,
        source: &SafeTensorMap,
        tensor_name: &str,
        element_offset: usize,
        element_len: usize,
        op_name: &str,
    ) -> Result<platform::U8Buffer> {
        let key = (tensor_name.to_string(), element_offset, element_len);
        let mut cache = self.gpu_u8_slices.lock().unwrap();
        if !cache.contains_key(&key) {
            let value = source.u8_tensor_slice_bytes(tensor_name, element_offset, element_len)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: value.len(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let value = self.platform.upload_u8_buffer(value)?;
            cache.insert(key.clone(), value);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        cache
            .get(&key)
            .cloned()
            .ok_or_else(|| eyre!("cached u8 slice is missing for {tensor_name}"))
    }

    fn mxfp4_experts_carousel_slabs_from_map(
        &self,
        source: &SafeTensorMap,
        expert_prefix: &str,
    ) -> Result<ExpertsCarouselSlabs> {
        let gate_up_blocks = self.u8_slice_buffer_from_map(
            source,
            &format!("{expert_prefix}.gate_up_proj_blocks"),
            0,
            mxfp4_slab_blocks_len(GATE_UP_VALUES)?,
            "op.mxfp4.gate_up",
        )?;
        let gate_up_scales = self.u8_slice_buffer_from_map(
            source,
            &format!("{expert_prefix}.gate_up_proj_scales"),
            0,
            mxfp4_slab_scales_len(GATE_UP_VALUES)?,
            "op.mxfp4.gate_up",
        )?;
        let gate_up_bias = self.bf16_matrix_buffer_from_map(
            source,
            &format!("{expert_prefix}.gate_up_proj_bias"),
            "op.mxfp4.gate_up",
        )?;
        let down_blocks = self.u8_slice_buffer_from_map(
            source,
            &format!("{expert_prefix}.down_proj_blocks"),
            0,
            mxfp4_slab_blocks_len(HIDDEN_SIZE)?,
            "op.mxfp4.down",
        )?;
        let down_scales = self.u8_slice_buffer_from_map(
            source,
            &format!("{expert_prefix}.down_proj_scales"),
            0,
            mxfp4_slab_scales_len(HIDDEN_SIZE)?,
            "op.mxfp4.down",
        )?;
        let down_bias = self.bf16_matrix_buffer_from_map(
            source,
            &format!("{expert_prefix}.down_proj_bias"),
            "op.mxfp4.down",
        )?;

        Ok(ExpertsCarouselSlabs {
            gate_up_blocks,
            gate_up_scales,
            gate_up_bias,
            down_blocks,
            down_scales,
            down_bias,
        })
    }
}

fn uses_tiled4_bf16_linear(op_name: &str) -> bool {
    matches!(
        op_name,
        "op.bf16.q_proj"
            | "op.bf16.k_proj"
            | "op.bf16.v_proj"
            | "op.bf16.o_proj"
            | "op.bf16.router"
    )
}
