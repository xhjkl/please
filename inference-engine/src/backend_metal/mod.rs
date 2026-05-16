use eyre::{Result, eyre};

use crate::model_store::{self, SourceModelReport};
use crate::runtime_core::ExpertScore;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex};
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
const MAX_PREFILL_PROBE_TOKENS: usize = 128;
const MAX_KV_CACHE_PROBE_TOKENS: usize = 256;
const LM_HEAD_TOP1_BLOCK_SIZE: usize = 256;

mod generation;
mod probes;
mod profile;
mod runtime;
mod weights;

pub use generation::MetalEngine;
pub use probes::*;
pub(crate) use probes::{decode_token_text, decode_tokens_text, metal_sampler_description};
use probes::{read_mxfp4_expert_blocks_metal, read_mxfp4_expert_scales_metal};
#[cfg(feature = "metal-stage-profile")]
use profile::StageProfileState;
pub use profile::{MetalProfileRecord, MetalProfileReport};
use profile::{ProfileDelta, ProfileState, StageMarker, TokenStage, stage_marker};
pub(crate) use runtime::platform;
use weights::{
    ResidentLayerExpertSlabs, ResidentWeights, bf16_linear_profile_name, mxfp4_profile_name,
    mxfp4_slab_blocks_len, mxfp4_slab_scales_len,
};

pub struct MetalOracleContext {
    platform: platform::MetalContext,
    lm_head: Option<platform::Bf16MatrixBuffer>,
    weights: Option<Arc<ResidentWeights>>,
    profile: Arc<Mutex<ProfileState>>,
    gpu_bf16_matrices: Mutex<HashMap<String, platform::Bf16MatrixBuffer>>,
    gpu_bf16_vectors: Mutex<HashMap<String, platform::F32VectorBuffer>>,
    gpu_bf16_rows: Mutex<HashMap<(String, usize), platform::F32VectorBuffer>>,
    gpu_u8_slices: Mutex<HashMap<(String, usize, usize), platform::U8Buffer>>,
}

impl MetalOracleContext {
    pub fn new() -> Result<Self> {
        Ok(Self {
            platform: platform::MetalContext::new()?,
            lm_head: None,
            weights: None,
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
            weights: Some(Arc::new(ResidentWeights::default())),
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
                "Metal lm_head weight is not cached; construct MetalOracleContext::with_lm_head"
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

    pub fn enable_profile(&self) {
        self.platform.set_profile_enabled(true);
        let _ = self.platform.take_gpu_time_ns();
        let mut profile = self.profile.lock().unwrap();
        profile.enabled = true;
        profile.records.clear();
        #[cfg(feature = "metal-stage-profile")]
        {
            profile.stage_profile = None;
        }
    }

    pub fn disable_profile(&self) {
        self.profile.lock().unwrap().enabled = false;
        self.platform.set_profile_enabled(false);
    }

    pub fn profile_report(&self) -> MetalProfileReport {
        let profile = self.profile.lock().unwrap();
        let records = profile.records.values().cloned().collect();
        #[cfg(feature = "metal-stage-profile")]
        let stage_profile = profile
            .stage_profile
            .as_ref()
            .map(StageProfileState::report);
        MetalProfileReport {
            records,
            #[cfg(feature = "metal-stage-profile")]
            stage_profile,
        }
    }

    fn profile_op<T>(
        &self,
        name: &str,
        stats: ProfileDelta,
        run: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.profile.lock().unwrap().enabled {
            return run();
        }

        let _ = self.platform.take_gpu_time_ns();
        let started = Instant::now();
        let result = run();
        let mut stats = stats;
        stats.wall = started.elapsed();
        stats.gpu_ns = self.platform.take_gpu_time_ns();
        self.record_profile(name, stats);
        result
    }

    fn record_profile(&self, name: &str, delta: ProfileDelta) {
        let mut profile = self.profile.lock().unwrap();
        if !profile.enabled {
            return;
        }

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

    #[cfg(feature = "metal-stage-profile")]
    fn reset_stage_profile(&self, ring_capacity: usize) {
        let mut profile = self.profile.lock().unwrap();
        if profile.enabled {
            profile.stage_profile = Some(StageProfileState::new(ring_capacity.max(1)));
        }
    }

    #[cfg(feature = "metal-stage-profile")]
    fn record_token_stage(&self, token_position: usize, stage: TokenStage, ns: u128) {
        if ns == 0 {
            return;
        }
        let mut profile = self.profile.lock().unwrap();
        if !profile.enabled {
            return;
        }
        let Some(stage_profile) = &mut profile.stage_profile else {
            return;
        };
        stage_profile.record(token_position, stage, ns);
    }

    fn rms_norm_profiled(&self, name: &str, x: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        let groups = x.len().div_ceil(256);
        let stats = ProfileDelta {
            command_buffers: 2,
            upload_bytes: (x.len() + weight.len()) * size_of::<f32>()
                + size_of::<u32>()
                + size_of::<f32>(),
            readback_bytes: (groups + x.len()) * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op(name, stats, || self.platform.rms_norm(x, weight))
    }

    fn rope_row_profiled(&self, row: &[f32], heads: usize, position: usize) -> Result<Vec<f32>> {
        let _ = (heads, position);
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: row.len() * size_of::<f32>() + 2 * size_of::<u32>(),
            readback_bytes: row.len() * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.rope", stats, || {
            self.platform.rope_row(row, heads, position)
        })
    }

    fn single_token_attention_profiled(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
    ) -> Result<Vec<f32>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: (q.len() + k.len() + v.len() + sinks.len()) * size_of::<f32>(),
            readback_bytes: q.len() * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.attention.single_token", stats, || {
            self.platform.single_token_attention(q, k, v, sinks)
        })
    }

    fn kv_cache_decode_attention_profiled(
        &self,
        layer: usize,
        query_position: usize,
        cache_start_position: usize,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        sinks: &[f32],
    ) -> Result<Vec<f32>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: (q.len() + k_cache.len() + v_cache.len() + sinks.len())
                * size_of::<f32>()
                + 4 * size_of::<u32>(),
            readback_bytes: q.len() * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.attention.kv_decode", stats, || {
            self.platform.kv_cache_decode_attention(
                layer,
                query_position,
                cache_start_position,
                q,
                k_cache,
                v_cache,
                sinks,
            )
        })
    }

    fn vector_add_profiled(&self, name: &str, left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: (left.len() + right.len()) * size_of::<f32>() + size_of::<u32>(),
            readback_bytes: left.len() * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op(name, stats, || self.platform.vector_add(left, right))
    }

    fn top4_softmax_profiled(&self, logits: &[f32]) -> Result<Vec<ExpertScore>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: logits.len() * size_of::<f32>() + size_of::<u32>(),
            readback_bytes: 4 * (size_of::<u32>() + 2 * size_of::<f32>()),
            ..ProfileDelta::default()
        };
        self.profile_op("op.router.top4", stats, || {
            self.platform.top4_softmax(logits)
        })
    }

    fn swiglu_profiled(&self, values: &[f32]) -> Result<Vec<f32>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: values.len() * size_of::<f32>() + size_of::<u32>(),
            readback_bytes: (values.len() / 2) * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.swiglu", stats, || self.platform.swiglu(values))
    }

    fn weighted_sum4_profiled(&self, vectors: [&[f32]; 4], weights: [f32; 4]) -> Result<Vec<f32>> {
        let n = vectors[0].len();
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: n * 4 * size_of::<f32>()
                + weights.len() * size_of::<f32>()
                + size_of::<u32>(),
            readback_bytes: n * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.moe.weighted_sum4", stats, || {
            self.platform.weighted_sum4(vectors, weights)
        })
    }

    fn bf16_matrix(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
    ) -> Result<Arc<model_store::Bf16Matrix>> {
        let Some(weights) = &self.weights else {
            return Ok(Arc::new(model_store::read_bf16_matrix(
                report,
                tensor_name,
            )?));
        };
        weights.bf16_matrix(report, tensor_name)
    }

    fn bf16_vector(&self, report: &SourceModelReport, tensor_name: &str) -> Result<Arc<Vec<f32>>> {
        let Some(weights) = &self.weights else {
            return Ok(Arc::new(model_store::read_bf16_vector(
                report,
                tensor_name,
            )?));
        };
        weights.bf16_vector(report, tensor_name)
    }

    fn bf16_matrix_row(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        row: usize,
    ) -> Result<Arc<Vec<f32>>> {
        let Some(weights) = &self.weights else {
            return Ok(Arc::new(model_store::read_bf16_matrix_row(
                report,
                tensor_name,
                row,
            )?));
        };
        weights.bf16_matrix_row(report, tensor_name, row)
    }

    fn u8_tensor_slice(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        element_offset: usize,
        element_len: usize,
    ) -> Result<Arc<Vec<u8>>> {
        let Some(weights) = &self.weights else {
            return Ok(Arc::new(model_store::read_u8_tensor_slice(
                report,
                tensor_name,
                element_offset,
                element_len,
            )?));
        };
        weights.u8_tensor_slice(report, tensor_name, element_offset, element_len)
    }

    fn bf16_linear_matvec(
        &self,
        report: &SourceModelReport,
        weight_name: &str,
        bias_name: &str,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let op_name = bf16_linear_profile_name(weight_name);
        self.profile_op(&op_name, ProfileDelta::default(), || {
            self.bf16_linear_matvec_inner(report, weight_name, bias_name, input, &op_name)
        })
    }

    fn bf16_linear_matvec_inner(
        &self,
        report: &SourceModelReport,
        weight_name: &str,
        bias_name: &str,
        input: &[f32],
        op_name: &str,
    ) -> Result<Vec<f32>> {
        if self.weights.is_none() {
            let bias = self.bf16_vector(report, bias_name)?;
            let weight = self.bf16_matrix(report, weight_name)?;
            return self.platform.bf16_matvec(
                &weight.values,
                weight.rows,
                weight.cols,
                input,
                &bias,
            );
        }

        let mut cache = self.gpu_bf16_matrices.lock().unwrap();
        if !cache.contains_key(weight_name) {
            let weight = self.bf16_matrix(report, weight_name)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: weight.values.len() * size_of::<u16>(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let weight =
                self.platform
                    .upload_bf16_matrix(&weight.values, weight.rows, weight.cols)?;
            cache.insert(weight_name.to_string(), weight);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        let Some(weight) = cache.get(weight_name) else {
            return Err(eyre!("cached BF16 matrix is missing for {weight_name}"));
        };

        let mut bias_cache = self.gpu_bf16_vectors.lock().unwrap();
        if !bias_cache.contains_key(bias_name) {
            let bias = self.bf16_vector(report, bias_name)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: bias.len() * size_of::<f32>(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let bias = self.platform.upload_f32_vector(&bias)?;
            bias_cache.insert(bias_name.to_string(), bias);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        let Some(bias) = bias_cache.get(bias_name) else {
            return Err(eyre!("cached BF16 bias is missing for {bias_name}"));
        };
        self.record_profile(
            op_name,
            ProfileDelta {
                command_buffers: 1,
                upload_bytes: input.len() * size_of::<f32>() + 2 * size_of::<u32>(),
                readback_bytes: weight.rows() * size_of::<f32>(),
                ..ProfileDelta::default()
            },
        );
        self.platform.bf16_matrix_matvec(weight, input, bias)
    }

    fn mxfp4_expert_matvec(
        &self,
        report: &SourceModelReport,
        blocks_name: &str,
        scales_name: &str,
        bias_name: &str,
        expert: usize,
        rows: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let op_name = mxfp4_profile_name(bias_name);
        self.profile_op(&op_name, ProfileDelta::default(), || {
            self.mxfp4_expert_matvec_inner(
                report,
                blocks_name,
                scales_name,
                bias_name,
                expert,
                rows,
                input,
                &op_name,
            )
        })
    }

    fn mxfp4_expert_matvec_inner(
        &self,
        report: &SourceModelReport,
        blocks_name: &str,
        scales_name: &str,
        bias_name: &str,
        expert: usize,
        rows: usize,
        input: &[f32],
        op_name: &str,
    ) -> Result<Vec<f32>> {
        if self.weights.is_none() {
            let blocks = read_mxfp4_expert_blocks_metal(self, report, blocks_name, expert, rows)?;
            let scales = read_mxfp4_expert_scales_metal(self, report, scales_name, expert, rows)?;
            let bias = self.bf16_matrix_row(report, bias_name, expert)?;
            return self
                .platform
                .mxfp4_matvec(&blocks, &scales, rows, input, &bias);
        }

        let blocks_per_expert = rows
            .checked_mul(MXFP4_GROUPS)
            .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
            .ok_or_else(|| eyre!("MXFP4 block slice size overflow"))?;
        let blocks_offset = expert
            .checked_mul(blocks_per_expert)
            .ok_or_else(|| eyre!("MXFP4 block offset overflow"))?;
        let scales_per_expert = rows
            .checked_mul(MXFP4_GROUPS)
            .ok_or_else(|| eyre!("MXFP4 scale slice size overflow"))?;
        let scales_offset = expert
            .checked_mul(scales_per_expert)
            .ok_or_else(|| eyre!("MXFP4 scale offset overflow"))?;

        let blocks_key = (blocks_name.to_string(), blocks_offset, blocks_per_expert);
        let scales_key = (scales_name.to_string(), scales_offset, scales_per_expert);
        let mut u8_cache = self.gpu_u8_slices.lock().unwrap();
        if !u8_cache.contains_key(&blocks_key) {
            let blocks =
                self.u8_tensor_slice(report, blocks_name, blocks_offset, blocks_per_expert)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: blocks.len(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let blocks = self.platform.upload_u8_buffer(&blocks)?;
            u8_cache.insert(blocks_key.clone(), blocks);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        if !u8_cache.contains_key(&scales_key) {
            let scales =
                self.u8_tensor_slice(report, scales_name, scales_offset, scales_per_expert)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: scales.len(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let scales = self.platform.upload_u8_buffer(&scales)?;
            u8_cache.insert(scales_key.clone(), scales);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        let Some(blocks) = u8_cache.get(&blocks_key) else {
            return Err(eyre!(
                "cached MXFP4 block slice is missing for {blocks_name} expert {expert}"
            ));
        };
        let Some(scales) = u8_cache.get(&scales_key) else {
            return Err(eyre!(
                "cached MXFP4 scale slice is missing for {scales_name} expert {expert}"
            ));
        };

        let bias_key = (bias_name.to_string(), expert);
        let mut row_cache = self.gpu_bf16_rows.lock().unwrap();
        if !row_cache.contains_key(&bias_key) {
            let bias = self.bf16_matrix_row(report, bias_name, expert)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: bias.len() * size_of::<f32>(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let bias = self.platform.upload_f32_vector(&bias)?;
            row_cache.insert(bias_key.clone(), bias);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        let Some(bias) = row_cache.get(&bias_key) else {
            return Err(eyre!(
                "cached MXFP4 bias row is missing for {bias_name} expert {expert}"
            ));
        };

        self.record_profile(
            op_name,
            ProfileDelta {
                command_buffers: 1,
                upload_bytes: input.len() * size_of::<f32>() + 2 * size_of::<u32>(),
                readback_bytes: rows * size_of::<f32>(),
                ..ProfileDelta::default()
            },
        );
        self.platform
            .mxfp4_matvec_resident(blocks, scales, rows, input, bias)
    }

    fn bf16_matrix_buffer(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        op_name: &str,
    ) -> Result<platform::Bf16MatrixBuffer> {
        let mut cache = self.gpu_bf16_matrices.lock().unwrap();
        if !cache.contains_key(tensor_name) {
            let weight = self.bf16_matrix(report, tensor_name)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: weight.values.len() * size_of::<u16>(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let weight =
                self.platform
                    .upload_bf16_matrix(&weight.values, weight.rows, weight.cols)?;
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

    fn bf16_vector_buffer(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        op_name: &str,
    ) -> Result<platform::F32VectorBuffer> {
        let mut cache = self.gpu_bf16_vectors.lock().unwrap();
        if !cache.contains_key(tensor_name) {
            let weight = self.bf16_vector(report, tensor_name)?;
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

    fn u8_slice_buffer(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        element_offset: usize,
        element_len: usize,
        op_name: &str,
    ) -> Result<platform::U8Buffer> {
        let key = (tensor_name.to_string(), element_offset, element_len);
        let mut cache = self.gpu_u8_slices.lock().unwrap();
        if !cache.contains_key(&key) {
            let value = self.u8_tensor_slice(report, tensor_name, element_offset, element_len)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: value.len(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let value = self.platform.upload_u8_buffer(&value)?;
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

    fn mxfp4_layer_expert_slabs(
        &self,
        report: &SourceModelReport,
        expert_prefix: &str,
    ) -> Result<ResidentLayerExpertSlabs> {
        let gate_up_blocks = self.u8_slice_buffer(
            report,
            &format!("{expert_prefix}.gate_up_proj_blocks"),
            0,
            mxfp4_slab_blocks_len(GATE_UP_VALUES)?,
            "op.mxfp4.gate_up",
        )?;
        let gate_up_scales = self.u8_slice_buffer(
            report,
            &format!("{expert_prefix}.gate_up_proj_scales"),
            0,
            mxfp4_slab_scales_len(GATE_UP_VALUES)?,
            "op.mxfp4.gate_up",
        )?;
        let gate_up_bias = self.bf16_matrix_buffer(
            report,
            &format!("{expert_prefix}.gate_up_proj_bias"),
            "op.mxfp4.gate_up",
        )?;
        let down_blocks = self.u8_slice_buffer(
            report,
            &format!("{expert_prefix}.down_proj_blocks"),
            0,
            mxfp4_slab_blocks_len(HIDDEN_SIZE)?,
            "op.mxfp4.down",
        )?;
        let down_scales = self.u8_slice_buffer(
            report,
            &format!("{expert_prefix}.down_proj_scales"),
            0,
            mxfp4_slab_scales_len(HIDDEN_SIZE)?,
            "op.mxfp4.down",
        )?;
        let down_bias = self.bf16_matrix_buffer(
            report,
            &format!("{expert_prefix}.down_proj_bias"),
            "op.mxfp4.down",
        )?;

        Ok(ResidentLayerExpertSlabs {
            gate_up_blocks,
            gate_up_scales,
            gate_up_bias,
            down_blocks,
            down_scales,
            down_bias,
        })
    }
}
