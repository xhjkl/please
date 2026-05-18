use eyre::Result;
use std::fmt;

use crate::model_store::gguf::{
    F32MatrixBytes, F32VectorBytes, Mxfp4ExpertTensorBytes, Q8_0MatrixBytes,
};
#[cfg(feature = "profile")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "profile")]
use std::time::Duration;

const LAYERS: usize = 24;
const Q_HEADS: usize = 64;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const ATTN_VALUES: usize = Q_HEADS * HEAD_DIM;
const KV_VALUES: usize = KV_HEADS * HEAD_DIM;
const HIDDEN_SIZE: usize = 2880;
const EXPERTS: usize = 32;
const LOCAL_WINDOW_TOKENS: usize = 128;
const MAX_RESIDENT_CONTEXT_TOKENS: usize = 4096;
const LM_HEAD_TOP1_BLOCK_SIZE: usize = 256;

mod generation;
mod platform;
mod profile;
mod weights;

pub use generation::{MetalEpisode, MetalModel, MetalTimings, TimedGenerationStream};
use profile::{GpuStage, ProfileDelta, StageMarker, TokenStage, stage_marker};
#[cfg(feature = "profile")]
pub use profile::{MetalProfile, MetalProfileRecord, MetalStageEnvelopeRow};
#[cfg(feature = "profile")]
use profile::{ProfileState, StageProfileState};

pub struct MetalRuntime {
    platform: platform::MetalContext,
    #[cfg(feature = "profile")]
    profile: Arc<Mutex<ProfileState>>,
}

#[derive(Debug, Clone)]
pub struct AttentionProbeReport {
    pub cases: Vec<AttentionProbeCase>,
}

#[derive(Debug, Clone)]
pub struct AttentionProbeCase {
    pub layer: usize,
    pub position: usize,
    pub max_abs_delta: f32,
    pub mean_abs_delta: f32,
    pub nonfinite_values: usize,
    pub gpu_ns: u128,
}

impl fmt::Display for AttentionProbeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "attention correctness probe:")?;
        writeln!(
            f,
            "{:>5}  {:>8}  {:>14}  {:>15}  {:>10}  {:>10}",
            "layer", "position", "max_abs_delta", "mean_abs_delta", "nonfinite", "gpu"
        )?;
        writeln!(
            f,
            "{:>5}  {:>8}  {:>14}  {:>15}  {:>10}  {:>10}",
            "-----", "--------", "-------------", "--------------", "---------", "---"
        )?;
        for case in &self.cases {
            writeln!(
                f,
                "{:>5}  {:>8}  {:>14.6e}  {:>15.6e}  {:>10}  {:>10.1}us",
                case.layer,
                case.position,
                case.max_abs_delta,
                case.mean_abs_delta,
                case.nonfinite_values,
                case.gpu_ns as f64 / 1_000.0
            )?;
        }
        Ok(())
    }
}

impl MetalRuntime {
    pub fn new() -> Result<Self> {
        Ok(Self {
            platform: platform::MetalContext::new()?,
            #[cfg(feature = "profile")]
            profile: Arc::new(Mutex::new(ProfileState::default())),
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
    pub fn profile_report(&self) -> MetalProfile {
        let profile = self.profile.lock().unwrap();
        let records = profile.records.values().cloned().collect();
        let stage_profile = profile
            .stage_profile
            .as_ref()
            .map(StageProfileState::snapshot);
        MetalProfile {
            records,
            stage_profile,
            counter_sampling: self.platform.counter_sampling_summary(),
        }
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

    #[cfg(feature = "profile")]
    fn record_gpu_stages(&self, token_position: usize, stages: Vec<(GpuStage, u128)>) {
        if stages.is_empty() {
            return;
        }
        let mut profile = self.profile.lock().unwrap();
        let Some(stage_profile) = &mut profile.stage_profile else {
            return;
        };
        for (stage, ns) in stages {
            stage_profile.record_gpu_stage(token_position, stage, ns);
        }
    }

    pub(crate) fn gguf_f32_vector_buffer(
        &self,
        value: F32VectorBytes<'_>,
        op_name: &str,
    ) -> Result<platform::F32VectorBuffer> {
        self.record_profile(
            op_name,
            ProfileDelta {
                upload_bytes: value.bytes.len(),
                cache_misses: 1,
                ..ProfileDelta::default()
            },
        );
        self.platform
            .upload_f32_vector_bytes(value.bytes, value.len)
    }

    pub(crate) fn gguf_f32_matrix_buffer(
        &self,
        value: F32MatrixBytes<'_>,
        op_name: &str,
    ) -> Result<platform::F32MatrixBuffer> {
        self.record_profile(
            op_name,
            ProfileDelta {
                upload_bytes: value.bytes.len(),
                cache_misses: 1,
                ..ProfileDelta::default()
            },
        );
        self.platform
            .upload_f32_matrix_bytes(value.bytes, value.rows, value.cols)
    }

    pub(crate) fn gguf_q8_0_matrix_buffer(
        &self,
        value: Q8_0MatrixBytes<'_>,
        op_name: &str,
    ) -> Result<platform::Q8_0MatrixBuffer> {
        self.record_profile(
            op_name,
            ProfileDelta {
                upload_bytes: value.bytes.len(),
                cache_misses: 1,
                ..ProfileDelta::default()
            },
        );
        self.platform
            .upload_q8_0_matrix_bytes(value.bytes, value.rows, value.cols)
    }

    pub(crate) fn gguf_mxfp4_expert_tensor_buffer(
        &self,
        value: Mxfp4ExpertTensorBytes<'_>,
        op_name: &str,
    ) -> Result<platform::Mxfp4ExpertTensorBuffer> {
        self.record_profile(
            op_name,
            ProfileDelta {
                upload_bytes: value.bytes.len(),
                cache_misses: 1,
                ..ProfileDelta::default()
            },
        );
        self.platform.upload_mxfp4_expert_tensor_bytes(
            value.bytes,
            value.experts,
            value.rows,
            value.cols,
        )
    }
}

pub fn run_attention_probe() -> Result<AttentionProbeReport> {
    let runtime = MetalRuntime::new()?;
    let layers = [0usize, 1, 3];
    let positions = [127usize, 128, 129, 512, 2048, 4080];
    let mut cases = Vec::with_capacity(layers.len() * positions.len());

    for layer in layers {
        for position in positions {
            let cache_len = position + 1;
            let q =
                deterministic_f32_values(ATTN_VALUES, layer as u64 * 17 + position as u64, 0.08);
            let k = deterministic_f32_values(
                cache_len * KV_VALUES,
                layer as u64 * 101 + position as u64 * 7 + 1,
                0.06,
            );
            let v = deterministic_f32_values(
                cache_len * KV_VALUES,
                layer as u64 * 211 + position as u64 * 13 + 2,
                0.20,
            );
            let sinks = deterministic_f32_values(Q_HEADS, layer as u64 * 307 + 3, 0.03);

            let q_buffer = runtime.platform.alloc_f32_vector(ATTN_VALUES)?;
            let k_buffer = runtime.platform.alloc_f32_vector(cache_len * KV_VALUES)?;
            let v_buffer = runtime.platform.alloc_f32_vector(cache_len * KV_VALUES)?;
            let sinks_buffer = runtime.platform.alloc_f32_vector(Q_HEADS)?;
            let production_out = runtime.platform.alloc_f32_vector(ATTN_VALUES)?;
            let serial_out = runtime.platform.alloc_f32_vector(ATTN_VALUES)?;

            runtime.platform.write_f32_buffer(&q_buffer, &q)?;
            runtime.platform.write_f32_buffer(&k_buffer, &k)?;
            runtime.platform.write_f32_buffer(&v_buffer, &v)?;
            runtime.platform.write_f32_buffer(&sinks_buffer, &sinks)?;

            let batch = runtime
                .platform
                .begin_labeled_batch("probe.attention_correctness");
            batch.kv_cache_decode_attention_into(
                layer,
                position,
                0,
                cache_len,
                &q_buffer,
                &k_buffer,
                &v_buffer,
                &sinks_buffer,
                &production_out,
            )?;
            batch.kv_cache_decode_attention_serial_probe_into(
                layer,
                position,
                0,
                cache_len,
                &q_buffer,
                &k_buffer,
                &v_buffer,
                &sinks_buffer,
                &serial_out,
            )?;
            let timing = batch.finish();

            let production = runtime.platform.read_f32_vector(&production_out);
            let serial = runtime.platform.read_f32_vector(&serial_out);
            let mut max_abs_delta = 0.0f32;
            let mut total_abs_delta = 0.0f64;
            let mut nonfinite_values = 0usize;
            for (production, serial) in production.iter().zip(&serial) {
                if !production.is_finite() {
                    nonfinite_values += 1;
                }
                if !serial.is_finite() {
                    nonfinite_values += 1;
                }
                let delta = (*production - *serial).abs();
                if !delta.is_finite() {
                    nonfinite_values += 1;
                    continue;
                }
                max_abs_delta = max_abs_delta.max(delta);
                total_abs_delta += delta as f64;
            }

            cases.push(AttentionProbeCase {
                layer,
                position,
                max_abs_delta,
                mean_abs_delta: (total_abs_delta / ATTN_VALUES as f64) as f32,
                nonfinite_values,
                gpu_ns: timing.gpu_ns,
            });
        }
    }

    Ok(AttentionProbeReport { cases })
}

fn deterministic_f32_values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as u32) as f32 / 16_777_215.0;
            (unit - 0.5) * scale
        })
        .collect()
}
