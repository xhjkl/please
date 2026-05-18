use eyre::Result;

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
pub use profile::{MetalProfile, MetalProfileRecord};
#[cfg(feature = "profile")]
use profile::{ProfileState, StageProfileState};

pub struct MetalRuntime {
    platform: platform::MetalContext,
    #[cfg(feature = "profile")]
    profile: Arc<Mutex<ProfileState>>,
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
