use eyre::{Result, eyre};

use crate::backend_cpu;
use crate::gptoss_spec::weights;
use crate::harmony_adapter::HarmonyAdapter;
use crate::model_store::{self, SourceModelReport};
use crate::runtime_core::kv_cache::{KvCachePlan, PlannedKvCache};
use crate::runtime_core::sampler::{SampleCandidate, Sampler};
use crate::runtime_core::{
    EngineRequest, ExpertScore, GenerationEvent, GreedyDecodeProbeReport, GreedyTextProbeReport,
    GreedyTokenReport, LmHeadTopKProbeReport, MetalMatvecProbeReport, MetalRmsNormProbeReport,
    MetalSelectedLogitsProbeReport, MetalTopKProbeReport, MetalVectorProbeReport, RuntimeNotice,
    SamplingConfig, SelectedLogit, StopReason,
};
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

#[derive(Debug, Clone)]
pub struct MetalProfileReport {
    pub records: Vec<MetalProfileRecord>,
    #[cfg(feature = "metal-stage-profile")]
    pub stage_profile: Option<MetalStageProfileReport>,
}

impl MetalProfileReport {
    pub fn render_for_cli(&self) -> String {
        let mut records = self.records.clone();
        records.sort_by(|left, right| {
            right
                .wall_ns
                .cmp(&left.wall_ns)
                .then_with(|| left.name.cmp(&right.name))
        });

        let total_wall_ns = records
            .iter()
            .find(|record| record.name == "phase.generate")
            .map(|record| record.wall_ns)
            .unwrap_or_else(|| records.iter().map(|record| record.wall_ns).sum());

        let mut out = String::new();
        out.push_str("\nmetal runtime profile:\n");
        out.push_str(&format!(
            "- recorded wall: {}\n",
            format_duration_ns(total_wall_ns)
        ));
        out.push_str("- gpu: Metal command-buffer GPU timestamps where available\n");
        out.push_str("\n");
        out.push_str(
            "pct     wall      gpu       calls  cb     upload    readback  cache       name\n",
        );
        out.push_str("-----   -------   -------   -----  -----  --------  --------  ----------  -------------------------\n");
        for record in records
            .iter()
            .filter(|record| record.name != "phase.generate")
        {
            if record.wall_ns == 0
                && record.upload_bytes == 0
                && record.readback_bytes == 0
                && record.command_buffers == 0
                && record.cache_hits == 0
                && record.cache_misses == 0
            {
                continue;
            }
            let percent = if total_wall_ns == 0 {
                0.0
            } else {
                (record.wall_ns as f64 / total_wall_ns as f64) * 100.0
            };
            out.push_str(&format!(
                "{percent:>5.1}%  {:>7}  {:>7}  {:>5}  {:>5}  {:>8}  {:>8}  {:>4}/{:<4}   {}\n",
                format_duration_ns(record.wall_ns),
                format_duration_ns(record.gpu_ns),
                record.calls,
                record.command_buffers,
                format_bytes(record.upload_bytes),
                format_bytes(record.readback_bytes),
                record.cache_hits,
                record.cache_misses,
                record.name
            ));
        }
        #[cfg(feature = "metal-stage-profile")]
        if let Some(stage_profile) = &self.stage_profile {
            out.push_str(&stage_profile.render_for_cli());
        }
        out
    }
}

#[cfg(feature = "metal-stage-profile")]
#[derive(Debug, Clone)]
pub struct MetalStageProfileReport {
    token_positions: Vec<Option<usize>>,
    stage_names: Vec<&'static str>,
    values_ns: Vec<Vec<u128>>,
}

#[cfg(feature = "metal-stage-profile")]
impl MetalStageProfileReport {
    fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\nmetal token-stage profile:\n");
        out.push_str("- source: per-batch Metal command-buffer GPU timestamps\n");
        out.push_str("- layout: profile[token_ring_slot][stage] = nanoseconds\n\n");

        out.push_str("slot   token      ");
        for name in &self.stage_names {
            out.push_str(&format!("{name:>14}"));
        }
        out.push('\n');
        out.push_str("-----  ---------  ");
        for _ in &self.stage_names {
            out.push_str("--------------");
        }
        out.push('\n');

        for (slot, position) in self.token_positions.iter().enumerate() {
            let Some(position) = position else {
                continue;
            };
            out.push_str(&format!("{slot:>5}  {position:>9}  "));
            for stage in 0..self.stage_names.len() {
                out.push_str(&format!(
                    "{:>14}",
                    format_duration_ns(self.values_ns[slot][stage])
                ));
            }
            out.push('\n');
        }
        out
    }
}

#[derive(Debug, Clone, Default)]
pub struct MetalProfileRecord {
    pub name: String,
    pub calls: usize,
    pub wall_ns: u128,
    pub gpu_ns: u128,
    pub command_buffers: usize,
    pub upload_bytes: usize,
    pub readback_bytes: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct ProfileDelta {
    wall: Duration,
    gpu_ns: u128,
    command_buffers: usize,
    upload_bytes: usize,
    readback_bytes: usize,
    cache_hits: usize,
    cache_misses: usize,
}

#[derive(Default)]
struct ProfileState {
    enabled: bool,
    records: HashMap<String, MetalProfileRecord>,
    #[cfg(feature = "metal-stage-profile")]
    stage_profile: Option<StageProfileState>,
}

#[cfg(feature = "metal-stage-profile")]
#[derive(Debug, Clone, Copy)]
enum TokenStage {
    Token,
    LmHead,
}

#[cfg(feature = "metal-stage-profile")]
impl TokenStage {
    const ALL: [Self; 2] = [Self::Token, Self::LmHead];

    fn index(self) -> usize {
        match self {
            Self::Token => 0,
            Self::LmHead => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::LmHead => "lm_head",
        }
    }
}

#[cfg(feature = "metal-stage-profile")]
#[derive(Debug, Clone)]
struct StageProfileState {
    token_positions: Vec<Option<usize>>,
    values_ns: Vec<Vec<u128>>,
}

#[cfg(feature = "metal-stage-profile")]
impl StageProfileState {
    fn new(ring_capacity: usize) -> Self {
        Self {
            token_positions: vec![None; ring_capacity],
            values_ns: vec![vec![0; TokenStage::ALL.len()]; ring_capacity],
        }
    }

    fn record(&mut self, token_position: usize, stage: TokenStage, ns: u128) {
        if self.token_positions.is_empty() {
            return;
        }
        let slot = token_position % self.token_positions.len();
        if self.token_positions[slot] != Some(token_position) {
            self.token_positions[slot] = Some(token_position);
            self.values_ns[slot].fill(0);
        }
        self.values_ns[slot][stage.index()] =
            self.values_ns[slot][stage.index()].saturating_add(ns);
    }

    fn report(&self) -> MetalStageProfileReport {
        MetalStageProfileReport {
            token_positions: self.token_positions.clone(),
            stage_names: TokenStage::ALL.iter().map(|stage| stage.name()).collect(),
            values_ns: self.values_ns.clone(),
        }
    }
}

#[cfg(feature = "metal-stage-profile")]
type StageMarker = Option<(usize, TokenStage)>;

#[cfg(not(feature = "metal-stage-profile"))]
type StageMarker = ();

macro_rules! stage_marker {
    ($position:expr, $stage:ident) => {{
        #[cfg(feature = "metal-stage-profile")]
        {
            Some(($position, TokenStage::$stage))
        }
        #[cfg(not(feature = "metal-stage-profile"))]
        {
            let _ = $position;
            ()
        }
    }};
}

fn format_duration_ns(ns: u128) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    } else if ns >= 1_000_000 {
        format!("{:.1}ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.1}us", ns as f64 / 1_000.0)
    } else {
        format!("{ns}ns")
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1}MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1}KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

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

fn bf16_linear_profile_name(weight_name: &str) -> String {
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        if weight_name.contains(projection) {
            return format!("op.bf16.{projection}");
        }
    }
    if weight_name.contains(".mlp.router.") {
        return "op.bf16.router".to_string();
    }
    "op.bf16.matvec".to_string()
}

fn mxfp4_profile_name(bias_name: &str) -> String {
    if bias_name.contains("gate_up_proj") {
        "op.mxfp4.gate_up".to_string()
    } else if bias_name.contains("down_proj") {
        "op.mxfp4.down".to_string()
    } else {
        "op.mxfp4.matvec".to_string()
    }
}

fn mxfp4_slab_blocks_len(rows: usize) -> Result<usize> {
    EXPERTS
        .checked_mul(rows)
        .and_then(|value| value.checked_mul(MXFP4_GROUPS))
        .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
        .ok_or_else(|| eyre!("MXFP4 expert slab block length overflow"))
}

fn mxfp4_slab_scales_len(rows: usize) -> Result<usize> {
    EXPERTS
        .checked_mul(rows)
        .and_then(|value| value.checked_mul(MXFP4_GROUPS))
        .ok_or_else(|| eyre!("MXFP4 expert slab scale length overflow"))
}

struct ResidentDecodeScratch {
    hidden_a: platform::F32VectorBuffer,
    hidden_b: platform::F32VectorBuffer,
    normed: platform::F32VectorBuffer,
    q: platform::F32VectorBuffer,
    q_rope: platform::F32VectorBuffer,
    k: platform::F32VectorBuffer,
    k_rope: platform::F32VectorBuffer,
    v: platform::F32VectorBuffer,
    attn: platform::F32VectorBuffer,
    projected: platform::F32VectorBuffer,
    residual: platform::F32VectorBuffer,
    router_input: platform::F32VectorBuffer,
    router_logits: platform::F32VectorBuffer,
    router_indices: platform::U32Buffer,
    router_selected_logits: platform::F32VectorBuffer,
    router_weights: platform::F32VectorBuffer,
    expert_acts_packed: platform::F32VectorBuffer,
    final_hidden: platform::F32VectorBuffer,
    lm_logits: platform::F32VectorBuffer,
    lm_top1_block_indices: platform::U32Buffer,
    lm_top1_block_values: platform::F32VectorBuffer,
    lm_top_indices: platform::U32Buffer,
    lm_top_values: platform::F32VectorBuffer,
}

impl ResidentDecodeScratch {
    fn new(platform: &platform::MetalContext, vocab: usize) -> Result<Self> {
        let lm_top1_blocks = vocab.div_ceil(LM_HEAD_TOP1_BLOCK_SIZE);
        Ok(Self {
            hidden_a: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            hidden_b: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            normed: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            q: platform.alloc_f32_vector(ATTN_VALUES)?,
            q_rope: platform.alloc_f32_vector(ATTN_VALUES)?,
            k: platform.alloc_f32_vector(KV_VALUES)?,
            k_rope: platform.alloc_f32_vector(KV_VALUES)?,
            v: platform.alloc_f32_vector(KV_VALUES)?,
            attn: platform.alloc_f32_vector(ATTN_VALUES)?,
            projected: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            residual: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            router_input: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            router_logits: platform.alloc_f32_vector(32)?,
            router_indices: platform.alloc_u32_buffer(4)?,
            router_selected_logits: platform.alloc_f32_vector(4)?,
            router_weights: platform.alloc_f32_vector(4)?,
            expert_acts_packed: platform.alloc_f32_vector(4 * HIDDEN_SIZE)?,
            final_hidden: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            lm_logits: platform.alloc_f32_vector(vocab)?,
            lm_top1_block_indices: platform.alloc_u32_buffer(lm_top1_blocks)?,
            lm_top1_block_values: platform.alloc_f32_vector(lm_top1_blocks)?,
            lm_top_indices: platform.alloc_u32_buffer(8)?,
            lm_top_values: platform.alloc_f32_vector(8)?,
        })
    }
}

struct ResidentGpuKvCache {
    layers: Vec<ResidentGpuLayerKvCache>,
    capacity: usize,
}

struct ResidentGpuLayerKvCache {
    k: platform::F32VectorBuffer,
    v: platform::F32VectorBuffer,
}

impl ResidentGpuKvCache {
    fn new(platform: &platform::MetalContext, layers: usize, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(eyre!("resident KV cache needs non-zero capacity"));
        }
        if capacity > MAX_KV_CACHE_PROBE_TOKENS {
            return Err(eyre!(
                "resident KV cache currently supports at most {MAX_KV_CACHE_PROBE_TOKENS} positions, got {capacity}"
            ));
        }
        let mut layer_caches = Vec::with_capacity(layers);
        for _ in 0..layers {
            layer_caches.push(ResidentGpuLayerKvCache {
                k: platform.alloc_f32_vector(capacity * KV_VALUES)?,
                v: platform.alloc_f32_vector(capacity * KV_VALUES)?,
            });
        }
        Ok(Self {
            layers: layer_caches,
            capacity,
        })
    }

    fn layer(&self, layer: usize) -> Result<&ResidentGpuLayerKvCache> {
        self.layers
            .get(layer)
            .ok_or_else(|| eyre!("resident KV cache has no layer {layer}"))
    }
}

#[derive(Clone)]
struct ResidentLayerExpertSlabs {
    gate_up_blocks: platform::U8Buffer,
    gate_up_scales: platform::U8Buffer,
    gate_up_bias: platform::Bf16MatrixBuffer,
    down_blocks: platform::U8Buffer,
    down_scales: platform::U8Buffer,
    down_bias: platform::Bf16MatrixBuffer,
}

#[derive(Default)]
struct ResidentWeights {
    bf16_matrices: Mutex<HashMap<String, Arc<model_store::Bf16Matrix>>>,
    bf16_vectors: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    bf16_rows: Mutex<HashMap<(String, usize), Arc<Vec<f32>>>>,
    u8_slices: Mutex<HashMap<(String, usize, usize), Arc<Vec<u8>>>>,
}

impl ResidentWeights {
    fn bf16_matrix(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
    ) -> Result<Arc<model_store::Bf16Matrix>> {
        if let Some(value) = self.bf16_matrices.lock().unwrap().get(tensor_name).cloned() {
            return Ok(value);
        }
        let value = Arc::new(model_store::read_bf16_matrix(report, tensor_name)?);
        self.bf16_matrices
            .lock()
            .unwrap()
            .insert(tensor_name.to_string(), value.clone());
        Ok(value)
    }

    fn bf16_vector(&self, report: &SourceModelReport, tensor_name: &str) -> Result<Arc<Vec<f32>>> {
        if let Some(value) = self.bf16_vectors.lock().unwrap().get(tensor_name).cloned() {
            return Ok(value);
        }
        let value = Arc::new(model_store::read_bf16_vector(report, tensor_name)?);
        self.bf16_vectors
            .lock()
            .unwrap()
            .insert(tensor_name.to_string(), value.clone());
        Ok(value)
    }

    fn bf16_matrix_row(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        row: usize,
    ) -> Result<Arc<Vec<f32>>> {
        let key = (tensor_name.to_string(), row);
        if let Some(value) = self.bf16_rows.lock().unwrap().get(&key).cloned() {
            return Ok(value);
        }
        let value = Arc::new(model_store::read_bf16_matrix_row(report, tensor_name, row)?);
        self.bf16_rows.lock().unwrap().insert(key, value.clone());
        Ok(value)
    }

    fn u8_tensor_slice(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        element_offset: usize,
        element_len: usize,
    ) -> Result<Arc<Vec<u8>>> {
        let key = (tensor_name.to_string(), element_offset, element_len);
        if let Some(value) = self.u8_slices.lock().unwrap().get(&key).cloned() {
            return Ok(value);
        }
        let value = Arc::new(model_store::read_u8_tensor_slice(
            report,
            tensor_name,
            element_offset,
            element_len,
        )?);
        self.u8_slices.lock().unwrap().insert(key, value.clone());
        Ok(value)
    }
}

pub struct MetalEngine {
    report: SourceModelReport,
    harmony: HarmonyAdapter,
    ctx: MetalOracleContext,
    layers: usize,
}

impl MetalEngine {
    pub fn load_canonical() -> Result<Self> {
        Self::load_canonical_with_layers(LAYERS)
    }

    pub fn load_canonical_with_layers(layers: usize) -> Result<Self> {
        if layers > LAYERS {
            return Err(eyre!(
                "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
            ));
        }

        let report = model_store::inspect_canonical_safetensors()?;
        let validation = weights::validate_gpt_oss_20b_source(&report);
        if !validation.is_ok() {
            return Err(eyre!(
                "canonical gpt-oss SafeTensors layout did not validate"
            ));
        }

        let harmony = HarmonyAdapter::gpt_oss()?;
        let ctx = MetalOracleContext::with_lm_head(&report)?;
        Ok(Self {
            report,
            harmony,
            ctx,
            layers,
        })
    }

    pub fn generate(&self, request: EngineRequest) -> Result<Vec<GenerationEvent>> {
        self.generate_inner(request)
    }

    pub fn generate_profiled(
        &self,
        request: EngineRequest,
    ) -> Result<(Vec<GenerationEvent>, MetalProfileReport)> {
        self.ctx.enable_profile();
        let started = Instant::now();
        let result = self.generate_inner(request);
        self.ctx.record_profile(
            "phase.generate",
            ProfileDelta {
                wall: started.elapsed(),
                ..ProfileDelta::default()
            },
        );
        let report = self.ctx.profile_report();
        self.ctx.disable_profile();
        result.map(|events| (events, report))
    }

    fn generate_inner(&self, request: EngineRequest) -> Result<Vec<GenerationEvent>> {
        let mut events = Vec::new();
        for notice in &request.prompt.notices {
            events.push(GenerationEvent::Notice(RuntimeNotice {
                message: notice.message.clone(),
            }));
        }

        let prompt_tokens = request_prompt_tokens(&request)?;
        if request.limits.max_new_tokens == 0 {
            events.push(GenerationEvent::Stop(StopReason::MaxGeneratedTokens));
            return Ok(events);
        }

        let generated = resident_sample_decode(
            &self.ctx,
            &self.report,
            &self.harmony,
            &prompt_tokens,
            self.layers,
            request.limits.max_new_tokens,
            request.sampling,
        )?;

        let mut output_bytes = 0usize;
        let mut stop_reason = generated.stop_reason;
        for token in generated.generated {
            output_bytes = output_bytes.saturating_add(token.text.len());
            if output_bytes > request.limits.max_output_bytes {
                stop_reason = StopReason::OutputByteLimit;
                break;
            }
            events.push(GenerationEvent::Token(token.token));
            events.push(GenerationEvent::Text(token.text));
        }
        events.push(GenerationEvent::Stop(stop_reason));
        Ok(events)
    }
}

fn request_prompt_tokens(request: &EngineRequest) -> Result<Vec<u32>> {
    if !request.prompt.tokens.is_empty() {
        return Ok(request.prompt.tokens.clone());
    }
    if let Some(fixture) = &request.fixture {
        if !fixture.prompt_tokens.is_empty() {
            return Ok(fixture.prompt_tokens.clone());
        }
    }
    Err(eyre!("MetalEngine request has no prompt tokens"))
}

fn resident_sample_decode(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prompt_tokens: &[u32],
    layers: usize,
    max_new_tokens: usize,
    sampling: SamplingConfig,
) -> Result<GreedyDecodeProbeReport> {
    ctx.profile_op(
        "phase.resident_sample_decode",
        ProfileDelta::default(),
        || {
            resident_sample_decode_inner(
                ctx,
                report,
                harmony,
                prompt_tokens,
                layers,
                max_new_tokens,
                sampling,
            )
        },
    )
}

fn resident_sample_decode_inner(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prompt_tokens: &[u32],
    layers: usize,
    max_new_tokens: usize,
    sampling: SamplingConfig,
) -> Result<GreedyDecodeProbeReport> {
    if prompt_tokens.is_empty() {
        return Err(eyre!("resident decode needs at least one prompt token"));
    }
    if layers > LAYERS {
        return Err(eyre!(
            "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
        ));
    }
    if max_new_tokens == 0 {
        return Err(eyre!("resident decode needs at least one new token"));
    }
    if sampling.repetition_penalty != 1.0 {
        return Err(eyre!(
            "repetition_penalty is not implemented yet; got {}",
            sampling.repetition_penalty
        ));
    }

    let context_tokens = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| eyre!("resident decode context length overflow"))?;
    if context_tokens > MAX_KV_CACHE_PROBE_TOKENS {
        return Err(eyre!(
            "resident decode currently supports at most {MAX_KV_CACHE_PROBE_TOKENS} context tokens, got {context_tokens}"
        ));
    }
    #[cfg(feature = "metal-stage-profile")]
    ctx.reset_stage_profile(context_tokens);

    let Some(lm_head) = &ctx.lm_head else {
        return Err(eyre!(
            "Metal lm_head weight is not cached; construct MetalOracleContext::with_lm_head"
        ));
    };
    let mut scratch = ResidentDecodeScratch::new(&ctx.platform, lm_head.rows())?;
    let mut kv_cache = ResidentGpuKvCache::new(&ctx.platform, layers, context_tokens)?;
    let embed = ctx.bf16_matrix_buffer(
        report,
        "model.embed_tokens.weight",
        "op.weight.embed_tokens",
    )?;

    let mut current_is_a = true;
    for (position, token) in prompt_tokens.iter().copied().enumerate() {
        current_is_a = resident_decode_token(
            ctx,
            report,
            &mut scratch,
            &mut kv_cache,
            &embed,
            layers,
            position,
            token,
        )?;
    }

    let stop_tokens = harmony.stop_tokens()?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut stop_reason = StopReason::MaxGeneratedTokens;
    let mut sampler = Sampler::new(sampling.clone());

    for step in 0..max_new_tokens {
        let output_position = prompt_tokens.len() + step;
        let current_hidden = resident_current_hidden(&scratch, current_is_a);
        let candidates = resident_lm_head_topk(
            ctx,
            report,
            harmony,
            &mut scratch,
            &current_hidden,
            sampler.candidate_count(),
            output_position,
        )?;
        let sample_candidates = candidates
            .iter()
            .map(|token| SampleCandidate {
                token: token.token,
                logit: token.logit,
                probability: 0.0,
            })
            .collect::<Vec<_>>();
        let sampled = sampler.choose(&sample_candidates)?;
        let token_id = sampled.token;
        let text = candidates
            .iter()
            .find(|token| token.token == token_id)
            .map(|token| token.text.clone())
            .unwrap_or(decode_token_text(harmony, token_id)?);
        generated.push(GreedyTokenReport {
            token: token_id,
            logit: sampled.logit,
            text,
        });

        if stop_tokens.contains(&token_id) {
            stop_reason = StopReason::EndOfGeneration;
            break;
        }
        if step + 1 == max_new_tokens {
            break;
        }

        let position = output_position;
        current_is_a = resident_decode_token(
            ctx,
            report,
            &mut scratch,
            &mut kv_cache,
            &embed,
            layers,
            position,
            token_id,
        )?;
    }

    let token_ids = generated
        .iter()
        .map(|token| token.token)
        .collect::<Vec<_>>();
    let text = decode_tokens_text(harmony, &token_ids)?;
    Ok(GreedyDecodeProbeReport {
        name: format!(
            "resident_sample_decode.layers{layers}.prompt{}.new{}",
            prompt_tokens.len(),
            max_new_tokens
        ),
        backend: "metal-resident".to_string(),
        scorer: metal_sampler_description(&sampling),
        layers,
        prompt_tokens: prompt_tokens.len(),
        max_new_tokens,
        stop_reason,
        generated,
        text,
    })
}

fn resident_hidden_pair(
    scratch: &ResidentDecodeScratch,
    current_is_a: bool,
) -> (platform::F32VectorBuffer, platform::F32VectorBuffer) {
    if current_is_a {
        (scratch.hidden_a.clone(), scratch.hidden_b.clone())
    } else {
        (scratch.hidden_b.clone(), scratch.hidden_a.clone())
    }
}

fn resident_current_hidden(
    scratch: &ResidentDecodeScratch,
    current_is_a: bool,
) -> platform::F32VectorBuffer {
    if current_is_a {
        scratch.hidden_a.clone()
    } else {
        scratch.hidden_b.clone()
    }
}

#[allow(clippy::too_many_arguments)]
fn resident_decode_token(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    scratch: &mut ResidentDecodeScratch,
    kv_cache: &mut ResidentGpuKvCache,
    embed: &platform::Bf16MatrixBuffer,
    layers: usize,
    position: usize,
    token: u32,
) -> Result<bool> {
    let batch = ctx.platform.begin_batch();
    batch.embedding_lookup_bf16_into(embed, token as usize, &scratch.hidden_a)?;
    let mut current_is_a = true;

    for layer in 0..layers {
        let (input, output) = resident_hidden_pair(scratch, current_is_a);
        resident_decode_layer(
            ctx, report, &batch, scratch, kv_cache, layer, position, &input, &output,
        )?;
        current_is_a = !current_is_a;
    }

    finish_resident_batch(ctx, batch, stage_marker!(position, Token));
    Ok(current_is_a)
}

fn resident_lm_head_topk(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    scratch: &mut ResidentDecodeScratch,
    hidden: &platform::F32VectorBuffer,
    k: usize,
    output_position: usize,
) -> Result<Vec<GreedyTokenReport>> {
    let Some(lm_head) = &ctx.lm_head else {
        return Err(eyre!(
            "Metal lm_head weight is not cached; construct MetalOracleContext::with_lm_head"
        ));
    };
    let norm_weight =
        ctx.bf16_vector_buffer(report, "model.norm.weight", "op.weight.final_norm")?;

    let batch = ctx.platform.begin_batch();
    batch.rms_norm_into(hidden, &norm_weight, &scratch.final_hidden)?;
    if k == 1 {
        batch.bf16_matrix_top1_into(
            lm_head,
            &scratch.final_hidden,
            &scratch.lm_logits,
            &scratch.lm_top1_block_indices,
            &scratch.lm_top1_block_values,
            &scratch.lm_top_indices,
            &scratch.lm_top_values,
        )?;
    } else {
        batch.bf16_matrix_topk_into(
            lm_head,
            &scratch.final_hidden,
            &scratch.lm_logits,
            &scratch.lm_top_indices,
            &scratch.lm_top_values,
            k,
        )?;
    }
    finish_resident_batch(ctx, batch, stage_marker!(output_position, LmHead));

    let indices = ctx.platform.read_u32_buffer(&scratch.lm_top_indices);
    let values = ctx.platform.read_f32_vector(&scratch.lm_top_values);
    indices
        .into_iter()
        .zip(values)
        .take(k)
        .map(|(token, logit)| {
            Ok(GreedyTokenReport {
                token,
                logit,
                text: decode_token_text(harmony, token)?,
            })
        })
        .collect()
}

fn resident_decode_layer(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    batch: &platform::MetalBatch<'_>,
    scratch: &mut ResidentDecodeScratch,
    kv_cache: &mut ResidentGpuKvCache,
    layer: usize,
    position: usize,
    input: &platform::F32VectorBuffer,
    output: &platform::F32VectorBuffer,
) -> Result<()> {
    if position >= kv_cache.capacity {
        return Err(eyre!(
            "resident decode position {position} exceeds KV capacity {}",
            kv_cache.capacity
        ));
    }

    let prefix = format!("model.layers.{layer}");
    let input_norm_weight = ctx.bf16_vector_buffer(
        report,
        &format!("{prefix}.input_layernorm.weight"),
        "op.weight.input_layernorm",
    )?;
    let q_weight = ctx.bf16_matrix_buffer(
        report,
        &format!("{prefix}.self_attn.q_proj.weight"),
        "op.bf16.q_proj",
    )?;
    let q_bias = ctx.bf16_vector_buffer(
        report,
        &format!("{prefix}.self_attn.q_proj.bias"),
        "op.bf16.q_proj",
    )?;
    let k_weight = ctx.bf16_matrix_buffer(
        report,
        &format!("{prefix}.self_attn.k_proj.weight"),
        "op.bf16.k_proj",
    )?;
    let k_bias = ctx.bf16_vector_buffer(
        report,
        &format!("{prefix}.self_attn.k_proj.bias"),
        "op.bf16.k_proj",
    )?;
    let v_weight = ctx.bf16_matrix_buffer(
        report,
        &format!("{prefix}.self_attn.v_proj.weight"),
        "op.bf16.v_proj",
    )?;
    let v_bias = ctx.bf16_vector_buffer(
        report,
        &format!("{prefix}.self_attn.v_proj.bias"),
        "op.bf16.v_proj",
    )?;
    let sinks = ctx.bf16_vector_buffer(
        report,
        &format!("{prefix}.self_attn.sinks"),
        "op.weight.attention_sinks",
    )?;
    let o_weight = ctx.bf16_matrix_buffer(
        report,
        &format!("{prefix}.self_attn.o_proj.weight"),
        "op.bf16.o_proj",
    )?;
    let o_bias = ctx.bf16_vector_buffer(
        report,
        &format!("{prefix}.self_attn.o_proj.bias"),
        "op.bf16.o_proj",
    )?;
    let post_norm_weight = ctx.bf16_vector_buffer(
        report,
        &format!("{prefix}.post_attention_layernorm.weight"),
        "op.weight.post_attention_layernorm",
    )?;
    let router_weight = ctx.bf16_matrix_buffer(
        report,
        &format!("{prefix}.mlp.router.weight"),
        "op.bf16.router",
    )?;
    let router_bias = ctx.bf16_vector_buffer(
        report,
        &format!("{prefix}.mlp.router.bias"),
        "op.bf16.router",
    )?;

    let layer_cache = kv_cache.layer(layer)?;
    batch.rms_norm_into(input, &input_norm_weight, &scratch.normed)?;
    batch.bf16_matrix_matvec_into(&q_weight, &scratch.normed, &q_bias, &scratch.q)?;
    batch.bf16_matrix_matvec_into(&k_weight, &scratch.normed, &k_bias, &scratch.k)?;
    batch.bf16_matrix_matvec_into(&v_weight, &scratch.normed, &v_bias, &scratch.v)?;
    batch.rope_row_into(&scratch.q, &scratch.q_rope, Q_HEADS, position)?;
    batch.rope_row_into(&scratch.k, &scratch.k_rope, KV_HEADS, position)?;
    batch.write_f32_slot_into(&scratch.k_rope, &layer_cache.k, position, KV_VALUES)?;
    batch.write_f32_slot_into(&scratch.v, &layer_cache.v, position, KV_VALUES)?;
    batch.kv_cache_decode_attention_into(
        layer,
        position,
        0,
        position + 1,
        &scratch.q_rope,
        &layer_cache.k,
        &layer_cache.v,
        &sinks,
        &scratch.attn,
    )?;
    batch.bf16_matrix_matvec_into(&o_weight, &scratch.attn, &o_bias, &scratch.projected)?;
    batch.vector_add_into(input, &scratch.projected, &scratch.residual)?;
    batch.rms_norm_into(&scratch.residual, &post_norm_weight, &scratch.router_input)?;
    batch.bf16_matrix_matvec_into(
        &router_weight,
        &scratch.router_input,
        &router_bias,
        &scratch.router_logits,
    )?;
    batch.top4_softmax_into(
        &scratch.router_logits,
        &scratch.router_indices,
        &scratch.router_selected_logits,
        &scratch.router_weights,
    )?;

    let expert_prefix = format!("{prefix}.mlp.experts");
    let expert_slabs = ctx.mxfp4_layer_expert_slabs(report, &expert_prefix)?;

    batch.mxfp4_top4_gate_swiglu_into(
        &expert_slabs.gate_up_blocks,
        &expert_slabs.gate_up_scales,
        &expert_slabs.gate_up_bias,
        &scratch.router_input,
        &scratch.router_indices,
        &scratch.expert_acts_packed,
    )?;
    batch.mxfp4_top4_down_weighted_into(
        &expert_slabs.down_blocks,
        &expert_slabs.down_scales,
        &expert_slabs.down_bias,
        &scratch.expert_acts_packed,
        &scratch.router_indices,
        &scratch.router_weights,
        &scratch.residual,
        output,
    )?;

    Ok(())
}

fn finish_resident_batch(
    ctx: &MetalOracleContext,
    batch: platform::MetalBatch<'_>,
    stage: StageMarker,
) {
    let gpu_ns = batch.finish();
    #[cfg(not(feature = "metal-stage-profile"))]
    {
        let _ = stage;
        let _ = gpu_ns;
    }
    ctx.record_profile(
        "phase.resident_sample_decode",
        ProfileDelta {
            command_buffers: 1,
            ..ProfileDelta::default()
        },
    );
    #[cfg(feature = "metal-stage-profile")]
    if let Some((token_position, stage)) = stage {
        ctx.record_token_stage(token_position, stage, gpu_ns);
    }
}

pub fn probe_rms_norm_embedding(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalRmsNormProbeReport> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let weight = model_store::read_bf16_vector(report, "model.layers.0.input_layernorm.weight")?;
    let cpu = backend_cpu::rms_norm_reference(&embedding, &weight)?;
    let metal = ctx.platform.rms_norm(&embedding, &weight)?;
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal RMSNorm returned {} values, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut max_abs_delta = 0.0f32;
    let mut sum_abs_delta = 0.0f64;
    for (cpu, metal) in cpu.iter().copied().zip(metal.iter().copied()) {
        let delta = (cpu - metal).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_abs_delta += delta as f64;
    }
    let mean_abs_delta = if cpu.is_empty() {
        0.0
    } else {
        (sum_abs_delta / cpu.len() as f64) as f32
    };

    Ok(MetalRmsNormProbeReport {
        token,
        values: cpu.len(),
        max_abs_delta,
        mean_abs_delta,
        cpu_first8: cpu.iter().copied().take(8).collect(),
        metal_first8: metal.into_iter().take(8).collect(),
    })
}

pub fn probe_layer0_q_proj(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    probe_layer0_projection(ctx, report, token, "q_proj")
}

pub fn probe_layer0_k_proj(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    probe_layer0_projection(ctx, report, token, "k_proj")
}

pub fn probe_layer0_v_proj(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    probe_layer0_projection(ctx, report, token, "v_proj")
}

fn probe_layer0_projection(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
    projection: &str,
) -> Result<MetalMatvecProbeReport> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.input_layernorm.weight")?;
    let input = backend_cpu::rms_norm_reference(&embedding, &norm_weight)?;

    let weight_name = format!("model.layers.0.self_attn.{projection}.weight");
    let bias_name = format!("model.layers.0.self_attn.{projection}.bias");
    let report_name = format!("layer0.self_attn.{projection}");

    let weight = model_store::read_bf16_matrix(report, &weight_name)?;
    let bias = model_store::read_bf16_vector(report, &bias_name)?;

    let mut cpu = model_store::matvec_bf16(report, &weight_name, &input)?;
    model_store::add_in_place(&mut cpu, &bias, &report_name)?;
    let metal =
        ctx.platform
            .bf16_matvec(&weight.values, weight.rows, weight.cols, &input, &bias)?;

    matvec_report(&report_name, token, weight.rows, weight.cols, &cpu, metal)
}

pub fn probe_layer0_q_rope(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
    position: usize,
) -> Result<MetalVectorProbeReport> {
    probe_layer0_rope(ctx, report, token, position, "q_proj", Q_HEADS)
}

pub fn probe_layer0_k_rope(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
    position: usize,
) -> Result<MetalVectorProbeReport> {
    probe_layer0_rope(ctx, report, token, position, "k_proj", KV_HEADS)
}

fn probe_layer0_rope(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
    position: usize,
    projection: &str,
    heads: usize,
) -> Result<MetalVectorProbeReport> {
    let input = layer0_projection_cpu(report, token, projection)?;
    let cpu = backend_cpu::apply_rope_reference(&input, heads, position)?;
    let metal = ctx.platform.rope_row(&input, heads, position)?;
    vector_report(
        &format!("layer0.self_attn.{projection}.rope"),
        token,
        position,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_single_token_attention(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let q = layer0_projection_cpu(report, token, "q_proj")?;
    let k = layer0_projection_cpu(report, token, "k_proj")?;
    let v = layer0_projection_cpu(report, token, "v_proj")?;
    let q = backend_cpu::apply_rope_reference(&q, Q_HEADS, 0)?;
    let k = backend_cpu::apply_rope_reference(&k, KV_HEADS, 0)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    let cpu = backend_cpu::single_token_attention_from_rope_reference(&q, &k, &v, &sinks)?;
    let metal = ctx.platform.single_token_attention(&q, &k, &v, &sinks)?;
    vector_report(
        "layer0.self_attn.single_token_attention",
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_sequence_attention(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.first().copied() else {
        return Err(eyre!("sequence attention probe needs at least one token"));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "sequence attention probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let (q, k, v) = layer0_rope_qkv_cpu(report, tokens)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    let cpu = backend_cpu::sequence_attention_from_rope_reference(0, &q, &k, &v, &sinks)?;
    let cpu = flatten_rows(&cpu);
    let q = flatten_rows(&q);
    let k = flatten_rows(&k);
    let v = flatten_rows(&v);
    let metal = ctx
        .platform
        .sequence_attention(0, &q, &k, &v, &sinks, tokens.len())?;
    vector_report(
        &format!(
            "layer0.self_attn.sequence_attention.prefill{}",
            tokens.len()
        ),
        token,
        tokens.len() - 1,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_kv_cache_decode_attention(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!("KV-cache decode probe needs at least one token"));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "KV-cache decode probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let (q, k, v) = layer0_rope_qkv_cpu(report, tokens)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    let cpu = backend_cpu::sequence_attention_from_rope_reference(0, &q, &k, &v, &sinks)?;
    let Some(cpu) = cpu.last() else {
        return Err(eyre!("sequence attention returned no rows"));
    };
    let query_position = tokens.len() - 1;
    let mut kv_cache = PlannedKvCache::new(KvCachePlan::gpt_oss_20b(tokens.len())?);
    let inputs = layer0_embedding_rows(report, tokens)?;
    let q = prefill_layer_attention_cache_metal(ctx, report, 0, &inputs, 0, &mut kv_cache)?;
    let view = kv_cache
        .layer(0)?
        .contiguous_view_for_query(query_position)?;
    let metal = ctx.platform.kv_cache_decode_attention(
        0,
        query_position,
        view.start_position,
        &q[query_position],
        &view.k,
        &view.v,
        &sinks,
    )?;
    vector_report(
        &format!(
            "layer0.self_attn.kv_cache_prefill_decode.prefill{}",
            tokens.len()
        ),
        token,
        query_position,
        cpu,
        metal,
    )
}

pub fn probe_kv_cache_window_rollover_attention(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<MetalVectorProbeReport> {
    kv_cache_attention_probe(
        ctx,
        report,
        tokens,
        0,
        &format!("kv_cache.window_rollover.layer0.prefill{}", tokens.len()),
    )
}

pub fn probe_kv_cache_dense_accumulation_attention(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<MetalVectorProbeReport> {
    kv_cache_attention_probe(
        ctx,
        report,
        tokens,
        1,
        &format!("kv_cache.dense_accumulation.layer1.prefill{}", tokens.len()),
    )
}

pub fn probe_prefill_layers_output(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!("prefill layer-output probe needs at least one token"));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "prefill layer-output probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }
    if layers > LAYERS {
        return Err(eyre!(
            "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
        ));
    }

    let cpu = backend_cpu::sequence_layers_reference(report, tokens, layers)?;
    let cpu = flatten_rows(&cpu);
    let metal = prefill_layers_metal(ctx, report, tokens, layers)?;
    let metal = flatten_rows(&metal);
    vector_report(
        &format!("prefill.layers{layers}.output.tokens{}", tokens.len()),
        token,
        tokens.len() - 1,
        &cpu,
        metal,
    )
}

pub fn probe_prefill_final_norm(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!("prefill final-norm probe needs at least one token"));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "prefill final-norm probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let cpu = prefill_final_norm_cpu(report, tokens, layers)?;
    let metal = prefill_final_norm_metal(ctx, report, tokens, layers)?;
    vector_report(
        &format!("prefill.layers{layers}.final_norm.tokens{}", tokens.len()),
        token,
        tokens.len() - 1,
        &cpu,
        metal,
    )
}

pub fn probe_prefill_selected_logits(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
    logit_tokens: &[u32],
) -> Result<MetalSelectedLogitsProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!(
            "prefill selected-logits probe needs at least one token"
        ));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "prefill selected-logits probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let cpu_final_hidden = prefill_final_norm_cpu(report, tokens, layers)?;
    let cpu = backend_cpu::selected_logits_reference(report, &cpu_final_hidden, logit_tokens)?;
    let metal_final_hidden = prefill_final_norm_metal(ctx, report, tokens, layers)?;
    let metal = selected_logits_metal(ctx, report, &metal_final_hidden, logit_tokens)?;
    selected_logits_report(
        &format!(
            "prefill.layers{layers}.selected_logits.tokens{}",
            tokens.len()
        ),
        token,
        layers,
        cpu,
        metal,
    )
}

pub fn probe_decode_one_final_norm(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<MetalVectorProbeReport> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let cpu = decode_one_final_norm_cpu(report, prefill_tokens, decode_token, layers)?;
    let metal = decode_one_final_norm_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    vector_report(
        &format!("decode_one.layers{layers}.final_norm.tokens{total_tokens}"),
        decode_token,
        prefill_tokens.len(),
        &cpu,
        metal,
    )
}

pub fn probe_decode_one_selected_logits(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
    logit_tokens: &[u32],
) -> Result<MetalSelectedLogitsProbeReport> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let cpu_final_hidden = decode_one_final_norm_cpu(report, prefill_tokens, decode_token, layers)?;
    let cpu = backend_cpu::selected_logits_reference(report, &cpu_final_hidden, logit_tokens)?;
    let metal_final_hidden =
        decode_one_final_norm_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    let metal = selected_logits_metal(ctx, report, &metal_final_hidden, logit_tokens)?;
    selected_logits_report(
        &format!("decode_one.layers{layers}.selected_logits.tokens{total_tokens}"),
        decode_token,
        layers,
        cpu,
        metal,
    )
}

pub fn probe_decode_one_greedy_text(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<GreedyTextProbeReport> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let cpu_hidden = decode_one_final_norm_cpu(report, prefill_tokens, decode_token, layers)?;
    let metal_hidden =
        decode_one_final_norm_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    let cpu = greedy_top1_text(report, harmony, &cpu_hidden)?;
    let metal = greedy_top1_text(report, harmony, &metal_hidden)?;
    Ok(GreedyTextProbeReport {
        name: format!("decode_one.layers{layers}.greedy_text.tokens{total_tokens}"),
        position: prefill_tokens.len(),
        layers,
        scorer: "streaming CPU BF16 lm_head top-1 for each hidden state".to_string(),
        token_match: cpu.token == metal.token,
        logit_delta: (cpu.logit - metal.logit).abs(),
        cpu,
        metal,
    })
}

pub fn probe_decode_one_lm_head_topk(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
    k: usize,
) -> Result<LmHeadTopKProbeReport> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let cpu_hidden = decode_one_final_norm_cpu(report, prefill_tokens, decode_token, layers)?;
    let metal_hidden =
        decode_one_final_norm_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    let cpu = lm_head_topk_cpu(report, harmony, &cpu_hidden, k)?;
    let metal = lm_head_topk_metal(ctx, harmony, &metal_hidden, k)?;
    lm_head_topk_report(
        &format!("decode_one.layers{layers}.lm_head_top{k}.tokens{total_tokens}"),
        prefill_tokens.len(),
        layers,
        k,
        cpu,
        metal,
    )
}

pub fn probe_greedy_decode(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prompt_tokens: &[u32],
    layers: usize,
    max_new_tokens: usize,
) -> Result<GreedyDecodeProbeReport> {
    probe_sample_decode(
        ctx,
        report,
        harmony,
        prompt_tokens,
        layers,
        max_new_tokens,
        SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        },
    )
}

pub fn probe_sample_decode(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prompt_tokens: &[u32],
    layers: usize,
    max_new_tokens: usize,
    sampling: SamplingConfig,
) -> Result<GreedyDecodeProbeReport> {
    if prompt_tokens.is_empty() {
        return Err(eyre!("sample decode needs at least one prompt token"));
    }
    if layers > LAYERS {
        return Err(eyre!(
            "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
        ));
    }
    if max_new_tokens == 0 {
        return Err(eyre!("sample decode needs at least one new token"));
    }
    if prompt_tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "sample decode probe supports at most {MAX_PREFILL_PROBE_TOKENS} prompt tokens, got {}",
            prompt_tokens.len()
        ));
    }
    if sampling.repetition_penalty != 1.0 {
        return Err(eyre!(
            "repetition_penalty is not implemented yet; got {}",
            sampling.repetition_penalty
        ));
    }

    let (hidden, mut kv_cache) = prefill_layers_with_cache_metal(
        ctx,
        report,
        prompt_tokens,
        layers,
        prompt_tokens
            .len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| eyre!("greedy decode context length overflow"))?,
    )?;
    let Some(hidden) = hidden.last() else {
        return Err(eyre!("greedy decode prefill returned no hidden states"));
    };

    let norm_weight = ctx.bf16_vector(report, "model.norm.weight")?;
    let stop_tokens = harmony.stop_tokens()?;
    let mut final_hidden = ctx.rms_norm_profiled("op.rms_norm.final", hidden, &norm_weight)?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut stop_reason = StopReason::MaxGeneratedTokens;
    let mut sampler = Sampler::new(sampling.clone());

    for step in 0..max_new_tokens {
        let candidates =
            lm_head_topk_metal(ctx, harmony, &final_hidden, sampler.candidate_count())?;
        let sample_candidates = candidates
            .iter()
            .map(|token| SampleCandidate {
                token: token.token,
                logit: token.logit,
                probability: 0.0,
            })
            .collect::<Vec<_>>();
        let sampled = sampler.choose(&sample_candidates)?;
        let token_id = sampled.token;
        let text = candidates
            .iter()
            .find(|token| token.token == token_id)
            .map(|token| token.text.clone())
            .unwrap_or(decode_token_text(harmony, token_id)?);
        let token = GreedyTokenReport {
            token: token_id,
            logit: sampled.logit,
            text,
        };
        generated.push(token);

        if stop_tokens.contains(&token_id) {
            stop_reason = StopReason::EndOfGeneration;
            break;
        }
        if step + 1 == max_new_tokens {
            break;
        }

        let mut x = ctx
            .bf16_matrix_row(report, "model.embed_tokens.weight", token_id as usize)?
            .as_ref()
            .clone();
        let position = prompt_tokens.len() + step;
        for layer in 0..layers {
            x = decode_layer_metal(ctx, report, layer, &x, position, &mut kv_cache)?;
        }
        final_hidden = ctx.rms_norm_profiled("op.rms_norm.final", &x, &norm_weight)?;
    }

    let token_ids = generated
        .iter()
        .map(|token| token.token)
        .collect::<Vec<_>>();
    let text = decode_tokens_text(harmony, &token_ids)?;
    Ok(GreedyDecodeProbeReport {
        name: format!(
            "sample_decode.layers{layers}.prompt{}.new{}",
            prompt_tokens.len(),
            max_new_tokens
        ),
        backend: "metal".to_string(),
        scorer: metal_sampler_description(&sampling),
        layers,
        prompt_tokens: prompt_tokens.len(),
        max_new_tokens,
        stop_reason,
        generated,
        text,
    })
}

pub fn probe_layer0_o_proj(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    let attention = layer0_single_token_attention_cpu(report, token)?;
    let weight_name = "model.layers.0.self_attn.o_proj.weight";
    let bias_name = "model.layers.0.self_attn.o_proj.bias";
    let report_name = "layer0.self_attn.o_proj";

    let weight = model_store::read_bf16_matrix(report, weight_name)?;
    let bias = model_store::read_bf16_vector(report, bias_name)?;

    let mut cpu = model_store::matvec_bf16(report, weight_name, &attention)?;
    model_store::add_in_place(&mut cpu, &bias, report_name)?;
    let metal =
        ctx.platform
            .bf16_matvec(&weight.values, weight.rows, weight.cols, &attention, &bias)?;

    matvec_report(report_name, token, weight.rows, weight.cols, &cpu, metal)
}

pub fn probe_layer0_attention_residual(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let o_proj = layer0_o_proj_cpu(report, token)?;
    let cpu = add_vectors(&embedding, &o_proj)?;
    let metal = ctx.platform.vector_add(&embedding, &o_proj)?;
    vector_report("layer0.attention_residual", token, 0, &cpu, metal)
}

pub fn probe_layer0_post_attention_rms_norm(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let residual = layer0_attention_residual_cpu(report, token)?;
    let weight =
        model_store::read_bf16_vector(report, "model.layers.0.post_attention_layernorm.weight")?;
    let cpu = backend_cpu::rms_norm_reference(&residual, &weight)?;
    let metal = ctx.platform.rms_norm(&residual, &weight)?;
    vector_report("layer0.post_attention_layernorm", token, 0, &cpu, metal)
}

pub fn probe_layer0_router(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    let residual = layer0_attention_residual_cpu(report, token)?;
    let norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.post_attention_layernorm.weight")?;
    let input = backend_cpu::rms_norm_reference(&residual, &norm_weight)?;
    let weight_name = "model.layers.0.mlp.router.weight";
    let bias_name = "model.layers.0.mlp.router.bias";
    let report_name = "layer0.mlp.router";

    let weight = model_store::read_bf16_matrix(report, weight_name)?;
    let bias = model_store::read_bf16_vector(report, bias_name)?;
    let mut cpu = model_store::matvec_bf16(report, weight_name, &input)?;
    model_store::add_in_place(&mut cpu, &bias, report_name)?;
    let metal =
        ctx.platform
            .bf16_matvec(&weight.values, weight.rows, weight.cols, &input, &bias)?;

    matvec_report(report_name, token, weight.rows, weight.cols, &cpu, metal)
}

pub fn probe_layer0_router_top4(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalTopKProbeReport> {
    let router = layer0_router_cpu(report, token)?;
    let cpu = backend_cpu::top_k_softmax_reference(&router, 4);
    let metal = ctx.platform.top4_softmax(&router)?;
    top_k_report("layer0.mlp.router.top4", token, cpu, metal)
}

pub fn probe_layer0_top_expert_gate_up(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let (expert, input, bias, cpu) = layer0_top_expert_gate_up_parts(report, token)?;
    let prefix = "model.layers.0.mlp.experts";
    let blocks_name = format!("{prefix}.gate_up_proj_blocks");
    let scales_name = format!("{prefix}.gate_up_proj_scales");

    let blocks = read_mxfp4_expert_blocks(report, &blocks_name, expert.index, GATE_UP_VALUES)?;
    let scales = read_mxfp4_expert_scales(report, &scales_name, expert.index, GATE_UP_VALUES)?;
    let metal = ctx
        .platform
        .mxfp4_matvec(&blocks, &scales, GATE_UP_VALUES, &input, &bias)?;
    vector_report(
        &format!("layer0.mlp.expert{}.gate_up_proj", expert.index),
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_top_expert_swiglu(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let (expert, _, _, gate_up) = layer0_top_expert_gate_up_parts(report, token)?;
    let cpu = backend_cpu::swiglu_reference(&gate_up)?;
    let metal = ctx.platform.swiglu(&gate_up)?;
    vector_report(
        &format!("layer0.mlp.expert{}.swiglu", expert.index),
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_top_expert_down_proj(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let (expert, _, _, gate_up) = layer0_top_expert_gate_up_parts(report, token)?;
    let input = backend_cpu::swiglu_reference(&gate_up)?;
    let prefix = "model.layers.0.mlp.experts";
    let blocks_name = format!("{prefix}.down_proj_blocks");
    let scales_name = format!("{prefix}.down_proj_scales");
    let bias_name = format!("{prefix}.down_proj_bias");

    let mut cpu = backend_cpu::mxfp4_expert_matvec_reference(
        report,
        &blocks_name,
        &scales_name,
        expert.index,
        HIDDEN_SIZE,
        &input,
    )?;
    let bias = model_store::read_bf16_matrix_row(report, &bias_name, expert.index)?;
    model_store::add_in_place(&mut cpu, &bias, &format!("{prefix}.down_proj"))?;

    let blocks = read_mxfp4_expert_blocks(report, &blocks_name, expert.index, HIDDEN_SIZE)?;
    let scales = read_mxfp4_expert_scales(report, &scales_name, expert.index, HIDDEN_SIZE)?;
    let metal = ctx
        .platform
        .mxfp4_matvec(&blocks, &scales, HIDDEN_SIZE, &input, &bias)?;
    vector_report(
        &format!("layer0.mlp.expert{}.down_proj", expert.index),
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_moe_top4(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let input = layer0_router_input_cpu(report, token)?;
    let router = layer0_router_cpu(report, token)?;
    let top_experts = backend_cpu::top_k_softmax_reference(&router, 4);
    if top_experts.len() != 4 {
        return Err(eyre!(
            "router returned {} experts, expected 4",
            top_experts.len()
        ));
    }

    let cpu = backend_cpu::layer_moe_reference(report, 0, &input, &top_experts)?;
    let metal = layer_moe_top4_metal(ctx, report, 0, &input, &router)?;
    vector_report("layer0.mlp.top4", token, 0, &cpu, metal)
}

pub fn probe_layer0_output(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let cpu = backend_cpu::single_token_layer_reference(report, 0, &embedding)?;
    let metal = single_token_layer_metal(ctx, report, 0, &embedding, 0)?;

    vector_report("layer0.output", token, 0, &cpu, metal)
}

pub fn probe_single_token_final_norm(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
    layers: usize,
) -> Result<MetalVectorProbeReport> {
    let cpu = backend_cpu::single_token_final_norm_reference(report, token, layers)?;
    let metal = single_token_final_norm_metal(ctx, report, token, layers)?;
    vector_report(
        &format!("single_token.layers{layers}.final_norm"),
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_single_token_selected_logits(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
    layers: usize,
    logit_tokens: &[u32],
) -> Result<MetalSelectedLogitsProbeReport> {
    let cpu_final_hidden = backend_cpu::single_token_final_norm_reference(report, token, layers)?;
    let cpu = backend_cpu::selected_logits_reference(report, &cpu_final_hidden, logit_tokens)?;

    let metal_final_hidden = single_token_final_norm_metal(ctx, report, token, layers)?;
    let metal = selected_logits_metal(ctx, report, &metal_final_hidden, logit_tokens)?;
    selected_logits_report(
        &format!("single_token.layers{layers}.selected_logits"),
        token,
        layers,
        cpu,
        metal,
    )
}

fn single_token_final_norm_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    token: u32,
    layers: usize,
) -> Result<Vec<f32>> {
    if layers > LAYERS {
        return Err(eyre!(
            "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
        ));
    }

    let mut x =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    for layer in 0..layers {
        x = single_token_layer_metal(ctx, report, layer, &x, 0)?;
    }

    let norm_weight = ctx.bf16_vector(report, "model.norm.weight")?;
    ctx.rms_norm_profiled("op.rms_norm.final", &x, &norm_weight)
}

fn single_token_layer_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    position: usize,
) -> Result<Vec<f32>> {
    let prefix = format!("model.layers.{layer}");
    let (q, k, v) = layer_attention_qkv_metal(ctx, report, layer, input, position)?;
    let sinks = ctx.bf16_vector(report, &format!("{prefix}.self_attn.sinks"))?;
    let attn = ctx.single_token_attention_profiled(&q, &k, &v, &sinks)?;
    let projected = layer_projection_metal(ctx, report, layer, "o_proj", &attn)?;
    let residual = ctx.vector_add_profiled("op.residual.attention", input, &projected)?;

    let post_norm_weight =
        ctx.bf16_vector(report, &format!("{prefix}.post_attention_layernorm.weight"))?;
    let router_input =
        ctx.rms_norm_profiled("op.rms_norm.post_attention", &residual, &post_norm_weight)?;
    let router = layer_router_metal(ctx, report, layer, &router_input)?;
    let moe = layer_moe_top4_metal(ctx, report, layer, &router_input, &router)?;
    ctx.vector_add_profiled("op.residual.moe", &residual, &moe)
}

fn prefill_layers_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<Vec<Vec<f32>>> {
    let (x, _) = prefill_layers_with_cache_metal(ctx, report, tokens, layers, tokens.len())?;
    Ok(x)
}

fn prefill_layers_with_cache_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
    context_tokens: usize,
) -> Result<(Vec<Vec<f32>>, PlannedKvCache)> {
    if layers > LAYERS {
        return Err(eyre!(
            "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
        ));
    }

    let mut x = layer0_embedding_rows_metal(ctx, report, tokens)?;
    let mut kv_cache = PlannedKvCache::new(KvCachePlan::gpt_oss_20b(context_tokens)?);
    for layer in 0..layers {
        x = prefill_layer_metal(ctx, report, layer, &x, 0, &mut kv_cache)?;
    }
    Ok((x, kv_cache))
}

fn prefill_final_norm_cpu(
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<Vec<f32>> {
    let hidden = backend_cpu::sequence_layers_reference(report, tokens, layers)?;
    let Some(hidden) = hidden.last() else {
        return Err(eyre!("prefill final norm needs at least one token"));
    };
    let norm_weight = model_store::read_bf16_vector(report, "model.norm.weight")?;
    backend_cpu::rms_norm_reference(hidden, &norm_weight)
}

fn prefill_final_norm_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<Vec<f32>> {
    let hidden = prefill_layers_metal(ctx, report, tokens, layers)?;
    let Some(hidden) = hidden.last() else {
        return Err(eyre!("prefill final norm needs at least one token"));
    };
    let norm_weight = ctx.bf16_vector(report, "model.norm.weight")?;
    ctx.rms_norm_profiled("op.rms_norm.final", hidden, &norm_weight)
}

fn decode_one_final_norm_cpu(
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<Vec<f32>> {
    let tokens = decode_sequence_tokens(prefill_tokens, decode_token)?;
    prefill_final_norm_cpu(report, &tokens, layers)
}

fn decode_one_final_norm_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<Vec<f32>> {
    let hidden = decode_one_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    let norm_weight = ctx.bf16_vector(report, "model.norm.weight")?;
    ctx.rms_norm_profiled("op.rms_norm.final", &hidden, &norm_weight)
}

fn decode_one_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<Vec<f32>> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let (_, mut kv_cache) =
        prefill_layers_with_cache_metal(ctx, report, prefill_tokens, layers, total_tokens)?;
    let mut x = ctx
        .bf16_matrix_row(report, "model.embed_tokens.weight", decode_token as usize)?
        .as_ref()
        .clone();
    let position = prefill_tokens.len();
    for layer in 0..layers {
        x = decode_layer_metal(ctx, report, layer, &x, position, &mut kv_cache)?;
    }
    Ok(x)
}

fn decode_layer_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    position: usize,
    kv_cache: &mut PlannedKvCache,
) -> Result<Vec<f32>> {
    let prefix = format!("model.layers.{layer}");
    let (q, k, v) = layer_attention_qkv_metal(ctx, report, layer, input, position)?;
    kv_cache.layer_mut(layer)?.push(position, &k, &v)?;
    let view = kv_cache.layer(layer)?.contiguous_view_for_query(position)?;
    let sinks = ctx.bf16_vector(report, &format!("{prefix}.self_attn.sinks"))?;
    let attn = ctx.kv_cache_decode_attention_profiled(
        layer,
        position,
        view.start_position,
        &q,
        &view.k,
        &view.v,
        &sinks,
    )?;
    let projected = layer_projection_metal(ctx, report, layer, "o_proj", &attn)?;
    let residual = ctx.vector_add_profiled("op.residual.attention", input, &projected)?;

    let post_norm_weight =
        ctx.bf16_vector(report, &format!("{prefix}.post_attention_layernorm.weight"))?;
    let router_input =
        ctx.rms_norm_profiled("op.rms_norm.post_attention", &residual, &post_norm_weight)?;
    let router = layer_router_metal(ctx, report, layer, &router_input)?;
    let moe = layer_moe_top4_metal(ctx, report, layer, &router_input, &router)?;
    ctx.vector_add_profiled("op.residual.moe", &residual, &moe)
}

fn prefill_layer_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    layer: usize,
    inputs: &[Vec<f32>],
    start_position: usize,
    kv_cache: &mut PlannedKvCache,
) -> Result<Vec<Vec<f32>>> {
    let prefix = format!("model.layers.{layer}");
    let sinks = ctx.bf16_vector(report, &format!("{prefix}.self_attn.sinks"))?;
    let q_rows =
        prefill_layer_attention_cache_metal(ctx, report, layer, inputs, start_position, kv_cache)?;

    let mut residuals = Vec::with_capacity(inputs.len());
    for (offset, (input, q)) in inputs.iter().zip(q_rows.iter()).enumerate() {
        let position = start_position + offset;
        let view = kv_cache.layer(layer)?.contiguous_view_for_query(position)?;
        let attn = ctx.kv_cache_decode_attention_profiled(
            layer,
            position,
            view.start_position,
            q,
            &view.k,
            &view.v,
            &sinks,
        )?;
        let projected = layer_projection_metal(ctx, report, layer, "o_proj", &attn)?;
        residuals.push(ctx.vector_add_profiled("op.residual.attention", input, &projected)?);
    }

    let post_norm_weight =
        ctx.bf16_vector(report, &format!("{prefix}.post_attention_layernorm.weight"))?;
    let mut out = Vec::with_capacity(residuals.len());
    for residual in residuals {
        let router_input =
            ctx.rms_norm_profiled("op.rms_norm.post_attention", &residual, &post_norm_weight)?;
        let router = layer_router_metal(ctx, report, layer, &router_input)?;
        let moe = layer_moe_top4_metal(ctx, report, layer, &router_input, &router)?;
        out.push(ctx.vector_add_profiled("op.residual.moe", &residual, &moe)?);
    }

    Ok(out)
}

fn prefill_layer_attention_cache_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    layer: usize,
    inputs: &[Vec<f32>],
    start_position: usize,
    kv_cache: &mut PlannedKvCache,
) -> Result<Vec<Vec<f32>>> {
    let mut q_rows = Vec::with_capacity(inputs.len());
    for (offset, input) in inputs.iter().enumerate() {
        let position = start_position + offset;
        let (q, k, v) = layer_attention_qkv_metal(ctx, report, layer, input, position)?;
        kv_cache.layer_mut(layer)?.push(position, &k, &v)?;
        q_rows.push(q);
    }
    Ok(q_rows)
}

fn layer_attention_qkv_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    position: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if input.len() != HIDDEN_SIZE {
        return Err(eyre!(
            "layer {layer} attention input has {} values, expected {HIDDEN_SIZE}",
            input.len()
        ));
    }

    let prefix = format!("model.layers.{layer}");
    let input_norm_weight = ctx.bf16_vector(report, &format!("{prefix}.input_layernorm.weight"))?;
    let attn_input = ctx.rms_norm_profiled("op.rms_norm.input", input, &input_norm_weight)?;

    let q = layer_projection_metal(ctx, report, layer, "q_proj", &attn_input)?;
    let k = layer_projection_metal(ctx, report, layer, "k_proj", &attn_input)?;
    let v = layer_projection_metal(ctx, report, layer, "v_proj", &attn_input)?;
    let q = ctx.rope_row_profiled(&q, Q_HEADS, position)?;
    let k = ctx.rope_row_profiled(&k, KV_HEADS, position)?;
    Ok((q, k, v))
}

fn layer_projection_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    layer: usize,
    projection: &str,
    input: &[f32],
) -> Result<Vec<f32>> {
    let weight_name = format!("model.layers.{layer}.self_attn.{projection}.weight");
    let bias_name = format!("model.layers.{layer}.self_attn.{projection}.bias");
    ctx.bf16_linear_matvec(report, &weight_name, &bias_name, input)
}

fn layer_router_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    let weight_name = format!("model.layers.{layer}.mlp.router.weight");
    let bias_name = format!("model.layers.{layer}.mlp.router.bias");
    ctx.bf16_linear_matvec(report, &weight_name, &bias_name, input)
}

fn layer_moe_top4_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    router: &[f32],
) -> Result<Vec<f32>> {
    let top_experts = ctx.top4_softmax_profiled(router)?;
    if top_experts.len() != 4 {
        return Err(eyre!(
            "Metal router returned {} experts, expected 4",
            top_experts.len()
        ));
    }

    let mut downs = Vec::with_capacity(4);
    for expert in &top_experts {
        downs.push(layer_expert_down_proj_metal(
            ctx,
            report,
            layer,
            expert.index,
            input,
        )?);
    }

    ctx.weighted_sum4_profiled(
        [&downs[0], &downs[1], &downs[2], &downs[3]],
        [
            top_experts[0].weight,
            top_experts[1].weight,
            top_experts[2].weight,
            top_experts[3].weight,
        ],
    )
}

fn layer0_top_expert_gate_up_parts(
    report: &SourceModelReport,
    token: u32,
) -> Result<(ExpertScore, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let router = layer0_router_cpu(report, token)?;
    let Some(expert) = backend_cpu::top_k_softmax_reference(&router, 4)
        .first()
        .cloned()
    else {
        return Err(eyre!("router top-4 returned no experts"));
    };
    let input = layer0_router_input_cpu(report, token)?;
    let prefix = "model.layers.0.mlp.experts";
    let blocks_name = format!("{prefix}.gate_up_proj_blocks");
    let scales_name = format!("{prefix}.gate_up_proj_scales");
    let bias_name = format!("{prefix}.gate_up_proj_bias");

    let mut gate_up = backend_cpu::mxfp4_expert_matvec_reference(
        report,
        &blocks_name,
        &scales_name,
        expert.index,
        GATE_UP_VALUES,
        &input,
    )?;
    let bias = model_store::read_bf16_matrix_row(report, &bias_name, expert.index)?;
    model_store::add_in_place(&mut gate_up, &bias, &format!("{prefix}.gate_up_proj"))?;

    Ok((expert, input, bias, gate_up))
}

fn layer_expert_down_proj_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    layer: usize,
    expert: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    let prefix = format!("model.layers.{layer}.mlp.experts");

    let gate_up_blocks_name = format!("{prefix}.gate_up_proj_blocks");
    let gate_up_scales_name = format!("{prefix}.gate_up_proj_scales");
    let gate_up_bias_name = format!("{prefix}.gate_up_proj_bias");
    let gate_up = ctx.mxfp4_expert_matvec(
        report,
        &gate_up_blocks_name,
        &gate_up_scales_name,
        &gate_up_bias_name,
        expert,
        GATE_UP_VALUES,
        input,
    )?;
    let swiglu = ctx.swiglu_profiled(&gate_up)?;

    let down_blocks_name = format!("{prefix}.down_proj_blocks");
    let down_scales_name = format!("{prefix}.down_proj_scales");
    let down_bias_name = format!("{prefix}.down_proj_bias");
    ctx.mxfp4_expert_matvec(
        report,
        &down_blocks_name,
        &down_scales_name,
        &down_bias_name,
        expert,
        HIDDEN_SIZE,
        &swiglu,
    )
}

fn selected_logits_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    final_hidden: &[f32],
    logit_tokens: &[u32],
) -> Result<Vec<SelectedLogit>> {
    let mut rows = Vec::with_capacity(logit_tokens.len() * final_hidden.len());
    for token in logit_tokens {
        let row =
            model_store::read_bf16_matrix_row_bits(report, "lm_head.weight", *token as usize)?;
        if row.len() != final_hidden.len() {
            return Err(eyre!(
                "lm_head row {} has {} values, but final hidden has {} values",
                token,
                row.len(),
                final_hidden.len()
            ));
        }
        rows.extend(row);
    }

    let bias = vec![0.0f32; logit_tokens.len()];
    let logits = ctx.platform.bf16_matvec(
        &rows,
        logit_tokens.len(),
        final_hidden.len(),
        final_hidden,
        &bias,
    )?;
    Ok(logit_tokens
        .iter()
        .copied()
        .zip(logits)
        .map(|(token, logit)| SelectedLogit { token, logit })
        .collect())
}

fn greedy_top1_text(
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    final_hidden: &[f32],
) -> Result<GreedyTokenReport> {
    let Some((token, logit)) =
        model_store::top_k_matvec_bf16(report, "lm_head.weight", final_hidden, 1)?
            .into_iter()
            .next()
    else {
        return Err(eyre!("lm_head top-1 returned no tokens"));
    };
    let token = token as u32;
    Ok(GreedyTokenReport {
        token,
        logit,
        text: decode_token_text(harmony, token)?,
    })
}

fn lm_head_topk_cpu(
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    final_hidden: &[f32],
    k: usize,
) -> Result<Vec<GreedyTokenReport>> {
    model_store::top_k_matvec_bf16(report, "lm_head.weight", final_hidden, k)?
        .into_iter()
        .map(|(token, logit)| {
            let token = token as u32;
            Ok(GreedyTokenReport {
                token,
                logit,
                text: decode_token_text(harmony, token)?,
            })
        })
        .collect()
}

fn lm_head_topk_metal(
    ctx: &MetalOracleContext,
    harmony: &HarmonyAdapter,
    final_hidden: &[f32],
    k: usize,
) -> Result<Vec<GreedyTokenReport>> {
    ctx.lm_head_topk(final_hidden, k)?
        .into_iter()
        .map(|(token, logit)| {
            let token = token as u32;
            Ok(GreedyTokenReport {
                token,
                logit,
                text: decode_token_text(harmony, token)?,
            })
        })
        .collect()
}

fn decode_token_text(harmony: &HarmonyAdapter, token: u32) -> Result<String> {
    match harmony.decode_utf8(&[token]) {
        Ok(text) => Ok(text),
        Err(_) => {
            let bytes = harmony.decode_bytes(&[token])?;
            Ok(format!("<bytes {bytes:?}>"))
        }
    }
}

fn decode_tokens_text(harmony: &HarmonyAdapter, tokens: &[u32]) -> Result<String> {
    match harmony.decode_utf8(tokens) {
        Ok(text) => Ok(text),
        Err(_) => {
            let bytes = harmony.decode_bytes(tokens)?;
            Ok(format!("<bytes {bytes:?}>"))
        }
    }
}

fn metal_sampler_description(sampling: &SamplingConfig) -> String {
    if sampling.temperature <= 0.0 {
        "cached Metal BF16 lm_head logits + Metal top-1".to_string()
    } else {
        format!(
            "cached Metal BF16 lm_head logits + seeded temperature/top-k/top-p sampler (seed {}, temperature {:.4}, top_k {}, top_p {:.4})",
            sampling.seed, sampling.temperature, sampling.top_k, sampling.top_p
        )
    }
}

fn lm_head_topk_report(
    name: &str,
    position: usize,
    layers: usize,
    k: usize,
    cpu: Vec<GreedyTokenReport>,
    metal: Vec<GreedyTokenReport>,
) -> Result<LmHeadTopKProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal lm_head top-k returned {} tokens, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut tokens_match = true;
    let mut max_abs_delta = 0.0f32;
    for (cpu, metal) in cpu.iter().zip(&metal) {
        tokens_match &= cpu.token == metal.token;
        max_abs_delta = max_abs_delta.max((cpu.logit - metal.logit).abs());
    }

    Ok(LmHeadTopKProbeReport {
        name: name.to_string(),
        position,
        layers,
        k,
        tokens_match,
        max_abs_delta,
        cpu,
        metal,
    })
}

fn layer0_projection_cpu(
    report: &SourceModelReport,
    token: u32,
    projection: &str,
) -> Result<Vec<f32>> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.input_layernorm.weight")?;
    let input = backend_cpu::rms_norm_reference(&embedding, &norm_weight)?;
    let weight_name = format!("model.layers.0.self_attn.{projection}.weight");
    let bias_name = format!("model.layers.0.self_attn.{projection}.bias");
    let report_name = format!("model.layers.0.self_attn.{projection}");
    let mut out = model_store::matvec_bf16(report, &weight_name, &input)?;
    let bias = model_store::read_bf16_vector(report, &bias_name)?;
    model_store::add_in_place(&mut out, &bias, &report_name)?;
    Ok(out)
}

fn layer0_embedding_rows(report: &SourceModelReport, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
    let mut rows = Vec::with_capacity(tokens.len());
    for token in tokens {
        rows.push(model_store::read_bf16_matrix_row(
            report,
            "model.embed_tokens.weight",
            *token as usize,
        )?);
    }
    Ok(rows)
}

fn layer0_embedding_rows_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<Vec<Vec<f32>>> {
    let mut rows = Vec::with_capacity(tokens.len());
    for token in tokens {
        rows.push(
            ctx.bf16_matrix_row(report, "model.embed_tokens.weight", *token as usize)?
                .as_ref()
                .clone(),
        );
    }
    Ok(rows)
}

fn decode_sequence_tokens(prefill_tokens: &[u32], decode_token: u32) -> Result<Vec<u32>> {
    decode_total_tokens(prefill_tokens)?;
    let mut tokens = Vec::with_capacity(prefill_tokens.len() + 1);
    tokens.extend_from_slice(prefill_tokens);
    tokens.push(decode_token);
    Ok(tokens)
}

fn decode_total_tokens(prefill_tokens: &[u32]) -> Result<usize> {
    let total_tokens = prefill_tokens
        .len()
        .checked_add(1)
        .ok_or_else(|| eyre!("decode token count overflow"))?;
    if total_tokens > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "decode-one probe supports at most {MAX_PREFILL_PROBE_TOKENS} total tokens, got {total_tokens}"
        ));
    }
    Ok(total_tokens)
}

fn layer0_sequence_projection_cpu(
    report: &SourceModelReport,
    tokens: &[u32],
    projection: &str,
) -> Result<Vec<Vec<f32>>> {
    let mut rows = Vec::with_capacity(tokens.len());
    for token in tokens {
        rows.push(layer0_projection_cpu(report, *token, projection)?);
    }
    Ok(rows)
}

fn layer0_rope_qkv_cpu(
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let mut q = layer0_sequence_projection_cpu(report, tokens, "q_proj")?;
    let mut k = layer0_sequence_projection_cpu(report, tokens, "k_proj")?;
    let v = layer0_sequence_projection_cpu(report, tokens, "v_proj")?;
    for (position, row) in q.iter_mut().enumerate() {
        *row = backend_cpu::apply_rope_reference(row, Q_HEADS, position)?;
    }
    for (position, row) in k.iter_mut().enumerate() {
        *row = backend_cpu::apply_rope_reference(row, KV_HEADS, position)?;
    }
    Ok((q, k, v))
}

fn kv_cache_attention_probe(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tokens: &[u32],
    layer: usize,
    name: &str,
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!("KV-cache attention probe needs at least one token"));
    };
    if tokens.len() > MAX_KV_CACHE_PROBE_TOKENS {
        return Err(eyre!(
            "KV-cache attention probe supports at most {MAX_KV_CACHE_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let (q, k, v) = layer0_rope_qkv_cpu(report, tokens)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    let cpu = backend_cpu::sequence_attention_from_rope_reference(layer, &q, &k, &v, &sinks)?;
    let query_position = tokens.len() - 1;
    let Some(cpu) = cpu.get(query_position) else {
        return Err(eyre!("sequence attention returned no row {query_position}"));
    };

    let mut kv_cache = PlannedKvCache::new(KvCachePlan::gpt_oss_20b(tokens.len())?);
    for position in 0..tokens.len() {
        kv_cache
            .layer_mut(layer)?
            .push(position, &k[position], &v[position])?;
    }
    let view = kv_cache
        .layer(layer)?
        .contiguous_view_for_query(query_position)?;
    let metal = ctx.platform.kv_cache_decode_attention(
        layer,
        query_position,
        view.start_position,
        &q[query_position],
        &view.k,
        &view.v,
        &sinks,
    )?;

    vector_report(name, token, query_position, cpu, metal)
}

fn flatten_rows(rows: &[Vec<f32>]) -> Vec<f32> {
    let values = rows.iter().map(Vec::len).sum();
    let mut flat = Vec::with_capacity(values);
    for row in rows {
        flat.extend_from_slice(row);
    }
    flat
}

fn layer0_single_token_attention_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let q = layer0_projection_cpu(report, token, "q_proj")?;
    let k = layer0_projection_cpu(report, token, "k_proj")?;
    let v = layer0_projection_cpu(report, token, "v_proj")?;
    let q = backend_cpu::apply_rope_reference(&q, Q_HEADS, 0)?;
    let k = backend_cpu::apply_rope_reference(&k, KV_HEADS, 0)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    backend_cpu::single_token_attention_from_rope_reference(&q, &k, &v, &sinks)
}

fn layer0_o_proj_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let attention = layer0_single_token_attention_cpu(report, token)?;
    let weight_name = "model.layers.0.self_attn.o_proj.weight";
    let bias_name = "model.layers.0.self_attn.o_proj.bias";
    let mut out = model_store::matvec_bf16(report, weight_name, &attention)?;
    let bias = model_store::read_bf16_vector(report, bias_name)?;
    model_store::add_in_place(&mut out, &bias, "model.layers.0.self_attn.o_proj")?;
    Ok(out)
}

fn layer0_attention_residual_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let o_proj = layer0_o_proj_cpu(report, token)?;
    add_vectors(&embedding, &o_proj)
}

fn layer0_router_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let input = layer0_router_input_cpu(report, token)?;
    let weight_name = "model.layers.0.mlp.router.weight";
    let bias_name = "model.layers.0.mlp.router.bias";
    let mut router = model_store::matvec_bf16(report, weight_name, &input)?;
    let bias = model_store::read_bf16_vector(report, bias_name)?;
    model_store::add_in_place(&mut router, &bias, "model.layers.0.mlp.router")?;
    Ok(router)
}

fn layer0_router_input_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let residual = layer0_attention_residual_cpu(report, token)?;
    let norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.post_attention_layernorm.weight")?;
    backend_cpu::rms_norm_reference(&residual, &norm_weight)
}

fn read_mxfp4_expert_blocks(
    report: &SourceModelReport,
    tensor_name: &str,
    expert: usize,
    rows: usize,
) -> Result<Vec<u8>> {
    let blocks_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
        .ok_or_else(|| eyre!("MXFP4 block slice size overflow"))?;
    let offset = expert
        .checked_mul(blocks_per_expert)
        .ok_or_else(|| eyre!("MXFP4 block offset overflow"))?;
    model_store::read_u8_tensor_slice(report, tensor_name, offset, blocks_per_expert)
}

fn read_mxfp4_expert_blocks_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tensor_name: &str,
    expert: usize,
    rows: usize,
) -> Result<Arc<Vec<u8>>> {
    let blocks_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
        .ok_or_else(|| eyre!("MXFP4 block slice size overflow"))?;
    let offset = expert
        .checked_mul(blocks_per_expert)
        .ok_or_else(|| eyre!("MXFP4 block offset overflow"))?;
    ctx.u8_tensor_slice(report, tensor_name, offset, blocks_per_expert)
}

fn read_mxfp4_expert_scales(
    report: &SourceModelReport,
    tensor_name: &str,
    expert: usize,
    rows: usize,
) -> Result<Vec<u8>> {
    let scales_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .ok_or_else(|| eyre!("MXFP4 scale slice size overflow"))?;
    let offset = expert
        .checked_mul(scales_per_expert)
        .ok_or_else(|| eyre!("MXFP4 scale offset overflow"))?;
    model_store::read_u8_tensor_slice(report, tensor_name, offset, scales_per_expert)
}

fn read_mxfp4_expert_scales_metal(
    ctx: &MetalOracleContext,
    report: &SourceModelReport,
    tensor_name: &str,
    expert: usize,
    rows: usize,
) -> Result<Arc<Vec<u8>>> {
    let scales_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .ok_or_else(|| eyre!("MXFP4 scale slice size overflow"))?;
    let offset = expert
        .checked_mul(scales_per_expert)
        .ok_or_else(|| eyre!("MXFP4 scale offset overflow"))?;
    ctx.u8_tensor_slice(report, tensor_name, offset, scales_per_expert)
}

fn add_vectors(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        return Err(eyre!(
            "vector add length mismatch: left {}, right {}",
            left.len(),
            right.len()
        ));
    }
    Ok(left
        .iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(left, right)| left + right)
        .collect())
}

fn top_k_report(
    name: &str,
    token: u32,
    cpu: Vec<ExpertScore>,
    metal: Vec<ExpertScore>,
) -> Result<MetalTopKProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal top-k returned {} experts, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut indices_match = true;
    let mut max_logit_delta = 0.0f32;
    let mut max_weight_delta = 0.0f32;
    for (cpu, metal) in cpu.iter().zip(&metal) {
        indices_match &= cpu.index == metal.index;
        max_logit_delta = max_logit_delta.max((cpu.logit - metal.logit).abs());
        max_weight_delta = max_weight_delta.max((cpu.weight - metal.weight).abs());
    }

    Ok(MetalTopKProbeReport {
        name: name.to_string(),
        token,
        indices_match,
        max_logit_delta,
        max_weight_delta,
        cpu,
        metal,
    })
}

fn selected_logits_report(
    name: &str,
    token: u32,
    layers: usize,
    cpu: Vec<SelectedLogit>,
    metal: Vec<SelectedLogit>,
) -> Result<MetalSelectedLogitsProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal selected-logits probe returned {} logits, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut max_abs_delta = 0.0f32;
    let mut sum_abs_delta = 0.0f64;
    for (cpu, metal) in cpu.iter().zip(&metal) {
        if cpu.token != metal.token {
            return Err(eyre!(
                "selected-logit token mismatch: CPU {}, Metal {}",
                cpu.token,
                metal.token
            ));
        }
        let delta = (cpu.logit - metal.logit).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_abs_delta += delta as f64;
    }
    let mean_abs_delta = if cpu.is_empty() {
        0.0
    } else {
        (sum_abs_delta / cpu.len() as f64) as f32
    };

    Ok(MetalSelectedLogitsProbeReport {
        name: name.to_string(),
        token,
        layers,
        max_abs_delta,
        mean_abs_delta,
        cpu,
        metal,
    })
}

fn matvec_report(
    name: &str,
    token: u32,
    rows: usize,
    cols: usize,
    cpu: &[f32],
    metal: Vec<f32>,
) -> Result<MetalMatvecProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal matvec returned {} values, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut max_abs_delta = 0.0f32;
    let mut sum_abs_delta = 0.0f64;
    for (cpu, metal) in cpu.iter().copied().zip(metal.iter().copied()) {
        let delta = (cpu - metal).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_abs_delta += delta as f64;
    }
    let mean_abs_delta = if cpu.is_empty() {
        0.0
    } else {
        (sum_abs_delta / cpu.len() as f64) as f32
    };

    Ok(MetalMatvecProbeReport {
        name: name.to_string(),
        token,
        rows,
        cols,
        max_abs_delta,
        mean_abs_delta,
        cpu_first8: cpu.iter().copied().take(8).collect(),
        metal_first8: metal.into_iter().take(8).collect(),
    })
}

fn vector_report(
    name: &str,
    token: u32,
    position: usize,
    cpu: &[f32],
    metal: Vec<f32>,
) -> Result<MetalVectorProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal vector probe returned {} values, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut max_abs_delta = 0.0f32;
    let mut sum_abs_delta = 0.0f64;
    for (cpu, metal) in cpu.iter().copied().zip(metal.iter().copied()) {
        let delta = (cpu - metal).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_abs_delta += delta as f64;
    }
    let mean_abs_delta = if cpu.is_empty() {
        0.0
    } else {
        (sum_abs_delta / cpu.len() as f64) as f32
    };

    Ok(MetalVectorProbeReport {
        name: name.to_string(),
        token,
        position,
        values: cpu.len(),
        max_abs_delta,
        mean_abs_delta,
        cpu_first8: cpu.iter().copied().take(8).collect(),
        metal_first8: metal.into_iter().take(8).collect(),
    })
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{
        ATTN_VALUES, EXPERTS, GATE_UP_VALUES, HIDDEN_SIZE, KV_VALUES, LM_HEAD_TOP1_BLOCK_SIZE,
        MAX_KV_CACHE_PROBE_TOKENS, MAX_PREFILL_PROBE_TOKENS, MXFP4_BYTES_PER_GROUP, MXFP4_GROUPS,
        Q_HEADS,
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
        bf16_matvec: Retained<MetalComputePipelineState>,
        bf16_matvec_logits: Retained<MetalComputePipelineState>,
        top1_logits_blocks: Retained<MetalComputePipelineState>,
        top1_logits_final: Retained<MetalComputePipelineState>,
        topk_logits: Retained<MetalComputePipelineState>,
        embedding_lookup_bf16: Retained<MetalComputePipelineState>,
        rope_row: Retained<MetalComputePipelineState>,
        single_token_attention: Retained<MetalComputePipelineState>,
        sequence_attention: Retained<MetalComputePipelineState>,
        kv_cache_decode_attention: Retained<MetalComputePipelineState>,
        write_f32_slot: Retained<MetalComputePipelineState>,
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
                bf16_matvec: pipeline(&device, &library, "bf16_matvec")?,
                bf16_matvec_logits: pipeline(&device, &library, "bf16_matvec_logits")?,
                top1_logits_blocks: pipeline(&device, &library, "top1_logits_blocks")?,
                top1_logits_final: pipeline(&device, &library, "top1_logits_final")?,
                topk_logits: pipeline(&device, &library, "topk_logits")?,
                embedding_lookup_bf16: pipeline(&device, &library, "embedding_lookup_bf16")?,
                rope_row: pipeline(&device, &library, "rope_row")?,
                single_token_attention: pipeline(&device, &library, "single_token_attention")?,
                sequence_attention: pipeline(&device, &library, "sequence_attention")?,
                kv_cache_decode_attention: pipeline(
                    &device,
                    &library,
                    "kv_cache_decode_attention",
                )?,
                write_f32_slot: pipeline(&device, &library, "write_f32_slot")?,
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
            if layer % 2 == 0 && query_position + 1 > MAX_PREFILL_PROBE_TOKENS {
                effective_key_start =
                    effective_key_start.max(query_position + 1 - MAX_PREFILL_PROBE_TOKENS);
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
            let effective_key_start =
                if layer % 2 == 0 && query_position + 1 > MAX_PREFILL_PROBE_TOKENS {
                    cache_start_position.max(query_position + 1 - MAX_PREFILL_PROBE_TOKENS)
                } else {
                    cache_start_position
                };
            let key_count = query_position + 1 - effective_key_start;
            if key_count > MAX_KV_CACHE_PROBE_TOKENS {
                return Err(eyre!(
                    "resident KV decode currently supports at most {MAX_KV_CACHE_PROBE_TOKENS} keys, got {key_count}"
                ));
            }

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
mod platform {
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

        pub fn rms_norm_into(
            &self,
            _input: &F32VectorBuffer,
            _weight: &F32VectorBuffer,
            _out: &F32VectorBuffer,
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

        pub fn write_f32_slot_into(
            &self,
            _input: &F32VectorBuffer,
            _output: &F32VectorBuffer,
            _slot: usize,
            _width: usize,
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
