#[cfg(feature = "profile")]
use std::collections::HashMap;
#[cfg(feature = "profile")]
use std::fmt;
use std::time::Duration;

#[cfg(feature = "profile")]
#[derive(Debug, Clone)]
pub struct MetalProfile {
    pub records: Vec<MetalProfileRecord>,
    #[cfg(feature = "profile")]
    pub stage_profile: Option<MetalStageProfile>,
    #[cfg(feature = "profile")]
    pub counter_sampling: String,
}

#[cfg(feature = "profile")]
impl fmt::Display for MetalProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.cli_text())
    }
}

#[cfg(feature = "profile")]
impl MetalProfile {
    fn cli_text(&self) -> String {
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
        out.push_str(&format!("- counter samples: {}\n", self.counter_sampling));
        out.push_str("\nkey metrics:\n");
        out.push_str(&format!(
            "- hot token door-to-door: {}\n",
            render_average_metric(
                records
                    .iter()
                    .find(|record| record.name == "metric.hot_token")
            )
            .unwrap_or_else(|| "n/a; generate at least 2 tokens".to_string())
        ));
        out.push_str(&format!(
            "- hot token GPU: {}\n",
            render_average_gpu_metric(
                records
                    .iter()
                    .find(|record| record.name == "metric.hot_token")
            )
            .unwrap_or_else(|| "n/a; generate at least 2 tokens".to_string())
        ));
        out.push_str(&format!(
            "- hot token wall/GPU gap: {}\n",
            render_average_metric(
                records
                    .iter()
                    .find(|record| record.name == "metric.hot_token_gap")
            )
            .unwrap_or_else(|| "n/a; generate at least 2 tokens".to_string())
        ));
        out.push_str(&format!(
            "- experts carousel page-spill time: {}\n",
            render_spill_time(
                records
                    .iter()
                    .find(|record| record.name == "metric.experts_carousel_page")
            )
        ));
        out.push_str(&format!(
            "- experts carousel page-spills: {}\n",
            render_spill_count(
                records
                    .iter()
                    .find(|record| record.name == "metric.experts_carousel_page")
            )
        ));
        out.push_str(&format!(
            "- cold inference to first token: {}\n",
            render_single_metric(
                records
                    .iter()
                    .find(|record| record.name == "metric.cold_start_to_first_token")
            )
            .unwrap_or_else(|| "n/a".to_string())
        ));
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
        #[cfg(feature = "profile")]
        if let Some(stage_profile) = &self.stage_profile {
            out.push_str(&stage_profile.render_hot_token_breakdown());
            out.push_str(&stage_profile.cli_text());
        }
        out
    }
}

#[cfg(feature = "profile")]
#[derive(Debug, Clone)]
pub struct MetalStageProfile {
    token_positions: Vec<Option<usize>>,
    stage_names: Vec<&'static str>,
    values_ns: Vec<Vec<u128>>,
    gpu_stage_names: Vec<&'static str>,
    gpu_values_ns: Vec<Vec<u128>>,
}

#[cfg(feature = "profile")]
impl MetalStageProfile {
    fn render_hot_token_breakdown(&self) -> String {
        let mut positions = self
            .token_positions
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        positions.sort_unstable();

        let mut combined_rows = Vec::new();
        let mut stitched_rows = Vec::new();
        for position in positions {
            let hot_token_gpu = self.value_at(position, TokenStage::HotToken);
            if hot_token_gpu != 0 {
                combined_rows.push((position, hot_token_gpu));
                continue;
            }
            if position != 0 {
                let decode_gpu = self.value_at(position - 1, TokenStage::Token);
                let lm_head_gpu = self.value_at(position, TokenStage::LmHead);
                if decode_gpu != 0 && lm_head_gpu != 0 {
                    stitched_rows.push((position, decode_gpu, lm_head_gpu));
                }
            }
        }

        let mut out = String::new();
        out.push_str("\nhot token stage breakdown:\n");
        if !combined_rows.is_empty() {
            out.push_str("- source: one command buffer per hot generated token\n\n");
            out.push_str("token      hot_token_gpu\n");
            out.push_str("---------  -------------\n");
            for (position, hot_token_gpu) in combined_rows {
                out.push_str(&format!(
                    "{position:>9}  {:>13}\n",
                    format_duration_ns(hot_token_gpu)
                ));
            }
            return out;
        }

        out.push_str("- source: GPU timestamps stitched as decode(previous token) + lm_head(current token)\n\n");
        if stitched_rows.is_empty() {
            out.push_str("(no full hot-token stage sample; generate at least 2 tokens)\n");
            return out;
        }

        out.push_str("token      decode_gpu     lm_head_gpu     gpu_total\n");
        out.push_str("---------  -------------  --------------  -------------\n");
        for (position, decode_gpu, lm_head_gpu) in stitched_rows {
            out.push_str(&format!(
                "{position:>9}  {:>13}  {:>14}  {:>13}\n",
                format_duration_ns(decode_gpu),
                format_duration_ns(lm_head_gpu),
                format_duration_ns(decode_gpu.saturating_add(lm_head_gpu))
            ));
        }
        out
    }

    fn render_gpu_stage_breakdown(&self) -> String {
        let active_stages = self.active_gpu_stages();
        let mut rows = self
            .token_positions
            .iter()
            .enumerate()
            .filter_map(|(slot, position)| Some((slot, (*position)?)))
            .filter(|(slot, _)| self.gpu_values_ns[*slot].iter().any(|ns| *ns != 0))
            .collect::<Vec<_>>();
        rows.sort_by_key(|(_, position)| *position);

        let mut out = String::new();
        out.push_str("\nhot token dispatch-stage profile:\n");
        out.push_str("- source: Metal counter samples around compute encoders\n");
        out.push_str(
            "- note: profile builds insert counter samples and are not production timing\n\n",
        );
        if rows.is_empty() || active_stages.is_empty() {
            out.push_str(
                "(no dispatch-stage samples; counter samples may be unsupported on this device)\n",
            );
            return out;
        }

        out.push_str("token      sampled_total");
        for stage in &active_stages {
            out.push_str(&format!("  {:>14}", self.gpu_stage_names[*stage]));
        }
        out.push('\n');
        out.push_str("---------  -------------");
        for _ in &active_stages {
            out.push_str("  --------------");
        }
        out.push('\n');

        let mut totals = vec![0u128; self.gpu_stage_names.len()];
        let mut row_count = 0u128;
        for (slot, position) in &rows {
            row_count += 1;
            let sampled_total = self.gpu_values_ns[*slot].iter().sum::<u128>();
            out.push_str(&format!(
                "{position:>9}  {:>13}",
                format_duration_ns(sampled_total)
            ));
            for stage in &active_stages {
                let value = self.gpu_values_ns[*slot][*stage];
                totals[*stage] = totals[*stage].saturating_add(value);
                out.push_str(&format!("  {:>14}", format_duration_ns(value)));
            }
            out.push('\n');
        }

        if row_count > 1 {
            out.push_str("average    ");
            let average_total = active_stages
                .iter()
                .map(|stage| totals[*stage])
                .sum::<u128>()
                / row_count;
            out.push_str(&format!("{:>13}", format_duration_ns(average_total)));
            for stage in &active_stages {
                out.push_str(&format!(
                    "  {:>14}",
                    format_duration_ns(totals[*stage] / row_count)
                ));
            }
            out.push('\n');
        }
        out
    }

    fn active_gpu_stages(&self) -> Vec<usize> {
        (0..self.gpu_stage_names.len())
            .filter(|stage| self.gpu_values_ns.iter().any(|row| row[*stage] != 0))
            .collect()
    }

    fn value_at(&self, token_position: usize, stage: TokenStage) -> u128 {
        if self.token_positions.is_empty() {
            return 0;
        }
        let slot = token_position % self.token_positions.len();
        if self.token_positions[slot] != Some(token_position) {
            return 0;
        }
        self.values_ns[slot][stage.index()]
    }

    fn cli_text(&self) -> String {
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
        out.push_str(&self.render_gpu_stage_breakdown());
        out
    }
}

#[cfg(feature = "profile")]
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
#[allow(dead_code)]
pub(crate) struct ProfileDelta {
    pub(crate) wall: Duration,
    pub(crate) gpu_ns: u128,
    pub(crate) command_buffers: usize,
    pub(crate) upload_bytes: usize,
    pub(crate) readback_bytes: usize,
    pub(crate) cache_hits: usize,
    pub(crate) cache_misses: usize,
}

#[cfg(feature = "profile")]
#[derive(Default)]
pub(crate) struct ProfileState {
    pub(crate) records: HashMap<String, MetalProfileRecord>,
    pub(crate) stage_profile: Option<StageProfileState>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) enum TokenStage {
    Token,
    LmHead,
    HotToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum GpuStage {
    Other,
    Embedding,
    InputNormQkv,
    RopeKvWrite,
    Attention,
    AttnProj,
    RouterTop4,
    ExpertsGate,
    ExpertsDown,
    LmHead,
}

impl GpuStage {
    #[cfg(feature = "profile")]
    pub(crate) const ALL: [Self; 10] = [
        Self::Other,
        Self::Embedding,
        Self::InputNormQkv,
        Self::RopeKvWrite,
        Self::Attention,
        Self::AttnProj,
        Self::RouterTop4,
        Self::ExpertsGate,
        Self::ExpertsDown,
        Self::LmHead,
    ];

    #[cfg(feature = "profile")]
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Other => 0,
            Self::Embedding => 1,
            Self::InputNormQkv => 2,
            Self::RopeKvWrite => 3,
            Self::Attention => 4,
            Self::AttnProj => 5,
            Self::RouterTop4 => 6,
            Self::ExpertsGate => 7,
            Self::ExpertsDown => 8,
            Self::LmHead => 9,
        }
    }

    #[cfg(feature = "profile")]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Other => "other",
            Self::Embedding => "embedding",
            Self::InputNormQkv => "norm_qkv",
            Self::RopeKvWrite => "rope_kv",
            Self::Attention => "attention",
            Self::AttnProj => "attn_proj",
            Self::RouterTop4 => "router_top4",
            Self::ExpertsGate => "experts_gate",
            Self::ExpertsDown => "experts_down",
            Self::LmHead => "lm_head",
        }
    }
}

impl TokenStage {
    #[cfg(feature = "profile")]
    const ALL: [Self; 3] = [Self::Token, Self::LmHead, Self::HotToken];

    #[cfg(feature = "profile")]
    fn index(self) -> usize {
        match self {
            Self::Token => 0,
            Self::LmHead => 1,
            Self::HotToken => 2,
        }
    }

    #[cfg(feature = "profile")]
    fn name(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::LmHead => "lm_head",
            Self::HotToken => "hot_token",
        }
    }
}

#[cfg(feature = "profile")]
#[derive(Debug, Clone)]
pub(crate) struct StageProfileState {
    token_positions: Vec<Option<usize>>,
    values_ns: Vec<Vec<u128>>,
    gpu_values_ns: Vec<Vec<u128>>,
}

#[cfg(feature = "profile")]
impl StageProfileState {
    pub(crate) fn new(ring_capacity: usize) -> Self {
        Self {
            token_positions: vec![None; ring_capacity],
            values_ns: vec![vec![0; TokenStage::ALL.len()]; ring_capacity],
            gpu_values_ns: vec![vec![0; GpuStage::ALL.len()]; ring_capacity],
        }
    }

    pub(crate) fn record(&mut self, token_position: usize, stage: TokenStage, ns: u128) {
        if self.token_positions.is_empty() {
            return;
        }
        let slot = token_position % self.token_positions.len();
        if self.token_positions[slot] != Some(token_position) {
            self.token_positions[slot] = Some(token_position);
            self.values_ns[slot].fill(0);
            self.gpu_values_ns[slot].fill(0);
        }
        self.values_ns[slot][stage.index()] =
            self.values_ns[slot][stage.index()].saturating_add(ns);
    }

    pub(crate) fn record_gpu_stage(&mut self, token_position: usize, stage: GpuStage, ns: u128) {
        if self.token_positions.is_empty() || ns == 0 {
            return;
        }
        let slot = token_position % self.token_positions.len();
        if self.token_positions[slot] != Some(token_position) {
            self.token_positions[slot] = Some(token_position);
            self.values_ns[slot].fill(0);
            self.gpu_values_ns[slot].fill(0);
        }
        self.gpu_values_ns[slot][stage.index()] =
            self.gpu_values_ns[slot][stage.index()].saturating_add(ns);
    }

    pub(crate) fn snapshot(&self) -> MetalStageProfile {
        MetalStageProfile {
            token_positions: self.token_positions.clone(),
            stage_names: TokenStage::ALL.iter().map(|stage| stage.name()).collect(),
            values_ns: self.values_ns.clone(),
            gpu_stage_names: GpuStage::ALL.iter().map(|stage| stage.name()).collect(),
            gpu_values_ns: self.gpu_values_ns.clone(),
        }
    }
}

#[cfg(feature = "profile")]
pub(crate) type StageMarker = Option<(usize, TokenStage)>;

#[cfg(not(feature = "profile"))]
pub(crate) type StageMarker = ();

#[cfg(feature = "profile")]
pub(crate) fn stage_marker(position: usize, stage: TokenStage) -> StageMarker {
    Some((position, stage))
}

#[cfg(not(feature = "profile"))]
pub(crate) fn stage_marker(position: usize, stage: TokenStage) -> StageMarker {
    let _ = (position, stage);
}

#[cfg(feature = "profile")]
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

#[cfg(feature = "profile")]
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1}MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1}KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(feature = "profile")]
fn render_single_metric(record: Option<&MetalProfileRecord>) -> Option<String> {
    let record = record?;
    Some(format_duration_ns(record.wall_ns))
}

#[cfg(feature = "profile")]
fn render_average_metric(record: Option<&MetalProfileRecord>) -> Option<String> {
    let record = record?;
    if record.calls == 0 {
        return None;
    }
    let avg = record.wall_ns / record.calls as u128;
    Some(format!(
        "{} avg over {} token{}",
        format_duration_ns(avg),
        record.calls,
        plural_suffix(record.calls)
    ))
}

#[cfg(feature = "profile")]
fn render_average_gpu_metric(record: Option<&MetalProfileRecord>) -> Option<String> {
    let record = record?;
    if record.calls == 0 {
        return None;
    }
    let avg = record.gpu_ns / record.calls as u128;
    Some(format!(
        "{} avg over {} token{}",
        format_duration_ns(avg),
        record.calls,
        plural_suffix(record.calls)
    ))
}

#[cfg(feature = "profile")]
fn render_spill_time(record: Option<&MetalProfileRecord>) -> String {
    let Some(record) = record else {
        return "n/a; 0 reloads".to_string();
    };
    let spills = spill_count(record);
    if spills == 0 {
        return "n/a; 0 reloads".to_string();
    }
    let avg = record.wall_ns / spills as u128;
    format!(
        "{} avg, {} total",
        format_duration_ns(avg),
        format_duration_ns(record.wall_ns)
    )
}

#[cfg(feature = "profile")]
fn render_spill_count(record: Option<&MetalProfileRecord>) -> usize {
    record.map(spill_count).unwrap_or(0)
}

#[cfg(feature = "profile")]
fn spill_count(record: &MetalProfileRecord) -> usize {
    record.cache_misses.max(record.calls)
}

#[cfg(feature = "profile")]
fn plural_suffix(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}
