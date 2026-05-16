use std::collections::HashMap;
use std::time::Duration;

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
            "- expert slab page-spill time: {}\n",
            render_spill_time(
                records
                    .iter()
                    .find(|record| record.name == "metric.expert_slab_page")
            )
        ));
        out.push_str(&format!(
            "- expert slab page-spills: {}\n",
            render_spill_count(
                records
                    .iter()
                    .find(|record| record.name == "metric.expert_slab_page")
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
        #[cfg(feature = "metal-stage-profile")]
        if let Some(stage_profile) = &self.stage_profile {
            out.push_str(&stage_profile.render_hot_token_breakdown());
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
    fn render_hot_token_breakdown(&self) -> String {
        let mut positions = self
            .token_positions
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        positions.sort_unstable();

        let mut rows = Vec::new();
        for position in positions {
            if position == 0 {
                continue;
            }
            let decode_gpu = self.value_at(position - 1, TokenStage::Token);
            let lm_head_gpu = self.value_at(position, TokenStage::LmHead);
            if decode_gpu == 0 || lm_head_gpu == 0 {
                continue;
            }
            rows.push((position, decode_gpu, lm_head_gpu));
        }

        let mut out = String::new();
        out.push_str("\nhot token stage breakdown:\n");
        out.push_str("- source: GPU timestamps stitched as decode(previous token) + lm_head(current token)\n\n");
        if rows.is_empty() {
            out.push_str("(no full hot-token stage sample; generate at least 2 tokens)\n");
            return out;
        }

        out.push_str("token      decode_gpu     lm_head_gpu     gpu_total\n");
        out.push_str("---------  -------------  --------------  -------------\n");
        for (position, decode_gpu, lm_head_gpu) in rows {
            out.push_str(&format!(
                "{position:>9}  {:>13}  {:>14}  {:>13}\n",
                format_duration_ns(decode_gpu),
                format_duration_ns(lm_head_gpu),
                format_duration_ns(decode_gpu.saturating_add(lm_head_gpu))
            ));
        }
        out
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
pub(crate) struct ProfileDelta {
    pub(crate) wall: Duration,
    pub(crate) gpu_ns: u128,
    pub(crate) command_buffers: usize,
    pub(crate) upload_bytes: usize,
    pub(crate) readback_bytes: usize,
    pub(crate) cache_hits: usize,
    pub(crate) cache_misses: usize,
}

#[derive(Default)]
pub(crate) struct ProfileState {
    pub(crate) enabled: bool,
    pub(crate) records: HashMap<String, MetalProfileRecord>,
    #[cfg(feature = "metal-stage-profile")]
    pub(crate) stage_profile: Option<StageProfileState>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum TokenStage {
    Token,
    LmHead,
}

impl TokenStage {
    #[cfg(feature = "metal-stage-profile")]
    const ALL: [Self; 2] = [Self::Token, Self::LmHead];

    #[cfg(feature = "metal-stage-profile")]
    fn index(self) -> usize {
        match self {
            Self::Token => 0,
            Self::LmHead => 1,
        }
    }

    #[cfg(feature = "metal-stage-profile")]
    fn name(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::LmHead => "lm_head",
        }
    }
}

#[cfg(feature = "metal-stage-profile")]
#[derive(Debug, Clone)]
pub(crate) struct StageProfileState {
    token_positions: Vec<Option<usize>>,
    values_ns: Vec<Vec<u128>>,
}

#[cfg(feature = "metal-stage-profile")]
impl StageProfileState {
    pub(crate) fn new(ring_capacity: usize) -> Self {
        Self {
            token_positions: vec![None; ring_capacity],
            values_ns: vec![vec![0; TokenStage::ALL.len()]; ring_capacity],
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
        }
        self.values_ns[slot][stage.index()] =
            self.values_ns[slot][stage.index()].saturating_add(ns);
    }

    pub(crate) fn report(&self) -> MetalStageProfileReport {
        MetalStageProfileReport {
            token_positions: self.token_positions.clone(),
            stage_names: TokenStage::ALL.iter().map(|stage| stage.name()).collect(),
            values_ns: self.values_ns.clone(),
        }
    }
}

#[cfg(feature = "metal-stage-profile")]
pub(crate) type StageMarker = Option<(usize, TokenStage)>;

#[cfg(not(feature = "metal-stage-profile"))]
pub(crate) type StageMarker = ();

#[cfg(feature = "metal-stage-profile")]
pub(crate) fn stage_marker(position: usize, stage: TokenStage) -> StageMarker {
    Some((position, stage))
}

#[cfg(not(feature = "metal-stage-profile"))]
pub(crate) fn stage_marker(position: usize, stage: TokenStage) -> StageMarker {
    let _ = (position, stage);
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

fn render_single_metric(record: Option<&MetalProfileRecord>) -> Option<String> {
    let record = record?;
    Some(format_duration_ns(record.wall_ns))
}

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

fn render_spill_count(record: Option<&MetalProfileRecord>) -> usize {
    record.map(spill_count).unwrap_or(0)
}

fn spill_count(record: &MetalProfileRecord) -> usize {
    record.cache_misses.max(record.calls)
}

fn plural_suffix(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}
