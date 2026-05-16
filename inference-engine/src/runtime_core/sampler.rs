use super::SamplingConfig;
use eyre::{Result, eyre};

#[derive(Debug, Clone)]
pub struct Sampler {
    pub config: SamplingConfig,
    rng: SplitMix64,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            rng: SplitMix64::new(config.seed),
            config,
        }
    }

    pub fn candidate_count(&self) -> usize {
        if self.config.temperature <= 0.0 {
            return 1;
        }
        let top_k = self.config.top_k;
        if top_k == 0 {
            8
        } else {
            top_k.clamp(1, 8) as usize
        }
    }

    pub fn needs_full_vocab(&self) -> bool {
        self.config.temperature > 0.0 && self.config.top_k == 0 && self.config.top_p < 1.0
    }

    pub fn choose(&mut self, candidates: &[SampleCandidate]) -> Result<SampleCandidate> {
        let Some(greedy) = candidates.first().copied() else {
            return Err(eyre!("sampler needs at least one candidate"));
        };
        if self.config.temperature <= 0.0 || candidates.len() == 1 {
            return Ok(greedy);
        }

        let temperature = self.config.temperature.max(1e-6);
        let mut filtered = candidates.to_vec();
        let max_logit = filtered
            .iter()
            .map(|candidate| candidate.logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0f32;
        for candidate in &mut filtered {
            candidate.probability = ((candidate.logit - max_logit) / temperature).exp();
            total += candidate.probability;
        }
        if total <= 0.0 || !total.is_finite() {
            return Ok(greedy);
        }
        for candidate in &mut filtered {
            candidate.probability /= total;
        }

        filtered.sort_by(|left, right| {
            right
                .probability
                .total_cmp(&left.probability)
                .then_with(|| left.token.cmp(&right.token))
        });

        let top_p = self.config.top_p.clamp(0.0, 1.0);
        if top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut keep = 0usize;
            for candidate in &filtered {
                cumulative += candidate.probability;
                keep += 1;
                if cumulative >= top_p {
                    break;
                }
            }
            filtered.truncate(keep.max(1));
            let total = filtered
                .iter()
                .map(|candidate| candidate.probability)
                .sum::<f32>();
            if total > 0.0 && total.is_finite() {
                for candidate in &mut filtered {
                    candidate.probability /= total;
                }
            }
        }

        let mut threshold = self.rng.next_f32();
        for candidate in filtered {
            threshold -= candidate.probability;
            if threshold <= 0.0 {
                return Ok(candidate);
            }
        }

        Ok(greedy)
    }

    pub fn choose_from_logits(&mut self, logits: &[f32]) -> Result<SampleCandidate> {
        let Some(greedy) = greedy_from_logits(logits) else {
            return Err(eyre!("sampler needs at least one finite logit"));
        };
        if self.config.temperature <= 0.0 {
            return Ok(greedy);
        }

        let mut filtered = logits
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(token, logit)| {
                if !logit.is_finite() {
                    return None;
                }
                Some(SampleCandidate {
                    token: token as u32,
                    logit,
                    probability: 0.0,
                })
            })
            .collect::<Vec<_>>();
        if filtered.is_empty() {
            return Ok(greedy);
        }

        filtered.sort_by(compare_logit_desc);
        if self.config.top_k > 0 {
            filtered.truncate(self.config.top_k as usize);
        }

        let temperature = self.config.temperature.max(1e-6);
        let max_logit = filtered
            .iter()
            .map(|candidate| candidate.logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0f32;
        for candidate in &mut filtered {
            candidate.probability = ((candidate.logit - max_logit) / temperature).exp();
            total += candidate.probability;
        }
        if total <= 0.0 || !total.is_finite() {
            return Ok(greedy);
        }
        for candidate in &mut filtered {
            candidate.probability /= total;
        }

        let top_p = self.config.top_p.clamp(0.0, 1.0);
        if top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut keep = 0usize;
            for candidate in &filtered {
                cumulative += candidate.probability;
                keep += 1;
                if cumulative >= top_p {
                    break;
                }
            }
            filtered.truncate(keep.max(1));
            let total = filtered
                .iter()
                .map(|candidate| candidate.probability)
                .sum::<f32>();
            if total > 0.0 && total.is_finite() {
                for candidate in &mut filtered {
                    candidate.probability /= total;
                }
            }
        }

        let mut threshold = self.rng.next_f32();
        for candidate in filtered {
            threshold -= candidate.probability;
            if threshold <= 0.0 {
                return Ok(candidate);
            }
        }

        Ok(greedy)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SampleCandidate {
    pub token: u32,
    pub logit: f32,
    pub probability: f32,
}

fn greedy_from_logits(logits: &[f32]) -> Option<SampleCandidate> {
    logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .max_by(compare_indexed_logit)
        .map(|(token, logit)| SampleCandidate {
            token: token as u32,
            logit,
            probability: 1.0,
        })
}

fn compare_indexed_logit(left: &(usize, f32), right: &(usize, f32)) -> std::cmp::Ordering {
    left.1
        .total_cmp(&right.1)
        .then_with(|| right.0.cmp(&left.0))
}

fn compare_logit_desc(left: &SampleCandidate, right: &SampleCandidate) -> std::cmp::Ordering {
    right
        .logit
        .total_cmp(&left.logit)
        .then_with(|| left.token.cmp(&right.token))
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        (value as f32) / ((1u64 << 24) as f32)
    }
}
