use eyre::{Result, eyre};

pub const GPT_OSS_20B_LAYERS: usize = 24;
pub const GPT_OSS_KV_HEADS: usize = 8;
pub const GPT_OSS_HEAD_DIM: usize = 64;
pub const GPT_OSS_WINDOW_TOKENS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheKind {
    DenseFull,
    WindowRing { window_tokens: usize },
}

impl KvCacheKind {
    fn capacity_tokens(self, context_tokens: usize) -> usize {
        match self {
            Self::DenseFull => context_tokens,
            Self::WindowRing { window_tokens } => window_tokens.min(context_tokens),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerKvPlan {
    pub layer_index: usize,
    pub kind: KvCacheKind,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub capacity_tokens: usize,
}

impl LayerKvPlan {
    pub fn values_per_token(self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn bytes(self) -> usize {
        self.capacity_tokens * self.values_per_token() * 2 * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCachePlan {
    pub context_tokens: usize,
    pub layers: Vec<LayerKvPlan>,
    pub total_slots: usize,
    pub bytes: usize,
}

impl KvCachePlan {
    pub fn gpt_oss_20b(context_tokens: usize) -> Result<Self> {
        if context_tokens == 0 {
            return Err(eyre!("KV-cache context capacity must be greater than zero"));
        }

        let mut layers = Vec::with_capacity(GPT_OSS_20B_LAYERS);
        let mut total_slots = 0usize;
        let mut bytes = 0usize;
        for layer_index in 0..GPT_OSS_20B_LAYERS {
            let kind = if layer_index % 2 == 0 {
                KvCacheKind::WindowRing {
                    window_tokens: GPT_OSS_WINDOW_TOKENS,
                }
            } else {
                KvCacheKind::DenseFull
            };
            let capacity_tokens = kind.capacity_tokens(context_tokens);
            let layer = LayerKvPlan {
                layer_index,
                kind,
                kv_heads: GPT_OSS_KV_HEADS,
                head_dim: GPT_OSS_HEAD_DIM,
                capacity_tokens,
            };
            total_slots += capacity_tokens;
            bytes += layer.bytes();
            layers.push(layer);
        }

        Ok(Self {
            context_tokens,
            layers,
            total_slots,
            bytes,
        })
    }

    pub fn layer(&self, layer_index: usize) -> Result<LayerKvPlan> {
        self.layers
            .get(layer_index)
            .copied()
            .ok_or_else(|| eyre!("KV-cache plan has no layer {layer_index}"))
    }
}

#[derive(Debug, Clone)]
pub struct PlannedKvCache {
    pub plan: KvCachePlan,
    layers: Vec<LayerKvCache>,
}

impl PlannedKvCache {
    pub fn new(plan: KvCachePlan) -> Self {
        let layers = plan.layers.iter().copied().map(LayerKvCache::new).collect();
        Self { plan, layers }
    }

    pub fn layer(&self, layer_index: usize) -> Result<&LayerKvCache> {
        self.layers
            .get(layer_index)
            .ok_or_else(|| eyre!("KV-cache has no layer {layer_index}"))
    }

    pub fn layer_mut(&mut self, layer_index: usize) -> Result<&mut LayerKvCache> {
        self.layers
            .get_mut(layer_index)
            .ok_or_else(|| eyre!("KV-cache has no layer {layer_index}"))
    }
}

#[derive(Debug, Clone)]
pub struct LayerKvCache {
    pub plan: LayerKvPlan,
    k: Vec<f32>,
    v: Vec<f32>,
    start_position: usize,
    len: usize,
    next_position: Option<usize>,
}

impl LayerKvCache {
    pub fn new(plan: LayerKvPlan) -> Self {
        let values = plan.capacity_tokens * plan.values_per_token();
        Self {
            plan,
            k: vec![0.0; values],
            v: vec![0.0; values],
            start_position: 0,
            len: 0,
            next_position: None,
        }
    }

    pub fn push(&mut self, position: usize, k: &[f32], v: &[f32]) -> Result<()> {
        if self.plan.capacity_tokens == 0 {
            return Err(eyre!(
                "KV-cache layer {} has zero token capacity",
                self.plan.layer_index
            ));
        }

        let values_per_token = self.plan.values_per_token();
        if k.len() != values_per_token {
            return Err(eyre!(
                "KV-cache layer {} K row has {} values, expected {values_per_token}",
                self.plan.layer_index,
                k.len()
            ));
        }
        if v.len() != values_per_token {
            return Err(eyre!(
                "KV-cache layer {} V row has {} values, expected {values_per_token}",
                self.plan.layer_index,
                v.len()
            ));
        }

        match self.next_position {
            Some(next_position) if position != next_position => {
                return Err(eyre!(
                    "KV-cache layer {} expected position {next_position}, got {position}",
                    self.plan.layer_index
                ));
            }
            Some(_) => {}
            None => {
                self.start_position = position;
            }
        }

        let slot = match self.plan.kind {
            KvCacheKind::DenseFull => {
                let slot = position - self.start_position;
                if slot >= self.plan.capacity_tokens {
                    return Err(eyre!(
                        "KV-cache dense layer {} capacity {} exhausted at position {position}",
                        self.plan.layer_index,
                        self.plan.capacity_tokens
                    ));
                }
                self.len += 1;
                slot
            }
            KvCacheKind::WindowRing { .. } => {
                let slot = position % self.plan.capacity_tokens;
                if self.len < self.plan.capacity_tokens {
                    self.len += 1;
                } else {
                    self.start_position = position + 1 - self.plan.capacity_tokens;
                }
                slot
            }
        };

        let offset = slot * values_per_token;
        self.k[offset..offset + values_per_token].copy_from_slice(k);
        self.v[offset..offset + values_per_token].copy_from_slice(v);
        self.next_position = Some(position + 1);
        Ok(())
    }

    pub fn contiguous_view_for_query(&self, query_position: usize) -> Result<KvCacheView> {
        let Some(next_position) = self.next_position else {
            return Err(eyre!("KV-cache layer {} is empty", self.plan.layer_index));
        };
        if query_position < self.start_position || query_position >= next_position {
            return Err(eyre!(
                "KV-cache layer {} cannot serve query position {query_position}; cached span is {}..{}",
                self.plan.layer_index,
                self.start_position,
                next_position
            ));
        }

        let mut start_position = self.start_position;
        if let KvCacheKind::WindowRing { window_tokens } = self.plan.kind {
            start_position = start_position.max((query_position + 1).saturating_sub(window_tokens));
        }
        let tokens = query_position + 1 - start_position;
        if tokens > self.len {
            return Err(eyre!(
                "KV-cache layer {} requested {tokens} tokens, but only {} are cached",
                self.plan.layer_index,
                self.len
            ));
        }

        let values_per_token = self.plan.values_per_token();
        let mut k = Vec::with_capacity(tokens * values_per_token);
        let mut v = Vec::with_capacity(tokens * values_per_token);
        for position in start_position..=query_position {
            let slot = self.slot_for_position(position)?;
            let offset = slot * values_per_token;
            k.extend_from_slice(&self.k[offset..offset + values_per_token]);
            v.extend_from_slice(&self.v[offset..offset + values_per_token]);
        }

        Ok(KvCacheView {
            start_position,
            tokens,
            k,
            v,
        })
    }

    fn slot_for_position(&self, position: usize) -> Result<usize> {
        if self.plan.capacity_tokens == 0 {
            return Err(eyre!(
                "KV-cache layer {} has zero token capacity",
                self.plan.layer_index
            ));
        }
        if position < self.start_position {
            return Err(eyre!(
                "KV-cache layer {} no longer contains position {position}",
                self.plan.layer_index
            ));
        }

        let slot = match self.plan.kind {
            KvCacheKind::DenseFull => position - self.start_position,
            KvCacheKind::WindowRing { .. } => position % self.plan.capacity_tokens,
        };
        if slot >= self.plan.capacity_tokens {
            return Err(eyre!(
                "KV-cache layer {} slot {slot} exceeds capacity {}",
                self.plan.layer_index,
                self.plan.capacity_tokens
            ));
        }
        Ok(slot)
    }
}

#[derive(Debug, Clone)]
pub struct KvCacheView {
    pub start_position: usize,
    pub tokens: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_oss_plan_alternates_window_and_dense_layers() {
        let plan = KvCachePlan::gpt_oss_20b(4096).unwrap();
        assert_eq!(plan.layers.len(), 24);
        assert_eq!(
            plan.layer(0).unwrap().kind,
            KvCacheKind::WindowRing { window_tokens: 128 }
        );
        assert_eq!(plan.layer(0).unwrap().capacity_tokens, 128);
        assert_eq!(plan.layer(1).unwrap().kind, KvCacheKind::DenseFull);
        assert_eq!(plan.layer(1).unwrap().capacity_tokens, 4096);
        assert_eq!(plan.total_slots, 12 * 128 + 12 * 4096);
        assert_eq!(plan.bytes, plan.total_slots * 8 * 64 * 2 * 4);
    }

    #[test]
    fn ring_cache_materializes_the_visible_span_in_position_order() {
        let plan = LayerKvPlan {
            layer_index: 0,
            kind: KvCacheKind::WindowRing { window_tokens: 3 },
            kv_heads: 1,
            head_dim: 2,
            capacity_tokens: 3,
        };
        let mut cache = LayerKvCache::new(plan);
        for position in 0..5 {
            cache
                .push(
                    position,
                    &[position as f32, position as f32 + 0.5],
                    &[-(position as f32), -(position as f32) - 0.5],
                )
                .unwrap();
        }

        let view = cache.contiguous_view_for_query(4).unwrap();
        assert_eq!(view.start_position, 2);
        assert_eq!(view.tokens, 3);
        assert_eq!(view.k, vec![2.0, 2.5, 3.0, 3.5, 4.0, 4.5]);
        assert_eq!(view.v, vec![-2.0, -2.5, -3.0, -3.5, -4.0, -4.5]);
    }

    #[test]
    fn dense_cache_reports_capacity_exhaustion() {
        let plan = LayerKvPlan {
            layer_index: 1,
            kind: KvCacheKind::DenseFull,
            kv_heads: 1,
            head_dim: 1,
            capacity_tokens: 2,
        };
        let mut cache = LayerKvCache::new(plan);
        cache.push(0, &[1.0], &[2.0]).unwrap();
        cache.push(1, &[3.0], &[4.0]).unwrap();
        assert!(cache.push(2, &[5.0], &[6.0]).is_err());
    }
}
