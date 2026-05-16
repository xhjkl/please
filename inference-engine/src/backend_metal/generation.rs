use eyre::{Result, eyre};

use super::{
    ATTN_VALUES, GpuStage, HIDDEN_SIZE, KV_HEADS, KV_VALUES, LAYERS, LM_HEAD_TOP1_BLOCK_SIZE,
    MAX_RESIDENT_CONTEXT_TOKENS, MetalRuntime, Q_HEADS, StageMarker, TokenStage, decode_token_text,
    decode_tokens_text, metal_sampler_description, platform, stage_marker,
    weights::{GptOssLayerWeights, GptOssWeights},
};
#[cfg(feature = "profile")]
use super::{MetalProfileReport, ProfileDelta};
use crate::gptoss_spec::weights as spec_weights;
use crate::harmony_adapter::HarmonyAdapter;
use crate::model_store;
use crate::runtime_core::sampler::{SampleCandidate, Sampler};
use crate::runtime_core::{
    EngineRequest, GenerationEvent, GenerationReport, GreedyTokenReport, RuntimeNotice,
    SamplingConfig, StopReason,
};
#[cfg(feature = "profile")]
use std::mem::size_of;
use std::sync::Mutex;
#[cfg(feature = "profile")]
use std::time::{Duration, Instant};

const PREFILL_MOE_CHUNK_TOKENS: usize = 16;
const PREFILL_SUFFIX_DECODE_THRESHOLD: usize = 4;

#[derive(Clone, Copy)]
enum PingPong {
    Ping,
    Pong,
}

impl PingPong {
    fn next(self) -> Self {
        match self {
            Self::Ping => Self::Pong,
            Self::Pong => Self::Ping,
        }
    }

    fn is_ping(self) -> bool {
        matches!(self, Self::Ping)
    }
}

struct GenerationScratch {
    hidden_ping: platform::F32VectorBuffer,
    hidden_pong: platform::F32VectorBuffer,
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
    prefill_hidden_ping: platform::F32VectorBuffer,
    prefill_hidden_pong: platform::F32VectorBuffer,
    prefill_normed: platform::F32VectorBuffer,
    prefill_q: platform::F32VectorBuffer,
    prefill_q_rope: platform::F32VectorBuffer,
    prefill_k: platform::F32VectorBuffer,
    prefill_k_rope: platform::F32VectorBuffer,
    prefill_v: platform::F32VectorBuffer,
    prefill_attn: platform::F32VectorBuffer,
    prefill_projected: platform::F32VectorBuffer,
    prefill_residual: platform::F32VectorBuffer,
    prefill_router_input: platform::F32VectorBuffer,
    prefill_router_logits: platform::F32VectorBuffer,
    prefill_router_indices: platform::U32Buffer,
    prefill_router_selected_logits: platform::F32VectorBuffer,
    prefill_router_weights: platform::F32VectorBuffer,
    prefill_expert_acts_packed: platform::F32VectorBuffer,
    prefill_tokens: platform::U32Buffer,
    prefix_final_hidden: platform::F32VectorBuffer,
    final_hidden: platform::F32VectorBuffer,
    lm_logits: platform::F32VectorBuffer,
    lm_top1_block_indices: platform::U32Buffer,
    lm_top1_block_values: platform::F32VectorBuffer,
    lm_top_indices: platform::U32Buffer,
    lm_top_values: platform::F32VectorBuffer,
}

impl GenerationScratch {
    fn new(platform: &platform::MetalContext, vocab: usize, context_tokens: usize) -> Result<Self> {
        let lm_top1_blocks = vocab.div_ceil(LM_HEAD_TOP1_BLOCK_SIZE);
        let prefill_values = context_tokens
            .checked_mul(HIDDEN_SIZE)
            .ok_or_else(|| eyre!("prefill hidden buffer length overflow"))?;
        let prefill_attn_values = context_tokens
            .checked_mul(ATTN_VALUES)
            .ok_or_else(|| eyre!("prefill attention buffer length overflow"))?;
        let prefill_kv_values = context_tokens
            .checked_mul(KV_VALUES)
            .ok_or_else(|| eyre!("prefill KV buffer length overflow"))?;
        let prefill_router_values = context_tokens
            .checked_mul(32)
            .ok_or_else(|| eyre!("prefill router buffer length overflow"))?;
        let prefill_router_choice_values = context_tokens
            .checked_mul(4)
            .ok_or_else(|| eyre!("prefill router choice buffer length overflow"))?;
        let prefill_expert_act_values = PREFILL_MOE_CHUNK_TOKENS
            .checked_mul(4)
            .and_then(|values| values.checked_mul(HIDDEN_SIZE))
            .ok_or_else(|| eyre!("prefill expert activation buffer length overflow"))?;
        Ok(Self {
            hidden_ping: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            hidden_pong: platform.alloc_f32_vector(HIDDEN_SIZE)?,
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
            prefill_hidden_ping: platform.alloc_f32_vector(prefill_values)?,
            prefill_hidden_pong: platform.alloc_f32_vector(prefill_values)?,
            prefill_normed: platform.alloc_f32_vector(prefill_values)?,
            prefill_q: platform.alloc_f32_vector(prefill_attn_values)?,
            prefill_q_rope: platform.alloc_f32_vector(prefill_attn_values)?,
            prefill_k: platform.alloc_f32_vector(prefill_kv_values)?,
            prefill_k_rope: platform.alloc_f32_vector(prefill_kv_values)?,
            prefill_v: platform.alloc_f32_vector(prefill_kv_values)?,
            prefill_attn: platform.alloc_f32_vector(prefill_attn_values)?,
            prefill_projected: platform.alloc_f32_vector(prefill_values)?,
            prefill_residual: platform.alloc_f32_vector(prefill_values)?,
            prefill_router_input: platform.alloc_f32_vector(prefill_values)?,
            prefill_router_logits: platform.alloc_f32_vector(prefill_router_values)?,
            prefill_router_indices: platform.alloc_u32_buffer(prefill_router_choice_values)?,
            prefill_router_selected_logits: platform
                .alloc_f32_vector(prefill_router_choice_values)?,
            prefill_router_weights: platform.alloc_f32_vector(prefill_router_choice_values)?,
            prefill_expert_acts_packed: platform.alloc_f32_vector(prefill_expert_act_values)?,
            prefill_tokens: platform.alloc_u32_buffer(context_tokens)?,
            prefix_final_hidden: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            final_hidden: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            lm_logits: platform.alloc_f32_vector(vocab)?,
            lm_top1_block_indices: platform.alloc_u32_buffer(lm_top1_blocks)?,
            lm_top1_block_values: platform.alloc_f32_vector(lm_top1_blocks)?,
            lm_top_indices: platform.alloc_u32_buffer(8)?,
            lm_top_values: platform.alloc_f32_vector(8)?,
        })
    }
}

#[derive(Clone)]
struct GpuKvCache {
    layers: Vec<LayerKvCache>,
    capacity: usize,
}

#[derive(Clone)]
struct LayerKvCache {
    k: platform::F32VectorBuffer,
    v: platform::F32VectorBuffer,
}

impl GpuKvCache {
    fn new(platform: &platform::MetalContext, layers: usize, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(eyre!("resident KV cache needs non-zero capacity"));
        }
        if capacity > MAX_RESIDENT_CONTEXT_TOKENS {
            return Err(eyre!(
                "resident KV cache currently supports at most {MAX_RESIDENT_CONTEXT_TOKENS} positions, got {capacity}"
            ));
        }
        let mut layer_caches = Vec::with_capacity(layers);
        for _ in 0..layers {
            layer_caches.push(LayerKvCache {
                k: platform.alloc_f32_vector(capacity * KV_VALUES)?,
                v: platform.alloc_f32_vector(capacity * KV_VALUES)?,
            });
        }
        Ok(Self {
            layers: layer_caches,
            capacity,
        })
    }

    fn layer(&self, layer: usize) -> Result<&LayerKvCache> {
        self.layers
            .get(layer)
            .ok_or_else(|| eyre!("resident KV cache has no layer {layer}"))
    }
}

// Prefix K/V is stored in the session working KV cache. This metadata is
// invalidated before that buffer is reused for any different prompt shape.
struct PrefixCache {
    tokens: Vec<u32>,
    layers: usize,
    capacity: usize,
}

struct MetalSession {
    scratch: GenerationScratch,
    kv_cache: GpuKvCache,
    prefix_cache: Option<PrefixCache>,
    context_capacity: usize,
    layers: usize,
    vocab: usize,
}

impl MetalSession {
    fn new(
        platform: &platform::MetalContext,
        vocab: usize,
        layers: usize,
        context_capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            scratch: GenerationScratch::new(platform, vocab, context_capacity)?,
            kv_cache: GpuKvCache::new(platform, layers, context_capacity)?,
            prefix_cache: None,
            context_capacity,
            layers,
            vocab,
        })
    }

    fn ensure<'a>(
        session: &'a mut Option<Self>,
        platform: &platform::MetalContext,
        vocab: usize,
        layers: usize,
        context_capacity: usize,
    ) -> Result<&'a mut Self> {
        let needs_rebuild = session.as_ref().is_none_or(|session| {
            session.vocab != vocab
                || session.layers != layers
                || session.context_capacity < context_capacity
        });
        if needs_rebuild {
            *session = Some(Self::new(platform, vocab, layers, context_capacity)?);
        }

        let Some(session) = session.as_mut() else {
            return Err(eyre!("resident session allocation failed"));
        };
        Ok(session)
    }
}

pub struct MetalEngine {
    harmony: HarmonyAdapter,
    ctx: MetalRuntime,
    weights: GptOssWeights,
    layers: usize,
    session: Mutex<Option<MetalSession>>,
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
        let validation = spec_weights::validate_gpt_oss_20b_source(&report);
        if !validation.is_ok() {
            return Err(eyre!(
                "canonical gpt-oss SafeTensors layout did not validate"
            ));
        }
        let source = model_store::SafeTensorMap::open(report)?;

        let harmony = HarmonyAdapter::gpt_oss()?;
        let ctx = MetalRuntime::with_lm_head_map(&source)?;
        let weights = GptOssWeights::load(&ctx, &source, layers)?;
        Ok(Self {
            harmony,
            ctx,
            weights,
            layers,
            session: Mutex::new(None),
        })
    }

    pub fn generate(&self, request: EngineRequest) -> Result<Vec<GenerationEvent>> {
        self.generate_inner(request)
    }

    #[cfg(feature = "profile")]
    pub fn generate_profiled(
        &self,
        request: EngineRequest,
    ) -> Result<(Vec<GenerationEvent>, MetalProfileReport)> {
        self.ctx.reset_profile();
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

        let mut session = self.session.lock().unwrap();
        let generated = generate_resident(
            &self.ctx,
            &self.weights,
            &self.harmony,
            &prompt_tokens,
            self.layers,
            request.limits.max_new_tokens,
            request.prompt.context_capacity,
            request.prompt.pinned_prefix_len,
            request.sampling,
            &mut session,
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

fn generate_resident(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    harmony: &HarmonyAdapter,
    prompt_tokens: &[u32],
    layers: usize,
    max_new_tokens: usize,
    context_capacity: usize,
    pinned_prefix_len: usize,
    sampling: SamplingConfig,
    session: &mut Option<MetalSession>,
) -> Result<GenerationReport> {
    #[cfg(feature = "profile")]
    {
        ctx.profile_op("phase.generate_resident", ProfileDelta::default(), || {
            generate_resident_inner(
                ctx,
                weights,
                harmony,
                prompt_tokens,
                layers,
                max_new_tokens,
                context_capacity,
                pinned_prefix_len,
                sampling,
                session,
            )
        })
    }
    #[cfg(not(feature = "profile"))]
    {
        generate_resident_inner(
            ctx,
            weights,
            harmony,
            prompt_tokens,
            layers,
            max_new_tokens,
            context_capacity,
            pinned_prefix_len,
            sampling,
            session,
        )
    }
}

fn generate_resident_inner(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    harmony: &HarmonyAdapter,
    prompt_tokens: &[u32],
    layers: usize,
    max_new_tokens: usize,
    context_capacity: usize,
    pinned_prefix_len: usize,
    sampling: SamplingConfig,
    session: &mut Option<MetalSession>,
) -> Result<GenerationReport> {
    #[cfg(feature = "profile")]
    let infer_started = Instant::now();
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
    if pinned_prefix_len > prompt_tokens.len() {
        return Err(eyre!(
            "pinned prefix length {pinned_prefix_len} exceeds prompt length {}",
            prompt_tokens.len()
        ));
    }

    let min_context_tokens = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| eyre!("resident decode context length overflow"))?;
    let context_tokens = context_capacity.max(min_context_tokens);
    if context_tokens > MAX_RESIDENT_CONTEXT_TOKENS {
        return Err(eyre!(
            "resident decode currently supports at most {MAX_RESIDENT_CONTEXT_TOKENS} context tokens, got {context_tokens}"
        ));
    }
    #[cfg(feature = "profile")]
    ctx.reset_stage_profile(context_tokens);

    let session = MetalSession::ensure(
        session,
        &ctx.platform,
        weights.lm_head.rows(),
        layers,
        context_tokens,
    )?;
    let current = prepare_prompt_state(
        ctx,
        weights,
        session,
        layers,
        prompt_tokens,
        context_tokens,
        pinned_prefix_len,
    )?;

    let stop_tokens = harmony.stop_tokens()?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut stop_reason = StopReason::MaxGeneratedTokens;
    let mut sampler = Sampler::new(sampling.clone());

    let first_output_position = prompt_tokens.len();
    let hidden = current_hidden(&session.scratch, current);
    let mut sampled = sample_from_hidden(
        ctx,
        harmony,
        weights,
        &mut session.scratch,
        &hidden,
        &mut sampler,
        first_output_position,
    )?;
    generated.push(sampled.clone());
    #[cfg(feature = "profile")]
    {
        ctx.record_profile(
            "metric.cold_start_to_first_token",
            ProfileDelta {
                wall: infer_started.elapsed(),
                ..ProfileDelta::default()
            },
        );
    }

    let mut decode_position = first_output_position;
    while generated.len() < max_new_tokens {
        if stop_tokens.contains(&sampled.token) {
            stop_reason = StopReason::EndOfGeneration;
            break;
        }

        #[cfg(feature = "profile")]
        let started = Instant::now();
        let scored = decode_and_score_next_token(
            ctx,
            harmony,
            weights,
            &mut session.scratch,
            &mut session.kv_cache,
            layers,
            decode_position,
            sampled.token,
            &mut sampler,
        )?;
        #[cfg(feature = "profile")]
        {
            let wall = started.elapsed();
            record_hot_token_metric(ctx, wall, scored.gpu_ns);
        }

        sampled = scored.token;
        generated.push(sampled.clone());
        decode_position += 1;
    }

    if stop_tokens.contains(&sampled.token) {
        stop_reason = StopReason::EndOfGeneration;
    }

    let token_ids = generated
        .iter()
        .map(|token| token.token)
        .collect::<Vec<_>>();
    let text = decode_tokens_text(harmony, &token_ids)?;
    Ok(GenerationReport {
        name: format!(
            "generate_resident.layers{layers}.prompt{}.new{}",
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

fn hidden_pair(
    scratch: &GenerationScratch,
    current: PingPong,
) -> (platform::F32VectorBuffer, platform::F32VectorBuffer) {
    if current.is_ping() {
        (scratch.hidden_ping.clone(), scratch.hidden_pong.clone())
    } else {
        (scratch.hidden_pong.clone(), scratch.hidden_ping.clone())
    }
}

fn prefill_hidden_pair(
    scratch: &GenerationScratch,
    current: PingPong,
) -> (platform::F32VectorBuffer, platform::F32VectorBuffer) {
    if current.is_ping() {
        (
            scratch.prefill_hidden_ping.clone(),
            scratch.prefill_hidden_pong.clone(),
        )
    } else {
        (
            scratch.prefill_hidden_pong.clone(),
            scratch.prefill_hidden_ping.clone(),
        )
    }
}

fn current_hidden(scratch: &GenerationScratch, current: PingPong) -> platform::F32VectorBuffer {
    if current.is_ping() {
        scratch.hidden_ping.clone()
    } else {
        scratch.hidden_pong.clone()
    }
}

struct ScoredToken {
    token: GreedyTokenReport,
    #[cfg(feature = "profile")]
    gpu_ns: u128,
}

#[allow(clippy::too_many_arguments)]
fn prepare_prompt_state(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    session: &mut MetalSession,
    layers: usize,
    prompt_tokens: &[u32],
    context_tokens: usize,
    pinned_prefix_len: usize,
) -> Result<PingPong> {
    if pinned_prefix_len == 0 {
        session.prefix_cache = None;
        prefill_embeddings(ctx, weights, &session.scratch, prompt_tokens)?;
        let current = prefill_layers(
            ctx,
            weights,
            &session.scratch,
            &mut session.kv_cache,
            layers,
            0,
            prompt_tokens.len(),
        )?;
        return Ok(current);
    }

    let prefix_tokens = &prompt_tokens[..pinned_prefix_len];
    let prefix_hit = session.prefix_cache.as_ref().is_some_and(|cache| {
        cache.layers == layers && cache.capacity >= context_tokens && cache.tokens == prefix_tokens
    });
    if prefix_hit {
        #[cfg(feature = "profile")]
        ctx.record_profile(
            "op.prefix_cache",
            ProfileDelta {
                cache_hits: 1,
                ..ProfileDelta::default()
            },
        );
        if prompt_tokens.len() == pinned_prefix_len {
            copy_hidden(
                ctx,
                &session.scratch.prefix_final_hidden,
                &session.scratch.hidden_ping,
            )?;
            return Ok(PingPong::Ping);
        }

        let current = prefill_suffix(
            ctx,
            weights,
            &mut session.scratch,
            &mut session.kv_cache,
            layers,
            pinned_prefix_len,
            &prompt_tokens[pinned_prefix_len..],
        )?;
        return Ok(current);
    }

    #[cfg(feature = "profile")]
    ctx.record_profile(
        "op.prefix_cache",
        ProfileDelta {
            cache_misses: 1,
            ..ProfileDelta::default()
        },
    );
    session.prefix_cache = None;
    prefill_embeddings(ctx, weights, &session.scratch, prefix_tokens)?;
    let current = prefill_layers(
        ctx,
        weights,
        &session.scratch,
        &mut session.kv_cache,
        layers,
        0,
        pinned_prefix_len,
    )?;
    let hidden = current_hidden(&session.scratch, current);
    copy_hidden(ctx, &hidden, &session.scratch.prefix_final_hidden)?;

    session.prefix_cache = Some(PrefixCache {
        tokens: prefix_tokens.to_vec(),
        layers,
        capacity: context_tokens,
    });

    if prompt_tokens.len() == pinned_prefix_len {
        return Ok(current);
    }

    let current = prefill_suffix(
        ctx,
        weights,
        &mut session.scratch,
        &mut session.kv_cache,
        layers,
        pinned_prefix_len,
        &prompt_tokens[pinned_prefix_len..],
    )?;
    Ok(current)
}

fn copy_hidden(
    ctx: &MetalRuntime,
    input: &platform::F32VectorBuffer,
    output: &platform::F32VectorBuffer,
) -> Result<()> {
    let batch = ctx.platform.begin_labeled_batch("generation.copy_hidden");
    batch.copy_f32_slot_into(input, 0, HIDDEN_SIZE, output)?;
    finish_generation_batch(ctx, batch, no_stage_marker());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_token_into_state(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    scratch: &mut GenerationScratch,
    kv_cache: &mut GpuKvCache,
    layers: usize,
    position: usize,
    token: u32,
) -> Result<PingPong> {
    let batch = ctx.platform.begin_labeled_batch("generation.decode_token");
    batch.set_stage(GpuStage::Embedding);
    batch.embedding_lookup_bf16_into(&weights.embed, token as usize, &scratch.hidden_ping)?;
    let mut current = PingPong::Ping;

    for layer in 0..layers {
        let (input, output) = hidden_pair(scratch, current);
        let layer_weights = weights.layer(layer)?;
        encode_decode_layer(
            &batch,
            scratch,
            kv_cache,
            layer_weights,
            layer,
            position,
            &input,
            &output,
        )?;
        current = current.next();
    }

    finish_generation_batch(ctx, batch, stage_marker(position, TokenStage::Token));
    Ok(current)
}

#[allow(clippy::too_many_arguments)]
fn prefill_suffix(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    scratch: &mut GenerationScratch,
    kv_cache: &mut GpuKvCache,
    layers: usize,
    start_position: usize,
    suffix_tokens: &[u32],
) -> Result<PingPong> {
    if suffix_tokens.is_empty() {
        return Err(eyre!("resident suffix prefill needs at least one token"));
    }
    let end_position = start_position
        .checked_add(suffix_tokens.len())
        .ok_or_else(|| eyre!("resident suffix prefill position overflow"))?;
    if end_position > kv_cache.capacity {
        return Err(eyre!(
            "resident suffix prefill end position {end_position} exceeds KV capacity {}",
            kv_cache.capacity
        ));
    }

    if suffix_tokens.len() <= PREFILL_SUFFIX_DECODE_THRESHOLD {
        let mut current = PingPong::Ping;
        for (offset, token) in suffix_tokens.iter().enumerate() {
            current = decode_token_into_state(
                ctx,
                weights,
                scratch,
                kv_cache,
                layers,
                start_position + offset,
                *token,
            )?;
        }
        return Ok(current);
    }

    prefill_embeddings(ctx, weights, scratch, suffix_tokens)?;
    prefill_layers(
        ctx,
        weights,
        scratch,
        kv_cache,
        layers,
        start_position,
        suffix_tokens.len(),
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_and_score_next_token(
    ctx: &MetalRuntime,
    harmony: &HarmonyAdapter,
    weights: &GptOssWeights,
    scratch: &mut GenerationScratch,
    kv_cache: &mut GpuKvCache,
    layers: usize,
    position: usize,
    token: u32,
    sampler: &mut Sampler,
) -> Result<ScoredToken> {
    let output_position = position + 1;
    let batch = ctx.platform.begin_labeled_batch("generation.hot_token");
    batch.set_stage(GpuStage::Embedding);
    batch.embedding_lookup_bf16_into(&weights.embed, token as usize, &scratch.hidden_ping)?;
    let mut current = PingPong::Ping;

    for layer in 0..layers {
        let (input, output) = hidden_pair(scratch, current);
        let layer_weights = weights.layer(layer)?;
        encode_decode_layer(
            &batch,
            scratch,
            kv_cache,
            layer_weights,
            layer,
            position,
            &input,
            &output,
        )?;
        current = current.next();
    }

    let hidden = current_hidden(scratch, current);
    batch.set_stage(GpuStage::LmHead);
    batch.rms_norm_into(&hidden, &weights.final_norm, &scratch.final_hidden)?;
    if sampler.needs_full_vocab() {
        batch.bf16_matrix_logits_into(
            &weights.lm_head,
            &scratch.final_hidden,
            &scratch.lm_logits,
        )?;
    } else {
        let k = sampler.candidate_count();
        if k == 1 {
            batch.bf16_matrix_top1_into(
                &weights.lm_head,
                &scratch.final_hidden,
                &scratch.lm_logits,
                &scratch.lm_top1_block_indices,
                &scratch.lm_top1_block_values,
                &scratch.lm_top_indices,
                &scratch.lm_top_values,
            )?;
        } else {
            batch.bf16_matrix_topk_into(
                &weights.lm_head,
                &scratch.final_hidden,
                &scratch.lm_logits,
                &scratch.lm_top_indices,
                &scratch.lm_top_values,
                k,
            )?;
        }
    }
    #[cfg(feature = "profile")]
    let gpu_ns = finish_generation_batch(
        ctx,
        batch,
        stage_marker(output_position, TokenStage::HotToken),
    );
    #[cfg(not(feature = "profile"))]
    finish_generation_batch(
        ctx,
        batch,
        stage_marker(output_position, TokenStage::HotToken),
    );

    let token = if sampler.needs_full_vocab() {
        #[cfg(feature = "profile")]
        ctx.record_profile(
            "op.lm_head.full_vocab_readback",
            ProfileDelta {
                readback_bytes: weights.lm_head.rows() * size_of::<f32>(),
                ..ProfileDelta::default()
            },
        );
        let logits = ctx.platform.read_f32_vector(&scratch.lm_logits);
        let sampled = sampler.choose_from_logits(&logits)?;
        GreedyTokenReport {
            token: sampled.token,
            logit: sampled.logit,
            text: decode_token_text(harmony, sampled.token)?,
        }
    } else {
        sample_from_topk_buffers(ctx, harmony, scratch, sampler.candidate_count(), sampler)?
    };

    Ok(ScoredToken {
        token,
        #[cfg(feature = "profile")]
        gpu_ns,
    })
}

fn prefill_embeddings(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    scratch: &GenerationScratch,
    prompt_tokens: &[u32],
) -> Result<()> {
    let vocab = weights.embed.rows();
    for token in prompt_tokens {
        let token = *token as usize;
        if token >= vocab {
            return Err(eyre!(
                "prompt token {token} exceeds embedding vocabulary rows {vocab}"
            ));
        }
    }

    ctx.platform
        .write_u32_buffer(&scratch.prefill_tokens, prompt_tokens)?;
    #[cfg(feature = "profile")]
    ctx.record_profile(
        "op.input.prompt_tokens",
        ProfileDelta {
            upload_bytes: prompt_tokens.len() * size_of::<u32>(),
            ..ProfileDelta::default()
        },
    );
    let batch = ctx
        .platform
        .begin_labeled_batch("generation.prefill_embeddings");
    batch.set_stage(GpuStage::Embedding);
    batch.embedding_lookup_bf16_batch_into(
        &weights.embed,
        &scratch.prefill_tokens,
        prompt_tokens.len(),
        &scratch.prefill_hidden_ping,
    )?;
    finish_generation_batch(ctx, batch, no_stage_marker());
    Ok(())
}

fn prefill_layers(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    scratch: &GenerationScratch,
    kv_cache: &mut GpuKvCache,
    layers: usize,
    start_position: usize,
    token_count: usize,
) -> Result<PingPong> {
    let mut current = PingPong::Ping;

    for layer in 0..layers {
        let (input, output) = prefill_hidden_pair(scratch, current);
        let layer_weights = weights.layer(layer)?;
        prefill_layer(
            ctx,
            scratch,
            kv_cache,
            layer_weights,
            layer,
            start_position,
            token_count,
            &input,
            &output,
        )?;
        current = current.next();
    }

    let final_prompt_hidden = if current.is_ping() {
        scratch.prefill_hidden_ping.clone()
    } else {
        scratch.prefill_hidden_pong.clone()
    };
    let batch = ctx
        .platform
        .begin_labeled_batch("generation.prefill_final_hidden");
    batch.copy_f32_slot_into(
        &final_prompt_hidden,
        token_count - 1,
        HIDDEN_SIZE,
        &scratch.hidden_ping,
    )?;
    finish_generation_batch(ctx, batch, no_stage_marker());
    Ok(PingPong::Ping)
}

fn prefill_layer(
    ctx: &MetalRuntime,
    scratch: &GenerationScratch,
    kv_cache: &mut GpuKvCache,
    weights: &GptOssLayerWeights,
    layer: usize,
    start_position: usize,
    token_count: usize,
    input: &platform::F32VectorBuffer,
    output: &platform::F32VectorBuffer,
) -> Result<()> {
    let end_position = start_position
        .checked_add(token_count)
        .ok_or_else(|| eyre!("resident prefill position overflow"))?;
    if end_position > kv_cache.capacity {
        return Err(eyre!(
            "resident prefill end position {end_position} exceeds KV capacity {}",
            kv_cache.capacity
        ));
    }

    let layer_cache = kv_cache.layer(layer)?;
    let batch = ctx.platform.begin_labeled_batch("generation.prefill_layer");
    batch.set_stage(GpuStage::InputNormQkv);
    batch.rms_norm_batch_into(
        input,
        &weights.input_norm,
        &scratch.prefill_normed,
        token_count,
        HIDDEN_SIZE,
    )?;
    batch.bf16_matrix_matvec_batch_into(
        &weights.attn.q.weight,
        &scratch.prefill_normed,
        &weights.attn.q.bias,
        &scratch.prefill_q,
        token_count,
    )?;
    batch.bf16_matrix_matvec_batch_into(
        &weights.attn.k.weight,
        &scratch.prefill_normed,
        &weights.attn.k.bias,
        &scratch.prefill_k,
        token_count,
    )?;
    batch.bf16_matrix_matvec_batch_into(
        &weights.attn.v.weight,
        &scratch.prefill_normed,
        &weights.attn.v.bias,
        &scratch.prefill_v,
        token_count,
    )?;
    batch.set_stage(GpuStage::RopeKvWrite);
    batch.rope_batch_into(
        &scratch.prefill_q,
        &scratch.prefill_q_rope,
        Q_HEADS,
        start_position,
        token_count,
    )?;
    batch.rope_batch_into(
        &scratch.prefill_k,
        &scratch.prefill_k_rope,
        KV_HEADS,
        start_position,
        token_count,
    )?;
    batch.write_f32_slots_batch_into(
        &scratch.prefill_k_rope,
        &layer_cache.k,
        start_position,
        token_count,
        KV_VALUES,
    )?;
    batch.write_f32_slots_batch_into(
        &scratch.prefill_v,
        &layer_cache.v,
        start_position,
        token_count,
        KV_VALUES,
    )?;
    batch.set_stage(GpuStage::Attention);
    if start_position == 0 {
        batch.sequence_attention_into(
            layer,
            &scratch.prefill_q_rope,
            &scratch.prefill_k_rope,
            &scratch.prefill_v,
            &weights.attn.sinks,
            &scratch.prefill_attn,
            token_count,
        )?;
    } else {
        batch.suffix_sequence_attention_into(
            layer,
            start_position,
            token_count,
            &scratch.prefill_q_rope,
            &layer_cache.k,
            &layer_cache.v,
            &weights.attn.sinks,
            &scratch.prefill_attn,
        )?;
    }
    batch.set_stage(GpuStage::AttnProj);
    batch.bf16_matrix_matvec_batch_into(
        &weights.attn.o.weight,
        &scratch.prefill_attn,
        &weights.attn.o.bias,
        &scratch.prefill_projected,
        token_count,
    )?;
    batch.vector_add_into(input, &scratch.prefill_projected, &scratch.prefill_residual)?;
    batch.set_stage(GpuStage::RouterTop4);
    batch.rms_norm_batch_into(
        &scratch.prefill_residual,
        &weights.post_attn_norm,
        &scratch.prefill_router_input,
        token_count,
        HIDDEN_SIZE,
    )?;
    batch.bf16_matrix_matvec_batch_into(
        &weights.sparse_mlp.router.weight,
        &scratch.prefill_router_input,
        &weights.sparse_mlp.router.bias,
        &scratch.prefill_router_logits,
        token_count,
    )?;
    batch.top4_softmax_batch_into(
        &scratch.prefill_router_logits,
        &scratch.prefill_router_indices,
        &scratch.prefill_router_selected_logits,
        &scratch.prefill_router_weights,
        token_count,
        32,
    )?;

    for row_offset in (0..token_count).step_by(PREFILL_MOE_CHUNK_TOKENS) {
        let rows = (token_count - row_offset).min(PREFILL_MOE_CHUNK_TOKENS);
        batch.set_stage(GpuStage::ExpertsGate);
        batch.mxfp4_top4_gate_swiglu_batch_into(
            &weights.sparse_mlp.experts_carousel.gate_up_blocks,
            &weights.sparse_mlp.experts_carousel.gate_up_scales,
            &weights.sparse_mlp.experts_carousel.gate_up_bias,
            &scratch.prefill_router_input,
            &scratch.prefill_router_indices,
            &scratch.prefill_expert_acts_packed,
            row_offset,
            rows,
        )?;
        batch.set_stage(GpuStage::ExpertsDown);
        batch.mxfp4_top4_down_weighted_batch_into(
            &weights.sparse_mlp.experts_carousel.down_blocks,
            &weights.sparse_mlp.experts_carousel.down_scales,
            &weights.sparse_mlp.experts_carousel.down_bias,
            &scratch.prefill_expert_acts_packed,
            &scratch.prefill_router_indices,
            &scratch.prefill_router_weights,
            &scratch.prefill_residual,
            output,
            row_offset,
            rows,
        )?;
    }

    finish_generation_batch(ctx, batch, no_stage_marker());
    Ok(())
}

fn sample_from_hidden(
    ctx: &MetalRuntime,
    harmony: &HarmonyAdapter,
    weights: &GptOssWeights,
    scratch: &mut GenerationScratch,
    hidden: &platform::F32VectorBuffer,
    sampler: &mut Sampler,
    output_position: usize,
) -> Result<GreedyTokenReport> {
    if sampler.needs_full_vocab() {
        sample_full_vocab_debug(
            ctx,
            harmony,
            weights,
            scratch,
            hidden,
            sampler,
            output_position,
        )
    } else {
        let candidates = score_topk(
            ctx,
            harmony,
            weights,
            scratch,
            hidden,
            sampler.candidate_count(),
            output_position,
        )?;
        sample_from_candidates(harmony, &candidates, sampler)
    }
}

fn sample_from_topk_buffers(
    ctx: &MetalRuntime,
    harmony: &HarmonyAdapter,
    scratch: &GenerationScratch,
    k: usize,
    sampler: &mut Sampler,
) -> Result<GreedyTokenReport> {
    let indices = ctx.platform.read_u32_buffer(&scratch.lm_top_indices);
    let values = ctx.platform.read_f32_vector(&scratch.lm_top_values);
    let candidates = indices
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
        .collect::<Result<Vec<_>>>()?;
    sample_from_candidates(harmony, &candidates, sampler)
}

fn sample_from_candidates(
    harmony: &HarmonyAdapter,
    candidates: &[GreedyTokenReport],
    sampler: &mut Sampler,
) -> Result<GreedyTokenReport> {
    let sample_candidates = candidates
        .iter()
        .map(|token| SampleCandidate {
            token: token.token,
            logit: token.logit,
            probability: 0.0,
        })
        .collect::<Vec<_>>();
    let sampled = sampler.choose(&sample_candidates)?;
    let text = candidates
        .iter()
        .find(|token| token.token == sampled.token)
        .map(|token| token.text.clone())
        .unwrap_or(decode_token_text(harmony, sampled.token)?);
    Ok(GreedyTokenReport {
        token: sampled.token,
        logit: sampled.logit,
        text,
    })
}

fn score_topk(
    ctx: &MetalRuntime,
    harmony: &HarmonyAdapter,
    weights: &GptOssWeights,
    scratch: &mut GenerationScratch,
    hidden: &platform::F32VectorBuffer,
    k: usize,
    output_position: usize,
) -> Result<Vec<GreedyTokenReport>> {
    let batch = ctx.platform.begin_labeled_batch("generation.score_topk");
    batch.set_stage(GpuStage::LmHead);
    batch.rms_norm_into(hidden, &weights.final_norm, &scratch.final_hidden)?;
    if k == 1 {
        batch.bf16_matrix_top1_into(
            &weights.lm_head,
            &scratch.final_hidden,
            &scratch.lm_logits,
            &scratch.lm_top1_block_indices,
            &scratch.lm_top1_block_values,
            &scratch.lm_top_indices,
            &scratch.lm_top_values,
        )?;
    } else {
        batch.bf16_matrix_topk_into(
            &weights.lm_head,
            &scratch.final_hidden,
            &scratch.lm_logits,
            &scratch.lm_top_indices,
            &scratch.lm_top_values,
            k,
        )?;
    }
    finish_generation_batch(
        ctx,
        batch,
        stage_marker(output_position, TokenStage::LmHead),
    );

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

fn sample_full_vocab_debug(
    ctx: &MetalRuntime,
    harmony: &HarmonyAdapter,
    weights: &GptOssWeights,
    scratch: &mut GenerationScratch,
    hidden: &platform::F32VectorBuffer,
    sampler: &mut Sampler,
    output_position: usize,
) -> Result<GreedyTokenReport> {
    let batch = ctx
        .platform
        .begin_labeled_batch("generation.full_vocab_debug");
    batch.set_stage(GpuStage::LmHead);
    batch.rms_norm_into(hidden, &weights.final_norm, &scratch.final_hidden)?;
    batch.bf16_matrix_logits_into(&weights.lm_head, &scratch.final_hidden, &scratch.lm_logits)?;
    finish_generation_batch(
        ctx,
        batch,
        stage_marker(output_position, TokenStage::LmHead),
    );

    #[cfg(feature = "profile")]
    ctx.record_profile(
        "op.lm_head.full_vocab_readback",
        ProfileDelta {
            readback_bytes: weights.lm_head.rows() * size_of::<f32>(),
            ..ProfileDelta::default()
        },
    );
    let logits = ctx.platform.read_f32_vector(&scratch.lm_logits);
    let sampled = sampler.choose_from_logits(&logits)?;
    Ok(GreedyTokenReport {
        token: sampled.token,
        logit: sampled.logit,
        text: decode_token_text(harmony, sampled.token)?,
    })
}

fn encode_decode_layer(
    batch: &platform::MetalBatch<'_>,
    scratch: &mut GenerationScratch,
    kv_cache: &mut GpuKvCache,
    weights: &GptOssLayerWeights,
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

    let layer_cache = kv_cache.layer(layer)?;
    batch.set_stage(GpuStage::InputNormQkv);
    batch.rms_norm_into(input, &weights.input_norm, &scratch.normed)?;
    batch.bf16_qkv_matvec_into(
        &weights.attn.q.weight,
        &weights.attn.k.weight,
        &weights.attn.v.weight,
        &scratch.normed,
        &weights.attn.q.bias,
        &weights.attn.k.bias,
        &weights.attn.v.bias,
        &scratch.q,
        &scratch.k,
        &scratch.v,
    )?;
    batch.set_stage(GpuStage::RopeKvWrite);
    batch.rope_row_into(&scratch.q, &scratch.q_rope, Q_HEADS, position)?;
    batch.rope_row_into(&scratch.k, &scratch.k_rope, KV_HEADS, position)?;
    batch.write_f32_slot_into(&scratch.k_rope, &layer_cache.k, position, KV_VALUES)?;
    batch.write_f32_slot_into(&scratch.v, &layer_cache.v, position, KV_VALUES)?;
    batch.set_stage(GpuStage::Attention);
    batch.kv_cache_decode_attention_into(
        layer,
        position,
        0,
        position + 1,
        &scratch.q_rope,
        &layer_cache.k,
        &layer_cache.v,
        &weights.attn.sinks,
        &scratch.attn,
    )?;
    batch.set_stage(GpuStage::AttnProj);
    batch.bf16_matrix_matvec_into(
        &weights.attn.o.weight,
        &scratch.attn,
        &weights.attn.o.bias,
        &scratch.projected,
    )?;
    batch.vector_add_into(input, &scratch.projected, &scratch.residual)?;
    batch.set_stage(GpuStage::RouterTop4);
    batch.rms_norm_into(
        &scratch.residual,
        &weights.post_attn_norm,
        &scratch.router_input,
    )?;
    batch.bf16_matrix_matvec_into(
        &weights.sparse_mlp.router.weight,
        &scratch.router_input,
        &weights.sparse_mlp.router.bias,
        &scratch.router_logits,
    )?;
    batch.top4_softmax_into(
        &scratch.router_logits,
        &scratch.router_indices,
        &scratch.router_selected_logits,
        &scratch.router_weights,
    )?;

    batch.set_stage(GpuStage::ExpertsGate);
    batch.mxfp4_top4_gate_swiglu_into(
        &weights.sparse_mlp.experts_carousel.gate_up_blocks,
        &weights.sparse_mlp.experts_carousel.gate_up_scales,
        &weights.sparse_mlp.experts_carousel.gate_up_bias,
        &scratch.router_input,
        &scratch.router_indices,
        &scratch.expert_acts_packed,
    )?;
    batch.set_stage(GpuStage::ExpertsDown);
    batch.mxfp4_top4_down_weighted_into(
        &weights.sparse_mlp.experts_carousel.down_blocks,
        &weights.sparse_mlp.experts_carousel.down_scales,
        &weights.sparse_mlp.experts_carousel.down_bias,
        &scratch.expert_acts_packed,
        &scratch.router_indices,
        &scratch.router_weights,
        &scratch.residual,
        output,
    )?;

    Ok(())
}

fn finish_generation_batch(
    ctx: &MetalRuntime,
    batch: platform::MetalBatch<'_>,
    stage: StageMarker,
) -> u128 {
    let timing = batch.finish();
    let gpu_ns = timing.gpu_ns;
    #[cfg(not(feature = "profile"))]
    {
        let _ = ctx;
        let _ = stage;
    }
    #[cfg(feature = "profile")]
    ctx.record_profile(
        "phase.generate_resident",
        ProfileDelta {
            command_buffers: 1,
            ..ProfileDelta::default()
        },
    );
    #[cfg(feature = "profile")]
    if let Some((token_position, stage)) = stage {
        ctx.record_token_stage(token_position, stage, gpu_ns);
        ctx.record_gpu_stages(token_position, timing.gpu_stages);
    }
    gpu_ns
}

fn no_stage_marker() -> StageMarker {
    #[cfg(feature = "profile")]
    {
        None
    }
    #[cfg(not(feature = "profile"))]
    {}
}

#[cfg(feature = "profile")]
fn record_hot_token_metric(ctx: &MetalRuntime, wall: Duration, gpu_ns: u128) {
    ctx.record_profile(
        "metric.hot_token",
        ProfileDelta {
            wall,
            gpu_ns,
            ..ProfileDelta::default()
        },
    );
    let gap_ns = wall.as_nanos().saturating_sub(gpu_ns);
    ctx.record_profile(
        "metric.hot_token_gap",
        ProfileDelta {
            wall: duration_from_ns(gap_ns),
            ..ProfileDelta::default()
        },
    );
}

#[cfg(feature = "profile")]
fn duration_from_ns(ns: u128) -> Duration {
    Duration::from_nanos(ns.min(u64::MAX as u128) as u64)
}
