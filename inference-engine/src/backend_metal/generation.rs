use eyre::{Result, eyre};

use super::{
    ATTN_VALUES, HIDDEN_SIZE, KV_HEADS, KV_VALUES, LAYERS, LM_HEAD_TOP1_BLOCK_SIZE,
    MAX_RESIDENT_CONTEXT_TOKENS, MetalOracleContext, MetalProfileReport, ProfileDelta, Q_HEADS,
    StageMarker, TokenStage, decode_token_text, decode_tokens_text, metal_sampler_description,
    platform, stage_marker,
    weights::{GptOssLayerWeights, GptOssWeights},
};
use crate::gptoss_spec::weights as spec_weights;
use crate::harmony_adapter::HarmonyAdapter;
use crate::model_store;
use crate::runtime_core::sampler::{SampleCandidate, Sampler};
use crate::runtime_core::{
    EngineRequest, GenerationEvent, GreedyDecodeProbeReport, GreedyTokenReport, RuntimeNotice,
    SamplingConfig, StopReason,
};
use std::mem::size_of;
use std::time::Instant;

const PREFILL_MOE_CHUNK_TOKENS: usize = 16;

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
    prefill_hidden: platform::F32VectorBuffer,
    prefill_hidden_b: platform::F32VectorBuffer,
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
    final_hidden: platform::F32VectorBuffer,
    lm_logits: platform::F32VectorBuffer,
    lm_top1_block_indices: platform::U32Buffer,
    lm_top1_block_values: platform::F32VectorBuffer,
    lm_top_indices: platform::U32Buffer,
    lm_top_values: platform::F32VectorBuffer,
}

impl ResidentDecodeScratch {
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
            prefill_hidden: platform.alloc_f32_vector(prefill_values)?,
            prefill_hidden_b: platform.alloc_f32_vector(prefill_values)?,
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
        if capacity > MAX_RESIDENT_CONTEXT_TOKENS {
            return Err(eyre!(
                "resident KV cache currently supports at most {MAX_RESIDENT_CONTEXT_TOKENS} positions, got {capacity}"
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

pub struct MetalEngine {
    harmony: HarmonyAdapter,
    ctx: MetalOracleContext,
    weights: GptOssWeights,
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
        let validation = spec_weights::validate_gpt_oss_20b_source(&report);
        if !validation.is_ok() {
            return Err(eyre!(
                "canonical gpt-oss SafeTensors layout did not validate"
            ));
        }
        let source = model_store::SafeTensorMap::open(report)?;

        let harmony = HarmonyAdapter::gpt_oss()?;
        let ctx = MetalOracleContext::with_lm_head_map(&source)?;
        let weights = GptOssWeights::load(&ctx, &source, layers)?;
        Ok(Self {
            harmony,
            ctx,
            weights,
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
            &self.weights,
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
    weights: &GptOssWeights,
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
                weights,
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
    weights: &GptOssWeights,
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
    if context_tokens > MAX_RESIDENT_CONTEXT_TOKENS {
        return Err(eyre!(
            "resident decode currently supports at most {MAX_RESIDENT_CONTEXT_TOKENS} context tokens, got {context_tokens}"
        ));
    }
    #[cfg(feature = "metal-stage-profile")]
    ctx.reset_stage_profile(context_tokens);

    let mut scratch =
        ResidentDecodeScratch::new(&ctx.platform, weights.lm_head.rows(), context_tokens)?;
    let mut kv_cache = ResidentGpuKvCache::new(&ctx.platform, layers, context_tokens)?;
    resident_prefill_embeddings(ctx, weights, &scratch, prompt_tokens)?;
    let mut current_is_a = resident_prefill_layers(
        ctx,
        weights,
        &scratch,
        &mut kv_cache,
        layers,
        prompt_tokens.len(),
    )?;

    let stop_tokens = harmony.stop_tokens()?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut stop_reason = StopReason::MaxGeneratedTokens;
    let mut sampler = Sampler::new(sampling.clone());

    for step in 0..max_new_tokens {
        let output_position = prompt_tokens.len() + step;
        let current_hidden = resident_current_hidden(&scratch, current_is_a);
        let candidates = resident_lm_head_topk(
            ctx,
            harmony,
            weights,
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
            weights,
            &mut scratch,
            &mut kv_cache,
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

fn resident_prefill_hidden_pair(
    scratch: &ResidentDecodeScratch,
    current_is_a: bool,
) -> (platform::F32VectorBuffer, platform::F32VectorBuffer) {
    if current_is_a {
        (
            scratch.prefill_hidden.clone(),
            scratch.prefill_hidden_b.clone(),
        )
    } else {
        (
            scratch.prefill_hidden_b.clone(),
            scratch.prefill_hidden.clone(),
        )
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
    weights: &GptOssWeights,
    scratch: &mut ResidentDecodeScratch,
    kv_cache: &mut ResidentGpuKvCache,
    layers: usize,
    position: usize,
    token: u32,
) -> Result<bool> {
    let batch = ctx.platform.begin_batch();
    batch.embedding_lookup_bf16_into(&weights.embed, token as usize, &scratch.hidden_a)?;
    let mut current_is_a = true;

    for layer in 0..layers {
        let (input, output) = resident_hidden_pair(scratch, current_is_a);
        let layer_weights = weights.layer(layer)?;
        resident_decode_layer(
            &batch,
            scratch,
            kv_cache,
            layer_weights,
            layer,
            position,
            &input,
            &output,
        )?;
        current_is_a = !current_is_a;
    }

    finish_resident_batch(ctx, batch, stage_marker(position, TokenStage::Token));
    Ok(current_is_a)
}

fn resident_prefill_embeddings(
    ctx: &MetalOracleContext,
    weights: &GptOssWeights,
    scratch: &ResidentDecodeScratch,
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

    let tokens = ctx.platform.upload_u32_buffer(prompt_tokens)?;
    ctx.record_profile(
        "op.input.prompt_tokens",
        ProfileDelta {
            upload_bytes: prompt_tokens.len() * size_of::<u32>(),
            cache_misses: 1,
            ..ProfileDelta::default()
        },
    );
    let batch = ctx.platform.begin_batch();
    batch.embedding_lookup_bf16_batch_into(
        &weights.embed,
        &tokens,
        prompt_tokens.len(),
        &scratch.prefill_hidden,
    )?;
    finish_resident_batch(ctx, batch, no_stage_marker());
    Ok(())
}

fn resident_prefill_layers(
    ctx: &MetalOracleContext,
    weights: &GptOssWeights,
    scratch: &ResidentDecodeScratch,
    kv_cache: &mut ResidentGpuKvCache,
    layers: usize,
    prompt_len: usize,
) -> Result<bool> {
    let mut current_is_a = true;

    for layer in 0..layers {
        let (input, output) = resident_prefill_hidden_pair(scratch, current_is_a);
        let layer_weights = weights.layer(layer)?;
        resident_prefill_layer(
            ctx,
            scratch,
            kv_cache,
            layer_weights,
            layer,
            prompt_len,
            &input,
            &output,
        )?;
        current_is_a = !current_is_a;
    }

    let final_prompt_hidden = if current_is_a {
        scratch.prefill_hidden.clone()
    } else {
        scratch.prefill_hidden_b.clone()
    };
    let batch = ctx.platform.begin_batch();
    batch.read_f32_slot_into(
        &final_prompt_hidden,
        prompt_len - 1,
        HIDDEN_SIZE,
        &scratch.hidden_a,
    )?;
    finish_resident_batch(ctx, batch, no_stage_marker());
    Ok(true)
}

fn resident_prefill_layer(
    ctx: &MetalOracleContext,
    scratch: &ResidentDecodeScratch,
    kv_cache: &mut ResidentGpuKvCache,
    weights: &GptOssLayerWeights,
    layer: usize,
    prompt_len: usize,
    input: &platform::F32VectorBuffer,
    output: &platform::F32VectorBuffer,
) -> Result<()> {
    if prompt_len > kv_cache.capacity {
        return Err(eyre!(
            "resident prefill prompt length {prompt_len} exceeds KV capacity {}",
            kv_cache.capacity
        ));
    }

    let layer_cache = kv_cache.layer(layer)?;
    let batch = ctx.platform.begin_batch();
    batch.rms_norm_batch_into(
        input,
        &weights.input_norm,
        &scratch.prefill_normed,
        prompt_len,
        HIDDEN_SIZE,
    )?;
    batch.bf16_matrix_matvec_batch_into(
        &weights.attn.q.weight,
        &scratch.prefill_normed,
        &weights.attn.q.bias,
        &scratch.prefill_q,
        prompt_len,
    )?;
    batch.bf16_matrix_matvec_batch_into(
        &weights.attn.k.weight,
        &scratch.prefill_normed,
        &weights.attn.k.bias,
        &scratch.prefill_k,
        prompt_len,
    )?;
    batch.bf16_matrix_matvec_batch_into(
        &weights.attn.v.weight,
        &scratch.prefill_normed,
        &weights.attn.v.bias,
        &scratch.prefill_v,
        prompt_len,
    )?;
    batch.rope_batch_into(
        &scratch.prefill_q,
        &scratch.prefill_q_rope,
        Q_HEADS,
        0,
        prompt_len,
    )?;
    batch.rope_batch_into(
        &scratch.prefill_k,
        &scratch.prefill_k_rope,
        KV_HEADS,
        0,
        prompt_len,
    )?;
    batch.write_f32_slots_batch_into(
        &scratch.prefill_k_rope,
        &layer_cache.k,
        0,
        prompt_len,
        KV_VALUES,
    )?;
    batch.write_f32_slots_batch_into(
        &scratch.prefill_v,
        &layer_cache.v,
        0,
        prompt_len,
        KV_VALUES,
    )?;
    batch.sequence_attention_into(
        layer,
        &scratch.prefill_q_rope,
        &scratch.prefill_k_rope,
        &scratch.prefill_v,
        &weights.attn.sinks,
        &scratch.prefill_attn,
        prompt_len,
    )?;
    batch.bf16_matrix_matvec_batch_into(
        &weights.attn.o.weight,
        &scratch.prefill_attn,
        &weights.attn.o.bias,
        &scratch.prefill_projected,
        prompt_len,
    )?;
    batch.vector_add_into(input, &scratch.prefill_projected, &scratch.prefill_residual)?;
    batch.rms_norm_batch_into(
        &scratch.prefill_residual,
        &weights.post_attn_norm,
        &scratch.prefill_router_input,
        prompt_len,
        HIDDEN_SIZE,
    )?;
    batch.bf16_matrix_matvec_batch_into(
        &weights.sparse_mlp.router.weight,
        &scratch.prefill_router_input,
        &weights.sparse_mlp.router.bias,
        &scratch.prefill_router_logits,
        prompt_len,
    )?;
    batch.top4_softmax_batch_into(
        &scratch.prefill_router_logits,
        &scratch.prefill_router_indices,
        &scratch.prefill_router_selected_logits,
        &scratch.prefill_router_weights,
        prompt_len,
        32,
    )?;

    for row_offset in (0..prompt_len).step_by(PREFILL_MOE_CHUNK_TOKENS) {
        let rows = (prompt_len - row_offset).min(PREFILL_MOE_CHUNK_TOKENS);
        batch.mxfp4_top4_gate_swiglu_batch_into(
            &weights.sparse_mlp.experts.gate_up_blocks,
            &weights.sparse_mlp.experts.gate_up_scales,
            &weights.sparse_mlp.experts.gate_up_bias,
            &scratch.prefill_router_input,
            &scratch.prefill_router_indices,
            &scratch.prefill_expert_acts_packed,
            row_offset,
            rows,
        )?;
        batch.mxfp4_top4_down_weighted_batch_into(
            &weights.sparse_mlp.experts.down_blocks,
            &weights.sparse_mlp.experts.down_scales,
            &weights.sparse_mlp.experts.down_bias,
            &scratch.prefill_expert_acts_packed,
            &scratch.prefill_router_indices,
            &scratch.prefill_router_weights,
            &scratch.prefill_residual,
            output,
            row_offset,
            rows,
        )?;
    }

    finish_resident_batch(ctx, batch, no_stage_marker());
    Ok(())
}

fn resident_lm_head_topk(
    ctx: &MetalOracleContext,
    harmony: &HarmonyAdapter,
    weights: &GptOssWeights,
    scratch: &mut ResidentDecodeScratch,
    hidden: &platform::F32VectorBuffer,
    k: usize,
    output_position: usize,
) -> Result<Vec<GreedyTokenReport>> {
    let batch = ctx.platform.begin_batch();
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
    finish_resident_batch(
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

fn resident_decode_layer(
    batch: &platform::MetalBatch<'_>,
    scratch: &mut ResidentDecodeScratch,
    kv_cache: &mut ResidentGpuKvCache,
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
    batch.rms_norm_into(input, &weights.input_norm, &scratch.normed)?;
    batch.bf16_matrix_matvec_into(
        &weights.attn.q.weight,
        &scratch.normed,
        &weights.attn.q.bias,
        &scratch.q,
    )?;
    batch.bf16_matrix_matvec_into(
        &weights.attn.k.weight,
        &scratch.normed,
        &weights.attn.k.bias,
        &scratch.k,
    )?;
    batch.bf16_matrix_matvec_into(
        &weights.attn.v.weight,
        &scratch.normed,
        &weights.attn.v.bias,
        &scratch.v,
    )?;
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
        &weights.attn.sinks,
        &scratch.attn,
    )?;
    batch.bf16_matrix_matvec_into(
        &weights.attn.o.weight,
        &scratch.attn,
        &weights.attn.o.bias,
        &scratch.projected,
    )?;
    batch.vector_add_into(input, &scratch.projected, &scratch.residual)?;
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

    batch.mxfp4_top4_gate_swiglu_into(
        &weights.sparse_mlp.experts.gate_up_blocks,
        &weights.sparse_mlp.experts.gate_up_scales,
        &weights.sparse_mlp.experts.gate_up_bias,
        &scratch.router_input,
        &scratch.router_indices,
        &scratch.expert_acts_packed,
    )?;
    batch.mxfp4_top4_down_weighted_into(
        &weights.sparse_mlp.experts.down_blocks,
        &weights.sparse_mlp.experts.down_scales,
        &weights.sparse_mlp.experts.down_bias,
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

fn no_stage_marker() -> StageMarker {
    #[cfg(feature = "metal-stage-profile")]
    {
        None
    }
    #[cfg(not(feature = "metal-stage-profile"))]
    {}
}
