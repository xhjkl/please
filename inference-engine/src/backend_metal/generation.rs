use eyre::{Result, eyre};

use super::{
    ATTN_VALUES, HIDDEN_SIZE, KV_HEADS, KV_VALUES, LAYERS, LM_HEAD_TOP1_BLOCK_SIZE,
    MAX_KV_CACHE_PROBE_TOKENS, MetalOracleContext, MetalProfileReport, ProfileDelta, Q_HEADS,
    StageMarker, TokenStage, decode_token_text, decode_tokens_text, metal_sampler_description,
    platform, stage_marker,
};
use crate::gptoss_spec::weights;
use crate::harmony_adapter::HarmonyAdapter;
use crate::model_store::{self, SourceModelReport};
use crate::runtime_core::sampler::{SampleCandidate, Sampler};
use crate::runtime_core::{
    EngineRequest, GenerationEvent, GreedyDecodeProbeReport, GreedyTokenReport, RuntimeNotice,
    SamplingConfig, StopReason,
};
use std::time::Instant;

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

    finish_resident_batch(ctx, batch, stage_marker(position, TokenStage::Token));
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
