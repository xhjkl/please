use eyre::{Result, eyre};

use crate::harmony_adapter::HarmonyAdapter;
use crate::model_store::{self, SourceModelReport};
use crate::runtime_core::{
    CpuLayer0Report, CpuOracleReport, CpuProbeReport, CpuPromptPrefillReport, CpuSingleTokenReport,
    ExpertScore, GreedyDecodeProbeReport, GreedyTokenReport, LayerCheckpoint, LogitScore,
    PromptFixture, SelectedLogit, StopReason,
};

pub const PROMPT_PREFILL_TOKEN_LIMIT: usize = 4;

const LAYERS: usize = 24;
const HIDDEN_SIZE: usize = 2880;
const HEAD_DIM: usize = 64;
const ATTN_HEADS: usize = 64;
const KV_HEADS: usize = 8;
const Q_MULT: usize = ATTN_HEADS / KV_HEADS;
const ATTN_VALUES: usize = ATTN_HEADS * HEAD_DIM;
const SLIDING_WINDOW: usize = 128;
const INITIAL_CONTEXT_LENGTH: f32 = 4096.0;
const ROPE_THETA: f32 = 150000.0;
const ROPE_SCALING_FACTOR: f32 = 32.0;
const ROPE_NTK_ALPHA: f32 = 1.0;
const ROPE_NTK_BETA: f32 = 32.0;
const EXPERTS: usize = 32;
const INTERMEDIATE_SIZE: usize = 2880;
const GATE_UP_VALUES: usize = INTERMEDIATE_SIZE * 2;
const MXFP4_GROUPS: usize = HIDDEN_SIZE / 32;
const MXFP4_BYTES_PER_GROUP: usize = 16;
const SWIGLU_LIMIT: f32 = 7.0;
const SWIGLU_ALPHA: f32 = 1.702;
const FP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub fn probe_first_prompt_embedding(
    report: &SourceModelReport,
    fixture: &PromptFixture,
) -> Result<Option<CpuProbeReport>> {
    let Some(token) = fixture.prompt_token_prefix.first().copied() else {
        return Ok(None);
    };
    let token = token as usize;

    let row = model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token)?;
    let stats = FloatStats::from_values(&row)?;
    Ok(Some(CpuProbeReport {
        first_prompt_token: token as u32,
        embedding_values: row.len(),
        embedding_min: stats.min,
        embedding_max: stats.max,
        embedding_mean: stats.mean,
        embedding_l2: stats.l2,
        embedding_sample: row.into_iter().take(8).collect(),
    }))
}

pub fn probe_layer0_math(
    report: &SourceModelReport,
    fixture: &PromptFixture,
) -> Result<Option<CpuLayer0Report>> {
    let Some(token) = fixture.prompt_token_prefix.first().copied() else {
        return Ok(None);
    };
    let token = token as usize;

    let x = model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token)?;
    expect_len("embedding", &x, HIDDEN_SIZE)?;

    let input_norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.input_layernorm.weight")?;
    let attn_input = rms_norm(&x, &input_norm_weight)?;

    let mut q = model_store::matvec_bf16(
        report,
        "model.layers.0.self_attn.q_proj.weight",
        &attn_input,
    )?;
    let q_bias = model_store::read_bf16_vector(report, "model.layers.0.self_attn.q_proj.bias")?;
    model_store::add_in_place(&mut q, &q_bias, "model.layers.0.self_attn.q_proj")?;

    let mut k = model_store::matvec_bf16(
        report,
        "model.layers.0.self_attn.k_proj.weight",
        &attn_input,
    )?;
    let k_bias = model_store::read_bf16_vector(report, "model.layers.0.self_attn.k_proj.bias")?;
    model_store::add_in_place(&mut k, &k_bias, "model.layers.0.self_attn.k_proj")?;

    let mut v = model_store::matvec_bf16(
        report,
        "model.layers.0.self_attn.v_proj.weight",
        &attn_input,
    )?;
    let v_bias = model_store::read_bf16_vector(report, "model.layers.0.self_attn.v_proj.bias")?;
    model_store::add_in_place(&mut v, &v_bias, "model.layers.0.self_attn.v_proj")?;

    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    let attn = single_token_attention(&q, &k, &v, &sinks)?;

    let mut projected =
        model_store::matvec_bf16(report, "model.layers.0.self_attn.o_proj.weight", &attn)?;
    let o_bias = model_store::read_bf16_vector(report, "model.layers.0.self_attn.o_proj.bias")?;
    model_store::add_in_place(&mut projected, &o_bias, "model.layers.0.self_attn.o_proj")?;
    expect_len("attention projection", &projected, HIDDEN_SIZE)?;

    let residual: Vec<f32> = x
        .iter()
        .copied()
        .zip(projected.iter().copied())
        .map(|(x, projected)| x + projected)
        .collect();

    let post_norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.post_attention_layernorm.weight")?;
    let router_input = rms_norm(&residual, &post_norm_weight)?;
    let mut router =
        model_store::matvec_bf16(report, "model.layers.0.mlp.router.weight", &router_input)?;
    let router_bias = model_store::read_bf16_vector(report, "model.layers.0.mlp.router.bias")?;
    model_store::add_in_place(&mut router, &router_bias, "model.layers.0.mlp.router")?;
    let top_experts = top_k_softmax(&router, 4);
    let moe = layer_moe(report, 0, &router_input, &top_experts)?;
    let layer_output: Vec<f32> = residual
        .iter()
        .copied()
        .zip(moe.iter().copied())
        .map(|(residual, moe)| residual + moe)
        .collect();

    Ok(Some(CpuLayer0Report {
        token: token as u32,
        hidden_size: x.len(),
        q_values: q.len(),
        k_values: k.len(),
        v_values: v.len(),
        attention_values: attn.len(),
        residual_values: residual.len(),
        moe_values: moe.len(),
        layer_output_values: layer_output.len(),
        q_sample: sample8(&q),
        k_sample: sample8(&k),
        v_sample: sample8(&v),
        attention_sample: sample8(&attn),
        residual_sample: sample8(&residual),
        moe_sample: sample8(&moe),
        layer_output_sample: sample8(&layer_output),
        top_experts,
    }))
}

pub fn probe_single_token_full_stack(
    report: &SourceModelReport,
    fixture: &PromptFixture,
) -> Result<Option<CpuSingleTokenReport>> {
    let Some(token) = fixture.prompt_token_prefix.first().copied() else {
        return Ok(None);
    };
    let token = token as usize;

    let mut x = model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token)?;
    expect_len("embedding", &x, HIDDEN_SIZE)?;

    for layer in 0..LAYERS {
        x = single_token_layer(report, layer, &x)?;
    }

    let norm_weight = model_store::read_bf16_vector(report, "model.norm.weight")?;
    let final_hidden = rms_norm(&x, &norm_weight)?;
    let top_logits = model_store::top_k_matvec_bf16(report, "lm_head.weight", &final_hidden, 8)?
        .into_iter()
        .map(|(token, logit)| LogitScore {
            token: token as u32,
            logit,
        })
        .collect::<Vec<_>>();

    Ok(Some(CpuSingleTokenReport {
        token: token as u32,
        layers: LAYERS,
        final_hidden_values: final_hidden.len(),
        final_hidden_sample: sample8(&final_hidden),
        top_logits,
    }))
}

pub fn probe_prompt_prefill(
    report: &SourceModelReport,
    fixture: &PromptFixture,
) -> Result<Option<CpuPromptPrefillReport>> {
    let prompt_tokens = fixture
        .prompt_tokens
        .iter()
        .copied()
        .take(fixture.prefill_token_count)
        .collect::<Vec<_>>();
    if prompt_tokens.is_empty() {
        return Ok(None);
    }

    let mut x = prompt_tokens
        .iter()
        .copied()
        .map(|token| {
            model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)
        })
        .collect::<Result<Vec<_>>>()?;
    for row in &x {
        expect_len("embedding", row, HIDDEN_SIZE)?;
    }

    for layer in 0..LAYERS {
        x = sequence_layer(report, layer, &x)?;
    }

    let norm_weight = model_store::read_bf16_vector(report, "model.norm.weight")?;
    let Some(final_hidden) = x.last() else {
        return Ok(None);
    };
    let final_hidden = rms_norm(final_hidden, &norm_weight)?;
    let top_logits = model_store::top_k_matvec_bf16(report, "lm_head.weight", &final_hidden, 8)?
        .into_iter()
        .map(|(token, logit)| LogitScore {
            token: token as u32,
            logit,
        })
        .collect::<Vec<_>>();

    Ok(Some(CpuPromptPrefillReport {
        final_position: prompt_tokens.len().saturating_sub(1),
        prompt_tokens,
        layers: LAYERS,
        final_hidden_values: final_hidden.len(),
        final_hidden_sample: sample8(&final_hidden),
        top_logits,
    }))
}

pub fn probe_prompt_prefill_oracle(
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
    logit_tokens: &[u32],
) -> Result<CpuOracleReport> {
    let mut x = tokens
        .iter()
        .copied()
        .map(|token| {
            model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)
        })
        .collect::<Result<Vec<_>>>()?;
    if x.is_empty() {
        return Err(eyre!("cpu oracle needs at least one prompt token"));
    }
    for row in &x {
        expect_len("embedding", row, HIDDEN_SIZE)?;
    }
    let embedding_final_first8 = sample8(x.last().expect("checked non-empty"));

    let mut layer_checkpoints = Vec::new();
    for layer in 0..layers {
        if layer >= LAYERS {
            return Err(eyre!(
                "layer {layer} is outside supported gpt-oss-20b depth {LAYERS}"
            ));
        }
        x = sequence_layer(report, layer, &x)?;
        let final_hidden = x
            .last()
            .expect("sequence_layer preserves non-empty sequence");
        layer_checkpoints.push(LayerCheckpoint {
            layer,
            final_l2: l2(final_hidden),
            final_mean: mean(final_hidden),
            final_first8: sample8(final_hidden),
        });
    }

    let norm_weight = model_store::read_bf16_vector(report, "model.norm.weight")?;
    let final_hidden = x.last().expect("checked non-empty");
    let final_hidden = rms_norm(final_hidden, &norm_weight)?;
    let selected_logits = logit_tokens
        .iter()
        .copied()
        .map(|token| {
            let row = model_store::read_bf16_matrix_row(report, "lm_head.weight", token as usize)?;
            if row.len() != final_hidden.len() {
                return Err(eyre!(
                    "lm_head row {} has {} values, but final hidden has {} values",
                    token,
                    row.len(),
                    final_hidden.len()
                ));
            }
            Ok(SelectedLogit {
                token,
                logit: dot(&row, &final_hidden),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(CpuOracleReport {
        weights: model_store::canonical_safetensors_dir()
            .display()
            .to_string(),
        tokens: tokens.to_vec(),
        layers,
        embedding_final_first8,
        layer_checkpoints,
        final_norm_first8: sample8(&final_hidden),
        selected_logits,
    })
}

pub fn rms_norm_reference(x: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    rms_norm(x, weight)
}

pub fn apply_rope_reference(row: &[f32], heads: usize, position: usize) -> Result<Vec<f32>> {
    let (concentration, inv_freq) = yarn_concentration_and_inv_freq();
    let mut row = row.to_vec();
    apply_rope_row(&mut row, heads, position, concentration, &inv_freq)?;
    Ok(row)
}

pub fn single_token_attention_from_rope_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sinks: &[f32],
) -> Result<Vec<f32>> {
    expect_len("q", q, ATTN_VALUES)?;
    expect_len("k", k, KV_HEADS * HEAD_DIM)?;
    expect_len("v", v, KV_HEADS * HEAD_DIM)?;
    expect_len("sinks", sinks, ATTN_HEADS)?;

    let sm_scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0.0f32; ATTN_VALUES];

    for kv_head in 0..KV_HEADS {
        let k_head_start = kv_head * HEAD_DIM;
        let k_head = &k[k_head_start..k_head_start + HEAD_DIM];
        let v_head = &v[k_head_start..k_head_start + HEAD_DIM];

        for q_index in 0..Q_MULT {
            let head = kv_head * Q_MULT + q_index;
            let q_start = head * HEAD_DIM;
            let q_head = &q[q_start..q_start + HEAD_DIM];

            let score = q_head
                .iter()
                .copied()
                .zip(k_head.iter().copied())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * sm_scale;
            let sink = sinks[head];
            let max = score.max(sink);
            let exp_score = (score - max).exp();
            let exp_sink = (sink - max).exp();
            let data_weight = exp_score / (exp_score + exp_sink);

            for dim in 0..HEAD_DIM {
                out[q_start + dim] = data_weight * v_head[dim];
            }
        }
    }

    Ok(out)
}

pub fn sequence_attention_from_rope_reference(
    layer: usize,
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    sinks: &[f32],
) -> Result<Vec<Vec<f32>>> {
    sequence_attention(layer, q, k, v, sinks)
}

pub fn sequence_layer_reference(
    report: &SourceModelReport,
    layer: usize,
    x: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    sequence_layer(report, layer, x)
}

pub fn sequence_layers_reference(
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut x = tokens
        .iter()
        .copied()
        .map(|token| {
            model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)
        })
        .collect::<Result<Vec<_>>>()?;
    if x.is_empty() {
        return Err(eyre!("sequence layers reference needs at least one token"));
    }
    for row in &x {
        expect_len("embedding", row, HIDDEN_SIZE)?;
    }

    for layer in 0..layers {
        if layer >= LAYERS {
            return Err(eyre!(
                "layer {layer} is outside supported gpt-oss-20b depth {LAYERS}"
            ));
        }
        x = sequence_layer(report, layer, &x)?;
    }
    Ok(x)
}

pub fn top_k_softmax_reference(values: &[f32], k: usize) -> Vec<ExpertScore> {
    top_k_softmax(values, k)
}

pub fn mxfp4_expert_matvec_reference(
    report: &SourceModelReport,
    blocks_name: &str,
    scales_name: &str,
    expert: usize,
    rows: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    mxfp4_expert_matvec(report, blocks_name, scales_name, expert, rows, input)
}

pub fn swiglu_reference(values: &[f32]) -> Result<Vec<f32>> {
    swiglu(values)
}

pub fn layer_moe_reference(
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    top_experts: &[ExpertScore],
) -> Result<Vec<f32>> {
    layer_moe(report, layer, input, top_experts)
}

pub fn single_token_layer_reference(
    report: &SourceModelReport,
    layer: usize,
    x: &[f32],
) -> Result<Vec<f32>> {
    single_token_layer(report, layer, x)
}

pub fn single_token_final_norm_reference(
    report: &SourceModelReport,
    token: u32,
    layers: usize,
) -> Result<Vec<f32>> {
    let mut x =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    expect_len("embedding", &x, HIDDEN_SIZE)?;

    for layer in 0..layers {
        if layer >= LAYERS {
            return Err(eyre!(
                "layer {layer} is outside supported gpt-oss-20b depth {LAYERS}"
            ));
        }
        x = single_token_layer(report, layer, &x)?;
    }

    let norm_weight = model_store::read_bf16_vector(report, "model.norm.weight")?;
    rms_norm(&x, &norm_weight)
}

pub fn selected_logits_reference(
    report: &SourceModelReport,
    final_hidden: &[f32],
    logit_tokens: &[u32],
) -> Result<Vec<SelectedLogit>> {
    logit_tokens
        .iter()
        .copied()
        .map(|token| {
            let row = model_store::read_bf16_matrix_row(report, "lm_head.weight", token as usize)?;
            if row.len() != final_hidden.len() {
                return Err(eyre!(
                    "lm_head row {} has {} values, but final hidden has {} values",
                    token,
                    row.len(),
                    final_hidden.len()
                ));
            }
            Ok(SelectedLogit {
                token,
                logit: dot(&row, final_hidden),
            })
        })
        .collect()
}

pub fn probe_greedy_decode(
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prompt_tokens: &[u32],
    layers: usize,
    max_new_tokens: usize,
) -> Result<GreedyDecodeProbeReport> {
    if prompt_tokens.is_empty() {
        return Err(eyre!("cpu greedy decode needs at least one prompt token"));
    }
    if max_new_tokens == 0 {
        return Err(eyre!("cpu greedy decode needs at least one new token"));
    }
    for layer in 0..layers {
        if layer >= LAYERS {
            return Err(eyre!(
                "layer {layer} is outside supported gpt-oss-20b depth {LAYERS}"
            ));
        }
    }

    let norm_weight = model_store::read_bf16_vector(report, "model.norm.weight")?;
    let stop_tokens = harmony.stop_tokens()?;
    let mut tokens = prompt_tokens.to_vec();
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut stop_reason = StopReason::MaxGeneratedTokens;

    for _ in 0..max_new_tokens {
        let hidden = sequence_layers_reference(report, &tokens, layers)?;
        let final_hidden = hidden
            .last()
            .ok_or_else(|| eyre!("cpu greedy decode produced no hidden states"))?;
        let final_hidden = rms_norm(final_hidden, &norm_weight)?;
        let Some((token, logit)) =
            model_store::top_k_matvec_bf16(report, "lm_head.weight", &final_hidden, 1)?
                .into_iter()
                .next()
        else {
            return Err(eyre!("cpu lm_head top-1 returned no token"));
        };

        let token = token as u32;
        generated.push(GreedyTokenReport {
            token,
            logit,
            text: decode_token_text(harmony, token)?,
        });
        tokens.push(token);

        if stop_tokens.contains(&token) {
            stop_reason = StopReason::EndOfGeneration;
            break;
        }
    }

    let token_ids = generated
        .iter()
        .map(|token| token.token)
        .collect::<Vec<_>>();
    Ok(GreedyDecodeProbeReport {
        name: format!(
            "greedy_decode.layers{layers}.prompt{}.new{}",
            prompt_tokens.len(),
            max_new_tokens
        ),
        backend: "cpu".to_string(),
        scorer: "full-sequence CPU recompute + streaming BF16 lm_head top-1".to_string(),
        layers,
        prompt_tokens: prompt_tokens.len(),
        max_new_tokens,
        stop_reason,
        generated,
        text: decode_tokens_text(harmony, &token_ids)?,
    })
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

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(left, right)| left * right)
        .sum()
}

fn l2(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt() as f32
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64) as f32
}

fn rms_norm(x: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if x.len() != weight.len() {
        return Err(eyre!(
            "RMSNorm input has {} values but weight has {} values",
            x.len(),
            weight.len()
        ));
    }
    let mean_square = x
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        / x.len() as f64;
    let scale = (mean_square + 1e-5).sqrt().recip() as f32;
    Ok(x.iter()
        .copied()
        .zip(weight.iter().copied())
        .map(|(value, weight)| value * scale * weight)
        .collect())
}

fn single_token_attention(q: &[f32], k: &[f32], v: &[f32], sinks: &[f32]) -> Result<Vec<f32>> {
    expect_len("q", q, ATTN_VALUES)?;
    expect_len("k", k, KV_HEADS * HEAD_DIM)?;
    expect_len("v", v, KV_HEADS * HEAD_DIM)?;
    expect_len("sinks", sinks, ATTN_HEADS)?;

    let rope_concentration = 0.1 * ROPE_SCALING_FACTOR.ln() + 1.0;
    let sm_scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0.0f32; ATTN_VALUES];

    for kv_head in 0..KV_HEADS {
        let k_head_start = kv_head * HEAD_DIM;
        let k_head = &k[k_head_start..k_head_start + HEAD_DIM];
        let v_head = &v[k_head_start..k_head_start + HEAD_DIM];

        for q_index in 0..Q_MULT {
            let head = kv_head * Q_MULT + q_index;
            let q_start = head * HEAD_DIM;
            let q_head = &q[q_start..q_start + HEAD_DIM];

            // Position 0 RoPE has sin=0 and cos=YaRN concentration.
            let score = q_head
                .iter()
                .copied()
                .zip(k_head.iter().copied())
                .map(|(q, k)| (q * rope_concentration) * (k * rope_concentration))
                .sum::<f32>()
                * sm_scale;
            let sink = sinks[head];
            let max = score.max(sink);
            let exp_score = (score - max).exp();
            let exp_sink = (sink - max).exp();
            let data_weight = exp_score / (exp_score + exp_sink);

            for dim in 0..HEAD_DIM {
                out[q_start + dim] = data_weight * v_head[dim];
            }
        }
    }

    Ok(out)
}

fn top_k_softmax(values: &[f32], k: usize) -> Vec<ExpertScore> {
    let mut scored: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    scored.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(k);

    let max = scored
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let denom = scored
        .iter()
        .map(|(_, value)| (*value - max).exp())
        .sum::<f32>();

    scored
        .into_iter()
        .map(|(index, logit)| ExpertScore {
            index,
            logit,
            weight: (logit - max).exp() / denom,
        })
        .collect()
}

fn single_token_layer(report: &SourceModelReport, layer: usize, x: &[f32]) -> Result<Vec<f32>> {
    expect_len("layer input", x, HIDDEN_SIZE)?;

    let prefix = format!("model.layers.{layer}");
    let input_norm_weight =
        model_store::read_bf16_vector(report, &format!("{prefix}.input_layernorm.weight"))?;
    let attn_input = rms_norm(x, &input_norm_weight)?;

    let mut q = model_store::matvec_bf16(
        report,
        &format!("{prefix}.self_attn.q_proj.weight"),
        &attn_input,
    )?;
    let q_bias = model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.q_proj.bias"))?;
    model_store::add_in_place(&mut q, &q_bias, &format!("{prefix}.self_attn.q_proj"))?;

    let mut k = model_store::matvec_bf16(
        report,
        &format!("{prefix}.self_attn.k_proj.weight"),
        &attn_input,
    )?;
    let k_bias = model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.k_proj.bias"))?;
    model_store::add_in_place(&mut k, &k_bias, &format!("{prefix}.self_attn.k_proj"))?;

    let mut v = model_store::matvec_bf16(
        report,
        &format!("{prefix}.self_attn.v_proj.weight"),
        &attn_input,
    )?;
    let v_bias = model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.v_proj.bias"))?;
    model_store::add_in_place(&mut v, &v_bias, &format!("{prefix}.self_attn.v_proj"))?;

    let sinks = model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.sinks"))?;
    let attn = single_token_attention(&q, &k, &v, &sinks)?;

    let mut projected =
        model_store::matvec_bf16(report, &format!("{prefix}.self_attn.o_proj.weight"), &attn)?;
    let o_bias = model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.o_proj.bias"))?;
    model_store::add_in_place(
        &mut projected,
        &o_bias,
        &format!("{prefix}.self_attn.o_proj"),
    )?;
    expect_len("attention projection", &projected, HIDDEN_SIZE)?;

    let residual: Vec<f32> = x
        .iter()
        .copied()
        .zip(projected.iter().copied())
        .map(|(x, projected)| x + projected)
        .collect();

    let post_norm_weight = model_store::read_bf16_vector(
        report,
        &format!("{prefix}.post_attention_layernorm.weight"),
    )?;
    let router_input = rms_norm(&residual, &post_norm_weight)?;
    let mut router = model_store::matvec_bf16(
        report,
        &format!("{prefix}.mlp.router.weight"),
        &router_input,
    )?;
    let router_bias = model_store::read_bf16_vector(report, &format!("{prefix}.mlp.router.bias"))?;
    model_store::add_in_place(&mut router, &router_bias, &format!("{prefix}.mlp.router"))?;
    let top_experts = top_k_softmax(&router, 4);
    let moe = layer_moe(report, layer, &router_input, &top_experts)?;

    Ok(residual
        .iter()
        .copied()
        .zip(moe.iter().copied())
        .map(|(residual, moe)| residual + moe)
        .collect())
}

fn sequence_layer(
    report: &SourceModelReport,
    layer: usize,
    x: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    for row in x {
        expect_len("layer input", row, HIDDEN_SIZE)?;
    }

    let prefix = format!("model.layers.{layer}");
    let input_norm_weight =
        model_store::read_bf16_vector(report, &format!("{prefix}.input_layernorm.weight"))?;

    let mut q = Vec::with_capacity(x.len());
    let mut k = Vec::with_capacity(x.len());
    let mut v = Vec::with_capacity(x.len());
    for row in x {
        let attn_input = rms_norm(row, &input_norm_weight)?;

        let mut row_q = model_store::matvec_bf16(
            report,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &attn_input,
        )?;
        let q_bias =
            model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.q_proj.bias"))?;
        model_store::add_in_place(&mut row_q, &q_bias, &format!("{prefix}.self_attn.q_proj"))?;
        expect_len("q", &row_q, ATTN_VALUES)?;

        let mut row_k = model_store::matvec_bf16(
            report,
            &format!("{prefix}.self_attn.k_proj.weight"),
            &attn_input,
        )?;
        let k_bias =
            model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.k_proj.bias"))?;
        model_store::add_in_place(&mut row_k, &k_bias, &format!("{prefix}.self_attn.k_proj"))?;
        expect_len("k", &row_k, KV_HEADS * HEAD_DIM)?;

        let mut row_v = model_store::matvec_bf16(
            report,
            &format!("{prefix}.self_attn.v_proj.weight"),
            &attn_input,
        )?;
        let v_bias =
            model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.v_proj.bias"))?;
        model_store::add_in_place(&mut row_v, &v_bias, &format!("{prefix}.self_attn.v_proj"))?;
        expect_len("v", &row_v, KV_HEADS * HEAD_DIM)?;

        q.push(row_q);
        k.push(row_k);
        v.push(row_v);
    }

    apply_rope_sequence(&mut q, &mut k)?;

    let sinks = model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.sinks"))?;
    let attn = sequence_attention(layer, &q, &k, &v, &sinks)?;

    let mut residuals = Vec::with_capacity(x.len());
    for (row, attn) in x.iter().zip(attn) {
        let mut projected =
            model_store::matvec_bf16(report, &format!("{prefix}.self_attn.o_proj.weight"), &attn)?;
        let o_bias =
            model_store::read_bf16_vector(report, &format!("{prefix}.self_attn.o_proj.bias"))?;
        model_store::add_in_place(
            &mut projected,
            &o_bias,
            &format!("{prefix}.self_attn.o_proj"),
        )?;
        expect_len("attention projection", &projected, HIDDEN_SIZE)?;
        residuals.push(
            row.iter()
                .copied()
                .zip(projected.iter().copied())
                .map(|(x, projected)| x + projected)
                .collect::<Vec<_>>(),
        );
    }

    let post_norm_weight = model_store::read_bf16_vector(
        report,
        &format!("{prefix}.post_attention_layernorm.weight"),
    )?;
    let mut out = Vec::with_capacity(residuals.len());
    for residual in residuals {
        let router_input = rms_norm(&residual, &post_norm_weight)?;
        let mut router = model_store::matvec_bf16(
            report,
            &format!("{prefix}.mlp.router.weight"),
            &router_input,
        )?;
        let router_bias =
            model_store::read_bf16_vector(report, &format!("{prefix}.mlp.router.bias"))?;
        model_store::add_in_place(&mut router, &router_bias, &format!("{prefix}.mlp.router"))?;
        let top_experts = top_k_softmax(&router, 4);
        let moe = layer_moe(report, layer, &router_input, &top_experts)?;
        out.push(
            residual
                .iter()
                .copied()
                .zip(moe.iter().copied())
                .map(|(residual, moe)| residual + moe)
                .collect(),
        );
    }

    Ok(out)
}

fn sequence_attention(
    layer: usize,
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    sinks: &[f32],
) -> Result<Vec<Vec<f32>>> {
    if q.len() != k.len() || q.len() != v.len() {
        return Err(eyre!(
            "attention sequence length mismatch: q={}, k={}, v={}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    for row in q {
        expect_len("q", row, ATTN_VALUES)?;
    }
    for row in k {
        expect_len("k", row, KV_HEADS * HEAD_DIM)?;
    }
    for row in v {
        expect_len("v", row, KV_HEADS * HEAD_DIM)?;
    }
    expect_len("sinks", sinks, ATTN_HEADS)?;

    let sliding_window = (layer % 2 == 0).then_some(SLIDING_WINDOW);
    let sm_scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = Vec::with_capacity(q.len());

    for query_position in 0..q.len() {
        let key_start = sliding_window
            .map(|window| (query_position + 1).saturating_sub(window))
            .unwrap_or(0);
        let mut row_out = vec![0.0f32; ATTN_VALUES];

        for kv_head in 0..KV_HEADS {
            let kv_start = kv_head * HEAD_DIM;

            for q_index in 0..Q_MULT {
                let head = kv_head * Q_MULT + q_index;
                let q_start = head * HEAD_DIM;
                let q_head = &q[query_position][q_start..q_start + HEAD_DIM];

                let mut scores = Vec::with_capacity(query_position + 1 - key_start);
                for key_position in key_start..=query_position {
                    let k_head = &k[key_position][kv_start..kv_start + HEAD_DIM];
                    let score = q_head
                        .iter()
                        .copied()
                        .zip(k_head.iter().copied())
                        .map(|(q, k)| q * k)
                        .sum::<f32>()
                        * sm_scale;
                    scores.push((key_position, score));
                }

                let sink = sinks[head];
                let max = scores.iter().map(|(_, score)| *score).fold(sink, f32::max);
                let exp_sink = (sink - max).exp();
                let denom = exp_sink
                    + scores
                        .iter()
                        .map(|(_, score)| (*score - max).exp())
                        .sum::<f32>();

                for (key_position, score) in scores {
                    let weight = (score - max).exp() / denom;
                    let v_head = &v[key_position][kv_start..kv_start + HEAD_DIM];
                    for dim in 0..HEAD_DIM {
                        row_out[q_start + dim] += weight * v_head[dim];
                    }
                }
            }
        }

        out.push(row_out);
    }

    Ok(out)
}

fn apply_rope_sequence(q: &mut [Vec<f32>], k: &mut [Vec<f32>]) -> Result<()> {
    if q.len() != k.len() {
        return Err(eyre!(
            "RoPE sequence length mismatch: q={}, k={}",
            q.len(),
            k.len()
        ));
    }
    let (concentration, inv_freq) = yarn_concentration_and_inv_freq();
    for (position, (q, k)) in q.iter_mut().zip(k.iter_mut()).enumerate() {
        apply_rope_row(q, ATTN_HEADS, position, concentration, &inv_freq)?;
        apply_rope_row(k, KV_HEADS, position, concentration, &inv_freq)?;
    }
    Ok(())
}

fn yarn_concentration_and_inv_freq() -> (f32, Vec<f32>) {
    let concentration = 0.1 * ROPE_SCALING_FACTOR.ln() + 1.0;
    let d_half = HEAD_DIM as f32 / 2.0;
    let low = d_half * (INITIAL_CONTEXT_LENGTH / (ROPE_NTK_BETA * 2.0 * std::f32::consts::PI)).ln()
        / ROPE_THETA.ln();
    let high = d_half
        * (INITIAL_CONTEXT_LENGTH / (ROPE_NTK_ALPHA * 2.0 * std::f32::consts::PI)).ln()
        / ROPE_THETA.ln();

    let inv_freq = (0..HEAD_DIM / 2)
        .map(|dim| {
            let freq = ROPE_THETA.powf((dim * 2) as f32 / HEAD_DIM as f32);
            let interpolation = 1.0 / (ROPE_SCALING_FACTOR * freq);
            let extrapolation = 1.0 / freq;
            let ramp = (dim as f32 - low) / (high - low);
            let mask = 1.0 - ramp.clamp(0.0, 1.0);
            interpolation * (1.0 - mask) + extrapolation * mask
        })
        .collect();
    (concentration, inv_freq)
}

fn apply_rope_row(
    row: &mut [f32],
    heads: usize,
    position: usize,
    concentration: f32,
    inv_freq: &[f32],
) -> Result<()> {
    expect_len("RoPE row", row, heads * HEAD_DIM)?;
    for head in 0..heads {
        let head_start = head * HEAD_DIM;
        let (first_half, second_half) =
            row[head_start..head_start + HEAD_DIM].split_at_mut(HEAD_DIM / 2);
        for dim in 0..HEAD_DIM / 2 {
            let theta = position as f32 * inv_freq[dim];
            let cos = theta.cos() * concentration;
            let sin = theta.sin() * concentration;
            let x1 = first_half[dim];
            let x2 = second_half[dim];
            first_half[dim] = x1 * cos - x2 * sin;
            second_half[dim] = x2 * cos + x1 * sin;
        }
    }
    Ok(())
}

fn layer_moe(
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    top_experts: &[ExpertScore],
) -> Result<Vec<f32>> {
    expect_len("MoE input", input, HIDDEN_SIZE)?;
    let mut out = vec![0.0f32; HIDDEN_SIZE];
    let prefix = format!("model.layers.{layer}.mlp.experts");

    for expert in top_experts {
        if expert.index >= EXPERTS {
            return Err(eyre!(
                "expert index {} is outside expected expert count {EXPERTS}",
                expert.index
            ));
        }

        let mut gate_up = mxfp4_expert_matvec(
            report,
            &format!("{prefix}.gate_up_proj_blocks"),
            &format!("{prefix}.gate_up_proj_scales"),
            expert.index,
            GATE_UP_VALUES,
            input,
        )?;
        let gate_up_bias = model_store::read_bf16_matrix_row(
            report,
            &format!("{prefix}.gate_up_proj_bias"),
            expert.index,
        )?;
        model_store::add_in_place(
            &mut gate_up,
            &gate_up_bias,
            &format!("{prefix}.gate_up_proj"),
        )?;
        let swiglu = swiglu(&gate_up)?;

        let mut down = mxfp4_expert_matvec(
            report,
            &format!("{prefix}.down_proj_blocks"),
            &format!("{prefix}.down_proj_scales"),
            expert.index,
            HIDDEN_SIZE,
            &swiglu,
        )?;
        let down_bias = model_store::read_bf16_matrix_row(
            report,
            &format!("{prefix}.down_proj_bias"),
            expert.index,
        )?;
        model_store::add_in_place(&mut down, &down_bias, &format!("{prefix}.down_proj"))?;

        for (out, value) in out.iter_mut().zip(down) {
            *out += value * expert.weight;
        }
    }

    Ok(out)
}

fn mxfp4_expert_matvec(
    report: &SourceModelReport,
    blocks_name: &str,
    scales_name: &str,
    expert: usize,
    rows: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    expect_len("MXFP4 input", input, HIDDEN_SIZE)?;

    let blocks_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
        .ok_or_else(|| eyre!("MXFP4 block slice size overflow"))?;
    let scales_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .ok_or_else(|| eyre!("MXFP4 scale slice size overflow"))?;
    let block_offset = expert
        .checked_mul(blocks_per_expert)
        .ok_or_else(|| eyre!("MXFP4 block offset overflow"))?;
    let scale_offset = expert
        .checked_mul(scales_per_expert)
        .ok_or_else(|| eyre!("MXFP4 scale offset overflow"))?;

    let blocks =
        model_store::read_u8_tensor_slice(report, blocks_name, block_offset, blocks_per_expert)?;
    let scales =
        model_store::read_u8_tensor_slice(report, scales_name, scale_offset, scales_per_expert)?;

    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0f32;
        for group in 0..MXFP4_GROUPS {
            let scale_index = row * MXFP4_GROUPS + group;
            let scale = 2.0f32.powi(scales[scale_index] as i32 - 127);
            let block_start = scale_index * MXFP4_BYTES_PER_GROUP;
            let input_start = group * 32;

            for byte_index in 0..MXFP4_BYTES_PER_GROUP {
                let packed = blocks[block_start + byte_index];
                let input_index = input_start + byte_index * 2;
                let lo = (packed & 0x0f) as usize;
                let hi = (packed >> 4) as usize;
                sum += FP4_VALUES[lo] * scale * input[input_index];
                sum += FP4_VALUES[hi] * scale * input[input_index + 1];
            }
        }
        out.push(sum);
    }
    Ok(out)
}

fn swiglu(values: &[f32]) -> Result<Vec<f32>> {
    if values.len() != GATE_UP_VALUES {
        return Err(eyre!(
            "SwiGLU input has {} values, expected {GATE_UP_VALUES}",
            values.len()
        ));
    }
    let mut out = Vec::with_capacity(INTERMEDIATE_SIZE);
    for pair in values.chunks_exact(2) {
        let x_glu = pair[0].min(SWIGLU_LIMIT);
        let x_linear = pair[1].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let out_glu = x_glu / (1.0 + (-SWIGLU_ALPHA * x_glu).exp());
        out.push(out_glu * (x_linear + 1.0));
    }
    Ok(out)
}

fn expect_len(name: &str, values: &[f32], len: usize) -> Result<()> {
    if values.len() != len {
        return Err(eyre!("{name} has {} values, expected {len}", values.len()));
    }
    Ok(())
}

fn sample8(values: &[f32]) -> Vec<f32> {
    values.iter().copied().take(8).collect()
}

struct FloatStats {
    min: f32,
    max: f32,
    mean: f32,
    l2: f32,
}

impl FloatStats {
    fn from_values(values: &[f32]) -> Result<Self> {
        let Some((&first, rest)) = values.split_first() else {
            return Err(eyre!("cannot summarize an empty vector"));
        };

        let mut min = first;
        let mut max = first;
        let mut sum = first as f64;
        let mut sum_sq = (first as f64) * (first as f64);

        for &value in rest {
            min = min.min(value);
            max = max.max(value);
            sum += value as f64;
            sum_sq += (value as f64) * (value as f64);
        }

        let len = values.len() as f64;
        Ok(Self {
            min,
            max,
            mean: (sum / len) as f32,
            l2: sum_sq.sqrt() as f32,
        })
    }
}
