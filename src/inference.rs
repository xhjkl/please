//! Inference boundary over llama.cpp-rs.
//!
//! The hot worker deals only in token ids. Harmony rendering/parsing lives one
//! layer above it, and callers decide how to display or route generated events.

use eyre::{Result, eyre};
use gg::context::LlamaContext;
use gg::context::params::LlamaContextParams;
use gg::llama_backend::LlamaBackend;
use gg::llama_batch::LlamaBatch;
use gg::model::LlamaModel;
use gg::model::params::LlamaModelParams;
use gg::sampling::LlamaSampler;
use gg::token::LlamaToken;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::harmony::HarmonyAdapter;
use crate::protocol::Message;

mod intuition;
use intuition::{pick_n_ctx_by_vram, vram_free_bytes};

const USE_MIROSTAT: bool = true;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Generated {
    Token(u32),
    Stop,
}

pub type GenerationSender = tokio::sync::mpsc::UnboundedSender<Generated>;

/// Load the model into memory through llama.cpp. Metal layers are enabled on macOS by the
/// dependency feature rather than through please-owned kernels.
pub fn load_model(model_path: &str) -> Result<(LlamaBackend, LlamaModel)> {
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;
    Ok((backend, model))
}

pub fn generate_tokens_into_stream(
    backend: &LlamaBackend,
    model: &LlamaModel,
    history: &[Message],
    generated: GenerationSender,
) -> Result<()> {
    let harmony = HarmonyAdapter::gpt_oss()?;
    let prompt_token_ids = harmony.render_protocol_tokens(history)?;

    let num_threads = std::thread::available_parallelism()
        .ok()
        .map(|n| n.get())
        .unwrap_or(1);

    let batch_size = 512;
    let n_ctx = vram_free_bytes()
        .map(|free| pick_n_ctx_by_vram(model, free))
        .unwrap_or_else(|| std::num::NonZeroU32::new(8_192.min(model.n_ctx_train())).unwrap());
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_threads(num_threads as i32)
        .with_n_threads_batch(num_threads as i32)
        .with_n_batch(batch_size as u32)
        .with_n_ubatch(batch_size as u32);
    let mut ctx = model.new_context(backend, ctx_params)?;
    let ctx_cap = ctx.n_ctx() as usize;

    let preamble_len = compute_preamble_len(&harmony, history, ctx_cap)?;
    let prompt_tokens = clip_to_ctx(prompt_token_ids, preamble_len, ctx_cap)
        .into_iter()
        .map(token_to_llama)
        .collect::<Result<Vec<_>>>()?;

    let mut batch = LlamaBatch::new(batch_size as usize, 1);
    ctx.clear_kv_cache();
    let mut logits_idx =
        prefill_returning_logits_idx(&mut ctx, &mut batch, &prompt_tokens, batch_size as usize)?;

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(31337);
    let mut sampler = if USE_MIROSTAT {
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.0, 0.0, 0.0),
            LlamaSampler::temp(1.0),
            LlamaSampler::mirostat_v2(seed, 5.0, 0.1),
        ])
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::temp(0.8),
            LlamaSampler::dist(seed),
        ])
    }
    .with_tokens(prompt_tokens.iter().copied());

    let mut rolling_tokens = prompt_tokens.clone();
    let mut pos = rolling_tokens.len();

    loop {
        if pos >= ctx_cap {
            let (compact, new_pos, new_logits_idx) = rebuild_kv_with_sliding_window(
                &mut ctx,
                &mut batch,
                &rolling_tokens,
                preamble_len,
                ctx_cap,
                batch_size as usize,
            )?;
            rolling_tokens = compact;
            pos = new_pos;
            logits_idx = new_logits_idx;
        }

        let token = sampler.sample(&ctx, logits_idx);
        let token_id = token_to_u32(token)?;
        let is_harmony_stop = harmony.is_stop_token(token_id);
        let is_model_eog = ctx.model.is_eog_token(token);

        if is_model_eog && !is_harmony_stop {
            break;
        }
        if generated.send(Generated::Token(token_id)).is_err() {
            break;
        }
        if is_harmony_stop {
            break;
        }

        sampler.accept(token);

        batch.clear();
        batch.add(token, pos as i32, &[0], true)?;
        ctx.decode(&mut batch)?;

        logits_idx = 0;
        pos += 1;
        rolling_tokens.push(token);
    }

    let _ = generated.send(Generated::Stop);
    Ok(())
}

fn token_to_llama(token: u32) -> Result<LlamaToken> {
    let token = i32::try_from(token)?;
    Ok(LlamaToken::new(token))
}

fn token_to_u32(token: LlamaToken) -> Result<u32> {
    u32::try_from(token.0).map_err(|error| eyre!(error))
}

fn compute_preamble_len(
    harmony: &HarmonyAdapter,
    history: &[Message],
    ctx_cap: usize,
) -> Result<usize> {
    let preamble_only = history
        .iter()
        .filter(|message| matches!(message, Message::System(_) | Message::Developer(_)))
        .cloned()
        .collect::<Vec<_>>();
    if preamble_only.is_empty() {
        return Ok(0);
    }
    let tokens = harmony.render_protocol_tokens(&preamble_only)?;
    Ok(tokens.len().min(ctx_cap.saturating_sub(1)))
}

fn clip_to_ctx(mut tokens: Vec<u32>, preamble_len: usize, ctx_cap: usize) -> Vec<u32> {
    let keep = tokens.len().min(preamble_len);
    if tokens.len() > ctx_cap.saturating_sub(1) {
        let tail_room = ctx_cap.saturating_sub(1 + keep);
        let start = tokens.len().saturating_sub(tail_room);
        let mut clipped = Vec::with_capacity(keep + tail_room);
        clipped.extend_from_slice(&tokens[..keep]);
        clipped.extend_from_slice(&tokens[start..]);
        tokens = clipped;
    }
    tokens
}

fn prefill_returning_logits_idx(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    toks: &[LlamaToken],
    batch_size: usize,
) -> Result<i32> {
    let mut pos = 0usize;
    let mut logits_idx = 0;
    for chunk in toks.chunks(batch_size) {
        batch.clear();
        for (i, &token) in chunk.iter().enumerate() {
            let want_logits = (pos + i + 1) == toks.len();
            if want_logits {
                logits_idx = i as i32;
            }
            batch.add(token, (pos + i) as i32, &[0], want_logits)?;
        }
        ctx.decode(batch)?;
        pos += chunk.len();
    }
    Ok(logits_idx)
}

fn rebuild_kv_with_sliding_window(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    rolling_tokens: &[LlamaToken],
    preamble_len: usize,
    ctx_cap: usize,
    batch_size: usize,
) -> Result<(Vec<LlamaToken>, usize, i32)> {
    let keep = rolling_tokens.len().min(preamble_len);
    let available_tail_room = ctx_cap.saturating_sub(1 + keep);
    let slack = ((ctx_cap + 31).saturating_div(32))
        .max(128)
        .min(available_tail_room);
    let tail_room = available_tail_room.saturating_sub(slack);
    let tail_start = rolling_tokens.len().saturating_sub(tail_room);

    tracing::trace!(
        ?ctx_cap,
        ?preamble_len,
        ?slack,
        "rebuilding kv with sliding window"
    );

    let mut compact = Vec::with_capacity(keep + (rolling_tokens.len() - tail_start));
    compact.extend_from_slice(&rolling_tokens[..keep]);
    compact.extend_from_slice(&rolling_tokens[tail_start..]);

    ctx.clear_kv_cache();

    let mut new_pos = 0usize;
    let mut logits_idx = 0;
    for chunk in compact.chunks(batch_size) {
        batch.clear();
        for (i, &token) in chunk.iter().enumerate() {
            let want_logits = (new_pos + i + 1) == compact.len();
            if want_logits {
                logits_idx = i as i32;
            }
            batch.add(token, (new_pos + i) as i32, &[0], want_logits)?;
        }
        ctx.decode(batch)?;
        new_pos += chunk.len();
    }

    Ok((compact, new_pos, logits_idx))
}
