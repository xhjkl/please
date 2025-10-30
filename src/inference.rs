//! Inference: load a model, render a chat prompt, and stream tokens with sliding-window KV cache reuse.
//! Terminology:
//! - "preamble" = system/dev messages pinned at the front of the prompt and preserved across compactions.
//! - "context capacity" (ctx_cap) = model context window in tokens.
//! - "logits_idx" = the batch index whose logits we sample from.

use eyre::{Result, eyre};
use gg::context::LlamaContext;
use gg::context::params::LlamaContextParams;
use gg::llama_backend::LlamaBackend;
use gg::llama_batch::LlamaBatch;
use gg::model::params::LlamaModelParams;
use gg::model::{AddBos, LlamaModel, Special};
use gg::sampling::LlamaSampler;
use gg::token::LlamaToken;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::harmony::templating::render_prompt_from_history;
use crate::protocol::Message;

mod intuition;
use intuition::{pick_n_ctx_by_vram, vram_free_bytes};

const USE_MIROSTAT: bool = true;

/// Load the model into memory (GPU layers enabled by default) and return backend+model.
pub fn load_model(model_path: &str) -> Result<(LlamaBackend, LlamaModel)> {
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;
    Ok((backend, model))
}

/// Infer and stream token ids via `token_tx`.
/// Sliding window keeps the system preamble pinned.
pub fn infer_token_ids_into_stream(
    backend: &LlamaBackend,
    model: &LlamaModel,
    history: &[Message],
    token_tx: tokio::sync::mpsc::UnboundedSender<u32>,
) -> Result<()> {
    // Render chat to text using Harmony markup to match the documented behavior.
    let prompt = render_prompt_from_history(history, true)?;

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

    // Number of tokens in the pinned preamble (system/dev), capped to ctx_cap-1.
    let preamble_len = compute_preamble_len(&mut ctx, history, ctx_cap)?;

    // Tokenize, clipping to context capacity while preserving the preamble + most recent tail.
    let prompt_tokens = tokenize_clip_to_ctx(&mut ctx, &prompt, preamble_len, ctx_cap)?;

    // Prefill: chunked; logits on the last token only.
    let mut batch = LlamaBatch::new(batch_size as usize, 1);
    ctx.clear_kv_cache();
    let mut logits_idx =
        prefill_returning_logits_idx(&mut ctx, &mut batch, &prompt_tokens, batch_size as usize)?;

    let seed: u32 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(31337);
    let mut sampler = if USE_MIROSTAT {
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.0, 0.0, 0.0),
            LlamaSampler::temp(1.0), // letting Mirostat control entropy
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
    // Prime repetition penalties with the prompt tokens.
    .with_tokens(prompt_tokens.iter().copied());

    // Rolling token buffer backing the sliding window.
    let mut rolling_tokens = prompt_tokens.clone();
    let mut pos = rolling_tokens.len();

    loop {
        // If we're at/over the context limit, rebuild KV with `[system prefix | recent tail]`.
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
        if ctx.model.is_eog_token(token) {
            // Done generating; stop the inference loop.
            break;
        }

        // Update repetition penalty state with the generated token.
        sampler.accept(token);

        // Stream token id
        let sent = token_tx.send(token.0 as u32);
        if sent.is_err() {
            // Consumer dropped; abort generation cleanly.
            break;
        }

        // Decode a single token at the current position; request logits at index 0
        batch.clear();
        batch.add(token, pos as i32, &[0], true)?;
        ctx.decode(&mut batch)?;

        // Single-token decode; logits are at index 0
        logits_idx = 0;
        pos += 1;
        rolling_tokens.push(token);
    }

    Ok(())
}

/// Generate the model response to the turn and stream UTF-8 text pieces through `piece_tx`.
pub async fn infer_into_stream(
    backend: &LlamaBackend,
    model: &LlamaModel,
    history: &[Message],
    piece_tx: tokio::sync::mpsc::UnboundedSender<String>,
) -> Result<Vec<u8>> {
    let (token_id_tx, mut token_id_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();

    // Use the provided chat history directly; template rendering occurs in inference.
    let history = history.to_owned();
    // Safety: transmute only to satisfy `spawn_blocking`'s `'static` bound.
    // We assume that:
    // * we await the `JoinHandle` before either reference can drop;
    // * the closure does not store or spawn further tasks;
    // * all access remains on this thread.
    // If this changes, this should be inside an `Arc` instead of `transmute`.
    let also_backend = unsafe { std::mem::transmute::<&_, &'static LlamaBackend>(backend) };
    let also_model = unsafe { std::mem::transmute::<&_, &'static LlamaModel>(model) };
    let inference = tokio::task::spawn_blocking(move || {
        infer_token_ids_into_stream(also_backend, also_model, &history, token_id_tx)
    });

    // Incrementally detokenize using the model's tokenizer emitting only valid UTF-8 code points.
    // We accumulate raw bytes from tokens and flush only valid UTF-8 slices downstream.
    let mut pending: Vec<u8> = Vec::new();

    while let Some(t) = token_id_rx.recv().await {
        // Convert token to bytes and accumulate; only emit valid UTF-8 codepoints.
        let token = LlamaToken::new(t as i32);
        let bytes = model
            .token_to_bytes(token, Special::Tokenize)
            .map_err(|e| eyre!(e))?;
        pending.extend_from_slice(&bytes);

        // Emit the maximal valid UTF-8 prefix; retain any incomplete tail.
        loop {
            match std::str::from_utf8(&pending) {
                Ok(piece) => {
                    if piece.is_empty() {
                        break;
                    }
                    piece_tx.send(piece.to_string())?;
                    pending.clear();
                    break; // nothing left to emit right now
                }
                Err(err) => {
                    let n = err.valid_up_to();
                    if n == 0 {
                        // Wait for more bytes to complete the first codepoint.
                        break;
                    }
                    // Emit the valid prefix and keep the incomplete tail.
                    let piece = std::str::from_utf8(&pending[..n]).unwrap();
                    piece_tx.send(piece.to_string())?;
                    pending.drain(..n);
                    // Continue the loop to try emitting further valid segments.
                }
            }
        }
    }

    // Ensure inference completed
    inference.await.map_err(|e| eyre!(e))??;
    Ok(pending)
}

/// Compute the number of tokens in the pinned preamble (system/dev only), clamped to `ctx_cap-1`.
fn compute_preamble_len(
    ctx: &mut LlamaContext,
    history: &[Message],
    ctx_cap: usize,
) -> Result<usize> {
    let preamble_only: Vec<Message> = history
        .iter()
        .filter_map(|m| match m {
            Message::System(s) => Some(Message::System(s.clone())),
            Message::Developer(s) => Some(Message::Developer(s.clone())),
            _ => None,
        })
        .collect();

    let n = if !preamble_only.is_empty() {
        let preamble_prompt = render_prompt_from_history(&preamble_only, true)?;
        ctx.model
            .str_to_token(&preamble_prompt, AddBos::Never)?
            .len()
    } else {
        0
    };
    Ok(n.min(ctx_cap.saturating_sub(1)))
}

/// Tokenize `prompt`, clipping to context capacity while keeping `[preamble | recent tail]`.
fn tokenize_clip_to_ctx(
    ctx: &mut LlamaContext,
    prompt: &str,
    preamble_len: usize,
    ctx_cap: usize,
) -> Result<Vec<LlamaToken>> {
    let mut toks = ctx.model.str_to_token(prompt, AddBos::Never)?;
    let keep = toks.len().min(preamble_len);

    if toks.len() > ctx_cap.saturating_sub(1) {
        let tail_room = ctx_cap.saturating_sub(1 + keep);
        let start = toks.len().saturating_sub(tail_room);
        let mut clipped = Vec::with_capacity(keep + tail_room);
        clipped.extend_from_slice(&toks[..keep]);
        clipped.extend_from_slice(&toks[start..]);
        toks = clipped;
    }
    Ok(toks)
}

/// Prefill the prompt in chunks; return the batch index (`logits_idx`) that has logits.
fn prefill_returning_logits_idx(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    toks: &[LlamaToken],
    batch_size: usize,
) -> Result<i32> {
    let mut pos = 0usize;
    let mut logits_idx: i32 = 0;
    for chunk in toks.chunks(batch_size) {
        batch.clear();
        for (i, &t) in chunk.iter().enumerate() {
            let want_logits = (pos + i + 1) == toks.len();
            if want_logits {
                logits_idx = i as i32;
            }
            batch.add(t, (pos + i) as i32, &[0], want_logits)?;
        }
        ctx.decode(batch)?;
        pos += chunk.len();
    }
    Ok(logits_idx)
}

/// Rebuild KV cache using a sliding window: `[preamble | most-recent tail]`.
/// Leaves a slack margin of headroom that scales with `ctx_cap` so the next compaction doesn't trigger immediately.
/// Returns `(tokens, pos, logits_idx)`.
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
    let mut logits_idx: i32 = 0;
    for chunk in compact.chunks(batch_size) {
        batch.clear();
        for (i, &t) in chunk.iter().enumerate() {
            let want_logits = (new_pos + i + 1) == compact.len();
            if want_logits {
                logits_idx = i as i32;
            }
            batch.add(t, (new_pos + i) as i32, &[0], want_logits)?;
        }
        ctx.decode(batch)?;
        new_pos += chunk.len();
    }

    tracing::trace!(
        ?new_pos,
        ?logits_idx,
        ?slack,
        "rebuilt kv with sliding window"
    );

    Ok((compact, new_pos, logits_idx))
}
