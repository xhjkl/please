//! Oco token generation with model-specific conversation framing.
//!
//! The hub keeps one session, sized at load to the GPU memory the weights leave, and every
//! request feeds only what it does not already hold: the longest common prefix of the
//! session's history and the newly rendered prompt stays, the session rewinds to it, and the
//! rest is prefilled. Another conversation keeps at least the static head, whose end is a
//! checkpoint; a different head is read from `~/.please/caches`, or prefilled and saved once.
use super::dialect::{Adapter, Dialect, OutputParser, Rendered};
use super::occupancy::Occupancy;
use super::preamble;
use crate::protocol::{Message, Setting};
use eyre::{Result, ensure, eyre};
use std::ops::Range;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

pub(super) mod clipping;
mod reuse;

const MAX_GENERATED_TOKENS: usize = 8_192;
/// Room kept free for the reply when deciding whether a prompt still fits.
const REPLY_ROOM: usize = 512;
/// Share of the GPU's working set the weights and the session's context may fill; the rest
/// covers fixed-size scratch and recurrent memory, roughly, and leaves the machine breathing.
const GPU_SHARE: f64 = 0.8;
/// Smallest context worth serving: half for instructions and stdin, the rest for a reply.
const MIN_CAPACITY: usize = 4 * REPLY_ROOM;

/// Loaded checkpoint and codec shared by all hub connections.
pub struct Engine {
    /// Canonical checkpoint path, reported to probes.
    pub path: std::path::PathBuf,
    pub load_seconds: f64,
    weights: oco::Weights,
    adapter: Box<dyn Adapter>,
    /// What this checkpoint's saved heads are named after besides their IDs.
    origin: preamble::Origin,
    /// Tokens the session holds, fixed at load.
    capacity: usize,
    /// The only session, `None` until the first request or after a failed one. Holding the
    /// lock serializes the served probe's requests, which may overlap across a reconnection.
    session: tokio::sync::Mutex<Option<Kept>>,
    /// The probe allowed to use the session.
    pub occupancy: Occupancy,
    connections: std::sync::atomic::AtomicU64,
}

/// The session with the conversation it last served.
struct Kept {
    session: oco::Session,
    /// The conversation whose requests it last served.
    owner: u64,
    /// That conversation's history indices clipped from every prompt it was fed.
    dropped: Vec<Range<usize>>,
}

/// Streamed prediction, or the end of one response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Generated {
    Token(u32),
    Stop,
}

/// Where a request's session came from, which decides what its first prefill had to cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// The previous request's session, which already held this prompt's static head.
    Kept,
    /// A fresh session restored from the cached static head.
    Opened,
    /// A fresh session that prefilled the static head and saved it.
    Prefilled,
    /// A fresh, empty session: the history has no cacheable head.
    Fresh,
}

/// What one request cost before its first prediction, and how its reply ended.
#[derive(Debug, Clone, Copy)]
pub struct Turn {
    pub start: Start,
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    /// The reply ended by itself rather than at the output limit.
    pub complete: bool,
}

/// Run file or allocation work off the async runtime.
async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<T> {
    Ok(tokio::task::spawn_blocking(work).await?)
}

impl Engine {
    /// Bind output parsing to this checkpoint's tokenizer and selected dialect.
    pub fn output_parser(&self) -> Result<Box<dyn OutputParser + '_>> {
        self.adapter.output_parser(self.weights.tokenizer())
    }

    /// Encode neutral history with this checkpoint's tokenizer and selected dialect.
    pub fn render(
        &self,
        setting: &Setting,
        history: &[Message],
        tail_budget: usize,
    ) -> Result<Rendered> {
        self.adapter
            .render(self.weights.tokenizer(), setting, history, tail_budget)
    }

    /// Tokens the session holds.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// What share of the context a prompt of this history fills.
    pub fn fill(&self, setting: &Setting, history: &[Message]) -> Result<f64> {
        let tokens = self.render(setting, history, self.capacity)?.tokens.len();
        Ok(tokens as f64 / self.capacity as f64)
    }

    /// A conversation key shared with no other, for prompts that extend nothing kept.
    pub fn connect(&self) -> u64 {
        self.connections
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Start a session from the static head: restored from the cache, or prefilled and saved.
    async fn start(&self, head: &[u32]) -> Result<(oco::Session, Start)> {
        let capacity = self.capacity;
        let weights = self.weights.clone();
        if head.is_empty() {
            let session = blocking(move || weights.new_session(capacity)).await?;
            let session = session.map_err(|error| eyre!(error))?;
            return Ok((session, Start::Fresh));
        }
        let path = self.origin.path(head);
        let opened = {
            let (weights, path) = (weights.clone(), path.clone());
            blocking(move || weights.open_session(path, capacity)).await?
        };
        let opened = opened
            .inspect_err(|error| tracing::debug!("hub: no cached head: {error:#}"))
            .ok();
        if let Some(session) = reuse::adopt_head(opened, head) {
            if let Err(error) = preamble::touch(&path) {
                tracing::debug!("hub: cannot mark the head as used: {error:#}");
            }
            return Ok((session, Start::Opened));
        }
        let session = blocking(move || weights.new_session(capacity)).await?;
        let mut session = session.map_err(|error| eyre!(error))?;
        let mut inference = session.infer(head).map_err(|error| eyre!(error))?;
        inference.next().await.map_err(|error| eyre!(error))?;
        drop(inference);
        // The head stays a checkpoint, so rewinds into the conversation never replay it.
        session.checkpoint().map_err(|error| eyre!(error))?;
        let (session, saved) = blocking(move || {
            // A failed save costs the next invocation a prefill, never this request.
            let saved = preamble::save(&path, |path| {
                session.save(path).map_err(|error| eyre!(error))
            });
            (session, saved.map(|()| path))
        })
        .await?;
        match saved {
            Ok(path) => tracing::debug!(
                "hub: saved {} head tokens to {}",
                head.len(),
                path.display()
            ),
            Err(error) => tracing::warn!("hub: cannot cache the head: {error:#}"),
        }
        Ok((session, Start::Prefilled))
    }
}

/// Load an existing checkpoint; neither the engine nor its tokenizer downloads weights.
pub async fn load_model(model_path: &std::path::Path, dialect: Dialect) -> Result<Engine> {
    let started = Instant::now();
    let path = std::fs::canonicalize(model_path)?;
    let load_path = path.clone();
    let weights = blocking(move || oco::load(load_path)).await?;
    let weights = weights.map_err(|error| eyre!(error))?;
    let capacity = weights.capacity_within(GPU_SHARE);
    // `PLEASE_CONTEXT` caps the context, to try long-conversation behavior on short ones.
    let capacity = match std::env::var("PLEASE_CONTEXT").ok().map(|cap| cap.parse()) {
        Some(Ok(cap)) => capacity.min(cap),
        Some(Err(_)) => eyre::bail!("PLEASE_CONTEXT must be a number of tokens"),
        None => capacity,
    };
    ensure!(
        capacity >= MIN_CAPACITY,
        "{} leaves too little GPU memory for a context ({capacity} tokens)",
        path.display()
    );
    tracing::info!("hub: context capacity {capacity} tokens");
    let adapter = dialect.adapter();
    let origin = preamble::Origin::new(&path)?;
    let load_seconds = started.elapsed().as_secs_f64();
    Ok(Engine {
        path,
        load_seconds,
        weights,
        adapter,
        origin,
        capacity,
        session: tokio::sync::Mutex::new(None),
        occupancy: Occupancy::default(),
        connections: std::sync::atomic::AtomicU64::new(0),
    })
}

/// Stream greedy predictions, stopping at a model boundary or a bounded output limit.
pub async fn generate(
    engine: &Engine,
    conversation: u64,
    setting: &Setting,
    history: &[Message],
    generated: UnboundedSender<Generated>,
    max_tokens: Option<usize>,
) -> Result<Option<Turn>> {
    let mut held = engine.session.lock().await;
    let started = Instant::now();
    if generated.is_closed() {
        return Ok(None);
    }
    let capacity = engine.capacity;
    let base = engine.render(setting, &[], capacity)?.tokens.len();
    if base >= capacity / 2 {
        eyre::bail!(
            "system instructions and stdin exceed half the {capacity}-token context; shorten the input"
        );
    }
    let tail_budget = capacity - base - REPLY_ROOM;
    let whole = engine.render(setting, history, tail_budget)?;
    let head = &whole.tokens[..whole.preamble];
    // Holding the head means rewinding to it is free, whichever conversation came before;
    // otherwise the old session goes before its replacement is allocated.
    let kept = held
        .take()
        .filter(|kept| kept.session.history().starts_with(head));
    let (mut kept, start) = match kept {
        Some(kept) => (kept, Start::Kept),
        None => {
            let (session, start) = engine.start(head).await?;
            let kept = Kept {
                session,
                owner: conversation,
                dropped: vec![],
            };
            (kept, start)
        }
    };
    if kept.owner != conversation {
        // Clipping indices describe the previous conversation's history.
        kept.owner = conversation;
        kept.dropped.clear();
    }
    let size = |visible: &[Message]| Ok(engine.render(setting, visible, tail_budget)?.tokens.len());
    let limit = capacity - REPLY_ROOM;
    kept.dropped = clipping::choose(history, &kept.dropped, limit, capacity / 2, size)?;
    let visible = clipping::visible(history, &kept.dropped);
    let mut tokens = engine.render(setting, &visible, tail_budget)?.tokens;
    let limit = max_tokens.unwrap_or(MAX_GENERATED_TOKENS);
    let mut reply = Vec::new();
    let mut turn = None;
    let outcome = loop {
        let reused = reuse::rewind_point(kept.session.history(), &tokens);
        tracing::debug!(
            prompt = tokens.len(),
            reused,
            fed = tokens.len() - reused,
            "hub: feeding prompt"
        );
        let first = match feed(&mut kept.session, &tokens, reused).await {
            Ok(first) => first,
            Err(error) => break Err(error),
        };
        turn.get_or_insert_with(|| {
            let turn = Turn {
                start,
                prompt_tokens: tokens.len(),
                reused_tokens: reused,
                complete: false,
            };
            // Measured from taking the session, so it includes any head prefill.
            let seconds = started.elapsed().as_secs_f64();
            tracing::debug!(?turn, seconds, "hub: first token");
            turn
        });
        let decoded = decode(
            engine,
            &mut kept.session,
            first,
            &generated,
            &mut reply,
            limit,
        )
        .await;
        match decoded {
            Ok(Some(ended)) => break Ok(ended),
            Ok(None) => {}
            Err(error) => break Err(error),
        }
        // The context filled mid-reply: drop more history so the reply can continue.
        let size = |visible: &[Message]| {
            Ok(engine.render(setting, visible, tail_budget)?.tokens.len() + reply.len())
        };
        kept.dropped = match clipping::choose(
            history,
            &kept.dropped,
            capacity - REPLY_ROOM,
            capacity / 2,
            size,
        ) {
            Ok(dropped) => dropped,
            Err(error) => break Err(error),
        };
        let visible = clipping::visible(history, &kept.dropped);
        tokens = match engine.render(setting, &visible, tail_budget) {
            Ok(rendered) => rendered.tokens,
            Err(error) => break Err(error),
        };
        tokens.extend(&reply);
    };
    // An error may leave GPU work in flight, so only a session that finished cleanly is kept.
    match outcome {
        Ok(ended) => {
            *held = Some(kept);
            if ended == Ended::Limit && max_tokens.is_none() {
                eyre::bail!("model exceeded the {limit}-token output limit");
            }
            Ok(turn.map(|turn| Turn {
                complete: ended == Ended::Finished,
                ..turn
            }))
        }
        Err(error) => Err(error),
    }
}

/// Feed the prompt past its reused prefix and return the first prediction, checkpointing the
/// consumed prompt so a later rewind replays at most one reply.
async fn feed(session: &mut oco::Session, tokens: &[u32], reused: usize) -> Result<u32> {
    session.rewind(reused).map_err(|error| eyre!(error))?;
    let mut inference = session
        .infer(&tokens[reused..])
        .map_err(|error| eyre!(error))?;
    let first = inference.next().await.map_err(|error| eyre!(error))?;
    drop(inference);
    let first = first.ok_or_else(|| eyre!("context capacity exhausted by the prompt"))?;
    session.checkpoint().map_err(|error| eyre!(error))?;
    Ok(first)
}

/// Why decoding a reply stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ended {
    /// A stop token, or the probe went away.
    Finished,
    /// The output limit.
    Limit,
}

/// Stream the reply from its first prediction; `None` means the context filled first.
async fn decode(
    engine: &Engine,
    session: &mut oco::Session,
    first: u32,
    generated: &UnboundedSender<Generated>,
    reply: &mut Vec<u32>,
    limit: usize,
) -> Result<Option<Ended>> {
    let mut inference = session.infer(vec![first]).map_err(|error| eyre!(error))?;
    let mut token = Some(first);
    while let Some(next) = token {
        if generated.is_closed() {
            return Ok(Some(Ended::Finished));
        }
        if reply.len() == limit {
            let _ = generated.send(Generated::Stop);
            return Ok(Some(Ended::Limit));
        }
        let stopped = engine
            .adapter
            .is_stop_token(engine.weights.tokenizer(), next);
        if generated.send(Generated::Token(next)).is_err() {
            return Ok(Some(Ended::Finished));
        }
        if stopped {
            let _ = generated.send(Generated::Stop);
            return Ok(Some(Ended::Finished));
        }
        reply.push(next);
        token = inference.next().await.map_err(|error| eyre!(error))?;
    }
    Ok(None)
}
