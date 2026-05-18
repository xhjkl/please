use eyre::{Result, eyre};

use super::{
    ATTN_VALUES, GpuStage, HIDDEN_SIZE, KV_HEADS, KV_VALUES, LAYERS, LM_HEAD_TOP1_BLOCK_SIZE,
    MAX_RESIDENT_CONTEXT_TOKENS, MetalRuntime, Q_HEADS, StageMarker, TokenStage, platform,
    stage_marker,
    weights::{GptOssLayerWeights, GptOssWeights},
};
#[cfg(feature = "profile")]
use super::{MetalProfile, ProfileDelta};
use crate::model_store;
use crate::{Generated, GenerationStream};
use std::fmt;
use std::mem::size_of;
use std::ops::Range;
use std::sync::{
    Arc, Mutex,
    mpsc::{self, Receiver, Sender},
};
use std::thread;
use std::time::{Duration, Instant};

const PREFILL_MOE_CHUNK_TOKENS: usize = 16;
const RMS_PARTIALS: usize = HIDDEN_SIZE.div_ceil(256);
// gpt-oss/Harmony terminal formatting tokens: <|return|>, <|end|>, <|call|>.
// When more model families exist, load this from the model/codec contract
// instead of keeping it as a backend constant.
const GPT_OSS_STOP_TOKENS: [u32; 3] = [200002, 200007, 200012];

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
    rms_partials: platform::F32VectorBuffer,
    normed: platform::F32VectorBuffer,
    q: platform::F32VectorBuffer,
    q_rope: platform::F32VectorBuffer,
    k: platform::F32VectorBuffer,
    v: platform::F32VectorBuffer,
    attn: platform::F32VectorBuffer,
    residual: platform::F32VectorBuffer,
    router_input: platform::F32VectorBuffer,
    router_logits: platform::F32VectorBuffer,
    router_indices: platform::U32Buffer,
    router_selected_logits: platform::F32VectorBuffer,
    router_weights: platform::F32VectorBuffer,
    expert_acts_packed: platform::F32VectorBuffer,
    expert_downs_packed: platform::F32VectorBuffer,
    prefill_hidden_ping: platform::F32VectorBuffer,
    prefill_hidden_pong: platform::F32VectorBuffer,
    prefill_normed: platform::F32VectorBuffer,
    prefill_q: platform::F32VectorBuffer,
    prefill_q_rope: platform::F32VectorBuffer,
    prefill_k: platform::F32VectorBuffer,
    prefill_k_rope: platform::F32VectorBuffer,
    prefill_v: platform::F32VectorBuffer,
    prefill_attn: platform::F32VectorBuffer,
    prefill_residual: platform::F32VectorBuffer,
    prefill_router_input: platform::F32VectorBuffer,
    prefill_router_logits: platform::F32VectorBuffer,
    prefill_router_indices: platform::U32Buffer,
    prefill_router_selected_logits: platform::F32VectorBuffer,
    prefill_router_weights: platform::F32VectorBuffer,
    prefill_expert_acts_packed: platform::F32VectorBuffer,
    prefill_tokens: platform::U32Buffer,
    final_hidden: platform::F32VectorBuffer,
    lm_logits: platform::F32VectorBuffer,
    lm_top1_block_indices: platform::U32Buffer,
    lm_top1_block_values: platform::F32VectorBuffer,
    lm_top_indices: platform::U32Buffer,
    lm_top_values: platform::F32VectorBuffer,
    lm_sample_result: platform::U32Buffer,
}

impl GenerationScratch {
    fn new(platform: &platform::MetalContext, vocab: usize, context_tokens: usize) -> Result<Self> {
        let lm_top1_blocks = vocab
            .div_ceil(LM_HEAD_TOP1_BLOCK_SIZE)
            .max(vocab.div_ceil(4));
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
            rms_partials: platform.alloc_f32_vector(RMS_PARTIALS)?,
            normed: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            q: platform.alloc_f32_vector(ATTN_VALUES)?,
            q_rope: platform.alloc_f32_vector(ATTN_VALUES)?,
            k: platform.alloc_f32_vector(KV_VALUES)?,
            v: platform.alloc_f32_vector(KV_VALUES)?,
            attn: platform.alloc_f32_vector(ATTN_VALUES)?,
            residual: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            router_input: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            router_logits: platform.alloc_f32_vector(32)?,
            router_indices: platform.alloc_u32_buffer(4)?,
            router_selected_logits: platform.alloc_f32_vector(4)?,
            router_weights: platform.alloc_f32_vector(4)?,
            expert_acts_packed: platform.alloc_f32_vector(4 * HIDDEN_SIZE)?,
            expert_downs_packed: platform.alloc_f32_vector(4 * HIDDEN_SIZE)?,
            prefill_hidden_ping: platform.alloc_f32_vector(prefill_values)?,
            prefill_hidden_pong: platform.alloc_f32_vector(prefill_values)?,
            prefill_normed: platform.alloc_f32_vector(prefill_values)?,
            prefill_q: platform.alloc_f32_vector(prefill_attn_values)?,
            prefill_q_rope: platform.alloc_f32_vector(prefill_attn_values)?,
            prefill_k: platform.alloc_f32_vector(prefill_kv_values)?,
            prefill_k_rope: platform.alloc_f32_vector(prefill_kv_values)?,
            prefill_v: platform.alloc_f32_vector(prefill_kv_values)?,
            prefill_attn: platform.alloc_f32_vector(prefill_attn_values)?,
            prefill_residual: platform.alloc_f32_vector(prefill_values)?,
            prefill_router_input: platform.alloc_f32_vector(prefill_values)?,
            prefill_router_logits: platform.alloc_f32_vector(prefill_router_values)?,
            prefill_router_indices: platform.alloc_u32_buffer(prefill_router_choice_values)?,
            prefill_router_selected_logits: platform
                .alloc_f32_vector(prefill_router_choice_values)?,
            prefill_router_weights: platform.alloc_f32_vector(prefill_router_choice_values)?,
            prefill_expert_acts_packed: platform.alloc_f32_vector(prefill_expert_act_values)?,
            prefill_tokens: platform.alloc_u32_buffer(context_tokens)?,
            final_hidden: platform.alloc_f32_vector(HIDDEN_SIZE)?,
            lm_logits: platform.alloc_f32_vector(vocab)?,
            lm_top1_block_indices: platform.alloc_u32_buffer(lm_top1_blocks)?,
            lm_top1_block_values: platform.alloc_f32_vector(lm_top1_blocks)?,
            lm_top_indices: platform.alloc_u32_buffer(8)?,
            lm_top_values: platform.alloc_f32_vector(8)?,
            lm_sample_result: platform.alloc_u32_buffer(4)?,
        })
    }
}

#[derive(Clone)]
struct KvCache {
    layers: Vec<LayerKvCache>,
    capacity: usize,
}

#[derive(Clone)]
struct LayerKvCache {
    k: platform::F32VectorBuffer,
    v: platform::F32VectorBuffer,
}

impl KvCache {
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

struct DecodeSession {
    scratch: GenerationScratch,
    kv_cache: KvCache,
    context_capacity: usize,
    layers: usize,
    vocab: usize,
}

impl DecodeSession {
    fn new(
        platform: &platform::MetalContext,
        vocab: usize,
        layers: usize,
        context_capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            scratch: GenerationScratch::new(platform, vocab, context_capacity)?,
            kv_cache: KvCache::new(platform, layers, context_capacity)?,
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

/// Loaded, warmed inference rig: GGUF weights, compiled Metal
/// kernels, and runtime state shared by episodes. Prompts live in episodes, not
/// on the model.
pub struct MetalModel {
    inner: Arc<Mutex<MetalModelInner>>,
}

struct MetalModelInner {
    ctx: MetalRuntime,
    weights: GptOssWeights,
    layers: usize,
}

pub struct MetalEpisode {
    model: Arc<Mutex<MetalModelInner>>,
    state: Arc<Mutex<EpisodeState>>,
}

struct EpisodeState {
    tokens: Vec<u32>,
    valid_kv_tokens: usize,
    context_capacity: usize,
    session: Option<DecodeSession>,
    in_flight: bool,
}

// Safety: this only permits moving one generation loop to its dedicated OS
// thread. The worker owns cloned Arc handles until it exits, model and episode
// state are protected by mutexes, and the single-flight guard prevents two
// command encoders from using the same scratch/KV buffers concurrently. It does
// not permit general unsynchronized Metal access.
unsafe impl Send for MetalModelInner {}
unsafe impl Send for EpisodeState {}

struct InFlightGuard {
    episode_state: Arc<Mutex<EpisodeState>>,
}

impl InFlightGuard {
    fn new(episode_state: Arc<Mutex<EpisodeState>>) -> Self {
        Self { episode_state }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let mut state = self
            .episode_state
            .lock()
            .expect("metal episode mutex poisoned");
        state.in_flight = false;
    }
}

/// TimedGenerationStream keeps diagnostics out of the generated event stream.
/// Tokens, stops, and misses flow through `stream`; wall/GPU accounting arrives
/// separately when the worker completes.
pub struct TimedGenerationStream {
    stream: GenerationStream,
    timings: Receiver<MetalTimings>,
}

impl TimedGenerationStream {
    pub fn into_parts(self) -> (GenerationStream, Receiver<MetalTimings>) {
        (self.stream, self.timings)
    }
}

/// Light production signal: wall time, command-buffer GPU
/// time, and hot-path shape counters. Deep profiler data stays behind
/// `MetalProfile`.
#[derive(Debug, Clone)]
pub struct MetalTimings {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_wall: Duration,
    pub cold_start_to_first_token: Duration,
    pub hot_token_wall: Duration,
    pub hot_token_gpu_ns: u128,
    pub hot_token_count: usize,
    pub hot_command_buffers: usize,
    pub hot_compute_encoders: usize,
    pub hot_dispatches: usize,
    pub hot_scalar_param_buffers: usize,
    pub hot_readback_calls: usize,
    pub hot_readback_bytes: usize,
    pub expert_page_spills: usize,
}

impl fmt::Display for MetalTimings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\nengine timings:")?;
        writeln!(
            f,
            "- source: production path wall timers, no Metal counter samples"
        )?;
        writeln!(f, "- prompt tokens: {}", self.prompt_tokens)?;
        writeln!(f, "- generated tokens: {}", self.generated_tokens)?;
        writeln!(f, "- total wall: {}", format_duration(self.total_wall))?;
        writeln!(
            f,
            "- cold first token: {}",
            format_duration(self.cold_start_to_first_token)
        )?;
        writeln!(
            f,
            "- hot token wall: {} avg over {} tokens",
            format_average_duration(self.hot_token_wall, self.hot_token_count),
            self.hot_token_count
        )?;
        writeln!(
            f,
            "- hot token GPU: {} avg over {} tokens",
            format_duration_ns_average(self.hot_token_gpu_ns, self.hot_token_count),
            self.hot_token_count
        )?;
        let hot_gap = self
            .hot_token_wall
            .as_nanos()
            .saturating_sub(self.hot_token_gpu_ns);
        writeln!(
            f,
            "- hot token wall/GPU gap: {} avg over {} tokens",
            format_duration_ns_average(hot_gap, self.hot_token_count),
            self.hot_token_count
        )?;
        writeln!(
            f,
            "- decode throughput: {:.2} tok/s",
            tokens_per_second(self.generated_tokens, self.total_wall)
        )?;
        writeln!(
            f,
            "- hot-token throughput: {:.2} tok/s",
            tokens_per_second(self.hot_token_count, self.hot_token_wall)
        )?;
        writeln!(
            f,
            "- hot command buffers/token: {:.1}",
            average_count(self.hot_command_buffers, self.hot_token_count)
        )?;
        writeln!(
            f,
            "- hot compute encoders/token: {:.1}",
            average_count(self.hot_compute_encoders, self.hot_token_count)
        )?;
        writeln!(
            f,
            "- hot dispatches/token: {:.1}",
            average_count(self.hot_dispatches, self.hot_token_count)
        )?;
        writeln!(
            f,
            "- hot scalar param buffers/token: {:.1}",
            average_count(self.hot_scalar_param_buffers, self.hot_token_count)
        )?;
        writeln!(
            f,
            "- hot readbacks/token: {:.1} calls, {}",
            average_count(self.hot_readback_calls, self.hot_token_count),
            format_bytes_average(self.hot_readback_bytes, self.hot_token_count)
        )?;
        writeln!(
            f,
            "- experts carousel page-spills: {}",
            self.expert_page_spills
        )
    }
}

struct ResidentGeneration {
    generated_tokens: usize,
    finish: Option<Generated>,
}

struct TimingRecorder {
    prompt_tokens: usize,
    started: Instant,
    cold_start_to_first_token: Option<Duration>,
    hot_token_wall: Duration,
    hot_token_gpu_ns: u128,
    hot_token_count: usize,
    hot_command_buffers: usize,
    hot_compute_encoders: usize,
    hot_dispatches: usize,
    hot_scalar_param_buffers: usize,
    hot_readback_calls: usize,
    hot_readback_bytes: usize,
    expert_page_spills: usize,
}

impl TimingRecorder {
    fn start(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            started: Instant::now(),
            cold_start_to_first_token: None,
            hot_token_wall: Duration::ZERO,
            hot_token_gpu_ns: 0,
            hot_token_count: 0,
            hot_command_buffers: 0,
            hot_compute_encoders: 0,
            hot_dispatches: 0,
            hot_scalar_param_buffers: 0,
            hot_readback_calls: 0,
            hot_readback_bytes: 0,
            expert_page_spills: 0,
        }
    }

    fn record_cold_first_token(&mut self) {
        self.cold_start_to_first_token = Some(self.started.elapsed());
    }

    fn record_hot_token(
        &mut self,
        wall: Duration,
        gpu_ns: u128,
        counters: platform::BatchCounters,
        readback_calls: usize,
        readback_bytes: usize,
    ) {
        self.hot_token_wall += wall;
        self.hot_token_gpu_ns = self.hot_token_gpu_ns.saturating_add(gpu_ns);
        self.hot_token_count += 1;
        self.hot_command_buffers = self
            .hot_command_buffers
            .saturating_add(counters.command_buffers);
        self.hot_compute_encoders = self
            .hot_compute_encoders
            .saturating_add(counters.compute_encoders);
        self.hot_dispatches = self.hot_dispatches.saturating_add(counters.dispatches);
        self.hot_scalar_param_buffers = self
            .hot_scalar_param_buffers
            .saturating_add(counters.scalar_param_buffers);
        self.hot_readback_calls = self.hot_readback_calls.saturating_add(readback_calls);
        self.hot_readback_bytes = self.hot_readback_bytes.saturating_add(readback_bytes);
    }

    fn finish(self, generated_tokens: usize) -> MetalTimings {
        MetalTimings {
            prompt_tokens: self.prompt_tokens,
            generated_tokens,
            total_wall: self.started.elapsed(),
            cold_start_to_first_token: self
                .cold_start_to_first_token
                .unwrap_or_else(|| self.started.elapsed()),
            hot_token_wall: self.hot_token_wall,
            hot_token_gpu_ns: self.hot_token_gpu_ns,
            hot_token_count: self.hot_token_count,
            hot_command_buffers: self.hot_command_buffers,
            hot_compute_encoders: self.hot_compute_encoders,
            hot_dispatches: self.hot_dispatches,
            hot_scalar_param_buffers: self.hot_scalar_param_buffers,
            hot_readback_calls: self.hot_readback_calls,
            hot_readback_bytes: self.hot_readback_bytes,
            expert_page_spills: self.expert_page_spills,
        }
    }
}

impl MetalModel {
    pub fn load_canonical() -> Result<Self> {
        Self::load_canonical_with_layers(LAYERS)
    }

    pub fn load_canonical_with_layers(layers: usize) -> Result<Self> {
        if layers > LAYERS {
            return Err(eyre!(
                "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
            ));
        }

        let map = model_store::gguf::GgufMap::open_canonical()?;
        let source = model_store::gguf::GptOss20bGguf::new(&map)?;
        let ctx = MetalRuntime::new()?;
        let weights = GptOssWeights::load(&ctx, &source, layers)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(MetalModelInner {
                ctx,
                weights,
                layers,
            })),
        })
    }

    pub fn episode(&self, context_capacity: usize) -> Result<MetalEpisode> {
        if context_capacity == 0 {
            return Err(eyre!(
                "MetalEpisode context capacity must be greater than zero"
            ));
        }
        if context_capacity > MAX_RESIDENT_CONTEXT_TOKENS {
            return Err(eyre!(
                "MetalEpisode currently supports at most {MAX_RESIDENT_CONTEXT_TOKENS} context tokens, got {context_capacity}"
            ));
        }
        Ok(MetalEpisode {
            model: Arc::clone(&self.inner),
            state: Arc::new(Mutex::new(EpisodeState {
                tokens: Vec::new(),
                valid_kv_tokens: 0,
                context_capacity,
                session: None,
                in_flight: false,
            })),
        })
    }

    #[cfg(feature = "profile")]
    pub fn reset_profile(&self) {
        let model = self.inner.lock().expect("metal model mutex poisoned");
        model.ctx.reset_profile();
    }

    #[cfg(feature = "profile")]
    pub fn profile_report(&self) -> MetalProfile {
        let model = self.inner.lock().expect("metal model mutex poisoned");
        model.ctx.profile_report()
    }
}

impl MetalEpisode {
    pub fn token_count(&self) -> usize {
        let state = self.state.lock().expect("metal episode mutex poisoned");
        state.tokens.len()
    }

    pub fn splice_tokens(&self, range: Range<usize>, tokens: &[u32]) -> Result<()> {
        let mut state = self.state.lock().expect("metal episode mutex poisoned");
        if range.start > range.end || range.end > state.tokens.len() {
            return Err(eyre!(
                "token splice range {}..{} is invalid for tape length {}",
                range.start,
                range.end,
                state.tokens.len()
            ));
        }
        let start = range.start;
        let removed = range.end - range.start;
        let next_len = state
            .tokens
            .len()
            .checked_sub(removed)
            .and_then(|len| len.checked_add(tokens.len()))
            .ok_or_else(|| eyre!("token splice length overflow"))?;
        if next_len > state.context_capacity {
            return Err(eyre!(
                "token splice would produce {next_len} tokens, but episode capacity is {}",
                state.context_capacity
            ));
        }

        state.tokens.splice(range, tokens.iter().copied());
        state.valid_kv_tokens = state.valid_kv_tokens.min(start);
        Ok(())
    }

    pub fn generate(&self, max_new_tokens: usize) -> Result<GenerationStream> {
        let (sender, receiver) = mpsc::channel();
        self.generate_to(max_new_tokens, sender)?;
        Ok(receiver)
    }

    pub fn generate_timed(&self, max_new_tokens: usize) -> Result<TimedGenerationStream> {
        let (sender, receiver) = mpsc::channel();
        let timings = self.generate_timed_to(max_new_tokens, sender)?;
        Ok(TimedGenerationStream {
            stream: receiver,
            timings,
        })
    }

    pub fn generate_to(&self, max_new_tokens: usize, sender: Sender<Generated>) -> Result<()> {
        self.spawn_generation(max_new_tokens, sender, None)
    }

    pub fn generate_timed_to(
        &self,
        max_new_tokens: usize,
        sender: Sender<Generated>,
    ) -> Result<Receiver<MetalTimings>> {
        let (timings, receiver) = mpsc::channel();
        self.spawn_generation(max_new_tokens, sender, Some(timings))?;
        Ok(receiver)
    }

    fn spawn_generation(
        &self,
        max_new_tokens: usize,
        sender: Sender<Generated>,
        timings: Option<Sender<MetalTimings>>,
    ) -> Result<()> {
        let model = Arc::clone(&self.model);
        let state = Arc::clone(&self.state);
        {
            let mut state = state.lock().expect("metal episode mutex poisoned");
            if state.in_flight {
                return Err(eyre!("MetalEpisode already has an active generation"));
            }
            state.in_flight = true;
        }

        let worker_state = Arc::clone(&state);
        let worker = thread::Builder::new()
            .name("inference".to_string())
            .spawn(move || {
                let _guard = InFlightGuard::new(Arc::clone(&worker_state));
                let result = run_generation_for_episode(
                    model,
                    worker_state,
                    max_new_tokens,
                    &sender,
                    timings.as_ref(),
                );
                match result {
                    Ok(Some(finish)) => {
                        let _ = sender.send(finish);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        let _ = sender.send(Generated::Error(error.to_string()));
                    }
                }
            });
        if let Err(error) = worker {
            let mut state = state.lock().expect("metal episode mutex poisoned");
            state.in_flight = false;
            return Err(eyre!("spawn generation worker: {error}"));
        }
        Ok(())
    }
}

fn run_generation_for_episode(
    model: Arc<Mutex<MetalModelInner>>,
    state: Arc<Mutex<EpisodeState>>,
    max_new_tokens: usize,
    sender: &Sender<Generated>,
    timings: Option<&Sender<MetalTimings>>,
) -> Result<Option<Generated>> {
    let model = model.lock().expect("metal model mutex poisoned");
    let mut state = state.lock().expect("metal episode mutex poisoned");
    validate_generation(&state, max_new_tokens)?;
    if max_new_tokens == 0 {
        if let Some(timings) = timings {
            let _ = timings.send(empty_timings(state.tokens.len()));
        }
        return Ok(Some(Generated::LimitReached));
    }

    let context_capacity = state.context_capacity;
    let EpisodeState {
        tokens,
        valid_kv_tokens,
        session,
        ..
    } = &mut *state;
    let result = if let Some(timings) = timings {
        let (generated, report) = generate_resident_timed(
            &model.ctx,
            &model.weights,
            tokens,
            valid_kv_tokens,
            model.layers,
            max_new_tokens,
            context_capacity,
            session,
            sender,
        )?;
        let _ = timings.send(report);
        generated
    } else {
        generate_resident(
            &model.ctx,
            &model.weights,
            tokens,
            valid_kv_tokens,
            model.layers,
            max_new_tokens,
            context_capacity,
            session,
            sender,
        )?
    };
    Ok(result.finish)
}

fn validate_generation(state: &EpisodeState, max_new_tokens: usize) -> Result<()> {
    if state.tokens.is_empty() {
        return Err(eyre!("MetalEpisode has no prompt tokens"));
    }
    let required = state
        .tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| eyre!("context capacity overflow"))?;
    if required > state.context_capacity {
        return Err(eyre!(
            "generation needs {required} context tokens, but episode capacity is {}",
            state.context_capacity
        ));
    }
    Ok(())
}

fn empty_timings(prompt_tokens: usize) -> MetalTimings {
    MetalTimings {
        prompt_tokens,
        generated_tokens: 0,
        total_wall: Duration::ZERO,
        cold_start_to_first_token: Duration::ZERO,
        hot_token_wall: Duration::ZERO,
        hot_token_gpu_ns: 0,
        hot_token_count: 0,
        hot_command_buffers: 0,
        hot_compute_encoders: 0,
        hot_dispatches: 0,
        hot_scalar_param_buffers: 0,
        hot_readback_calls: 0,
        hot_readback_bytes: 0,
        expert_page_spills: 0,
    }
}

fn generate_resident(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    tokens: &mut Vec<u32>,
    valid_kv_tokens: &mut usize,
    layers: usize,
    max_new_tokens: usize,
    context_capacity: usize,
    session: &mut Option<DecodeSession>,
    sender: &Sender<Generated>,
) -> Result<ResidentGeneration> {
    #[cfg(feature = "profile")]
    {
        let started = Instant::now();
        let result = generate_resident_inner(
            ctx,
            weights,
            tokens,
            valid_kv_tokens,
            layers,
            max_new_tokens,
            context_capacity,
            session,
            sender,
            None,
        );
        ctx.record_profile(
            "phase.generate_resident",
            ProfileDelta {
                wall: started.elapsed(),
                ..ProfileDelta::default()
            },
        );
        result
    }
    #[cfg(not(feature = "profile"))]
    {
        generate_resident_inner(
            ctx,
            weights,
            tokens,
            valid_kv_tokens,
            layers,
            max_new_tokens,
            context_capacity,
            session,
            sender,
            None,
        )
    }
}

fn generate_resident_timed(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    tokens: &mut Vec<u32>,
    valid_kv_tokens: &mut usize,
    layers: usize,
    max_new_tokens: usize,
    context_capacity: usize,
    session: &mut Option<DecodeSession>,
    sender: &Sender<Generated>,
) -> Result<(ResidentGeneration, MetalTimings)> {
    let mut timings = TimingRecorder::start(tokens.len());
    #[cfg(feature = "profile")]
    let started = Instant::now();
    let generated = generate_resident_inner(
        ctx,
        weights,
        tokens,
        valid_kv_tokens,
        layers,
        max_new_tokens,
        context_capacity,
        session,
        sender,
        Some(&mut timings),
    )?;
    #[cfg(feature = "profile")]
    ctx.record_profile(
        "phase.generate_resident",
        ProfileDelta {
            wall: started.elapsed(),
            ..ProfileDelta::default()
        },
    );
    let timings = timings.finish(generated.generated_tokens);
    Ok((generated, timings))
}

#[allow(clippy::too_many_arguments)]
fn generate_resident_inner(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    tokens: &mut Vec<u32>,
    valid_kv_tokens: &mut usize,
    layers: usize,
    max_new_tokens: usize,
    context_capacity: usize,
    session: &mut Option<DecodeSession>,
    sender: &Sender<Generated>,
    mut timings: Option<&mut TimingRecorder>,
) -> Result<ResidentGeneration> {
    #[cfg(feature = "profile")]
    let infer_started = Instant::now();
    if tokens.is_empty() {
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
    if *valid_kv_tokens > tokens.len() {
        return Err(eyre!(
            "episode has {} valid KV tokens for a {} token tape",
            *valid_kv_tokens,
            tokens.len()
        ));
    }
    let min_context_tokens = tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| eyre!("resident decode context length overflow"))?;
    if min_context_tokens > context_capacity {
        return Err(eyre!(
            "resident decode needs {min_context_tokens} context tokens, but episode capacity is {context_capacity}"
        ));
    }
    let context_tokens = context_capacity;
    if context_tokens > MAX_RESIDENT_CONTEXT_TOKENS {
        return Err(eyre!(
            "resident decode currently supports at most {MAX_RESIDENT_CONTEXT_TOKENS} context tokens, got {context_tokens}"
        ));
    }
    #[cfg(feature = "profile")]
    ctx.reset_stage_profile(context_tokens);

    let session = DecodeSession::ensure(
        session,
        &ctx.platform,
        weights.lm_head.rows(),
        layers,
        context_tokens,
    )?;
    let suffix_start = if *valid_kv_tokens == tokens.len() {
        tokens.len() - 1
    } else {
        *valid_kv_tokens
    };
    let current = prepare_prompt_state(
        ctx,
        weights,
        session,
        layers,
        suffix_start,
        &tokens[suffix_start..],
        context_tokens,
    )?;
    *valid_kv_tokens = tokens.len();

    let mut generated_tokens = 0usize;
    let first_output_position = tokens.len();
    let hidden = current_hidden(&session.scratch, current);
    let mut sampled = sample_from_hidden(
        ctx,
        weights,
        &session.scratch,
        hidden,
        first_output_position,
    )?;
    if let Some(timings) = timings.as_mut() {
        timings.record_cold_first_token();
    }
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
    if is_gpt_oss_stop_token(sampled) {
        return Ok(ResidentGeneration {
            generated_tokens,
            finish: Some(Generated::Stop),
        });
    }
    if !send_generated_token(sender, sampled) {
        return Ok(ResidentGeneration {
            generated_tokens,
            finish: None,
        });
    }
    tokens.push(sampled);
    generated_tokens += 1;

    let mut decode_position = *valid_kv_tokens;
    while generated_tokens < max_new_tokens {
        let started = (timings.is_some() || cfg!(feature = "profile")).then(Instant::now);
        let scored = decode_and_score_next_token(
            ctx,
            weights,
            &session.scratch,
            &mut session.kv_cache,
            layers,
            decode_position,
            sampled,
        )?;
        let wall = started.map(|started| started.elapsed()).unwrap_or_default();
        if let Some(timings) = &mut timings {
            timings.record_hot_token(
                wall,
                scored.gpu_ns,
                scored.counters,
                scored.readback_calls,
                scored.readback_bytes,
            );
        }
        #[cfg(feature = "profile")]
        {
            record_hot_token_metric(ctx, wall, scored.gpu_ns);
        }

        *valid_kv_tokens += 1;
        decode_position += 1;

        sampled = scored.token;
        if is_gpt_oss_stop_token(sampled) {
            return Ok(ResidentGeneration {
                generated_tokens,
                finish: Some(Generated::Stop),
            });
        }
        if !send_generated_token(sender, sampled) {
            return Ok(ResidentGeneration {
                generated_tokens,
                finish: None,
            });
        }
        tokens.push(sampled);
        generated_tokens += 1;
    }

    Ok(ResidentGeneration {
        generated_tokens,
        finish: Some(Generated::LimitReached),
    })
}

fn hidden_pair(
    scratch: &GenerationScratch,
    current: PingPong,
) -> (&platform::F32VectorBuffer, &platform::F32VectorBuffer) {
    if current.is_ping() {
        (&scratch.hidden_ping, &scratch.hidden_pong)
    } else {
        (&scratch.hidden_pong, &scratch.hidden_ping)
    }
}

fn prefill_hidden_pair(
    scratch: &GenerationScratch,
    current: PingPong,
) -> (&platform::F32VectorBuffer, &platform::F32VectorBuffer) {
    if current.is_ping() {
        (&scratch.prefill_hidden_ping, &scratch.prefill_hidden_pong)
    } else {
        (&scratch.prefill_hidden_pong, &scratch.prefill_hidden_ping)
    }
}

fn current_hidden(scratch: &GenerationScratch, current: PingPong) -> &platform::F32VectorBuffer {
    if current.is_ping() {
        &scratch.hidden_ping
    } else {
        &scratch.hidden_pong
    }
}

struct ScoredToken {
    token: u32,
    gpu_ns: u128,
    counters: platform::BatchCounters,
    readback_calls: usize,
    readback_bytes: usize,
}

#[allow(clippy::too_many_arguments)]
fn prepare_prompt_state(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    session: &mut DecodeSession,
    layers: usize,
    start_position: usize,
    prompt_tokens: &[u32],
    context_tokens: usize,
) -> Result<PingPong> {
    let end_position = start_position
        .checked_add(prompt_tokens.len())
        .ok_or_else(|| eyre!("prompt prefill position overflow"))?;
    if prompt_tokens.is_empty() {
        return Err(eyre!("prompt prefill suffix is empty"));
    }
    if end_position > context_tokens {
        return Err(eyre!(
            "prompt prefill end position {end_position} exceeds context capacity {context_tokens}",
        ));
    }

    prefill_embeddings(ctx, weights, &session.scratch, prompt_tokens)?;
    prefill_layers(
        ctx,
        weights,
        &session.scratch,
        &mut session.kv_cache,
        layers,
        start_position,
        prompt_tokens.len(),
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_and_score_next_token(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    scratch: &GenerationScratch,
    kv_cache: &mut KvCache,
    layers: usize,
    position: usize,
    token: u32,
) -> Result<ScoredToken> {
    let output_position = position + 1;
    let batch = ctx.platform.begin_labeled_batch("generation.hot_token");
    batch.set_stage(GpuStage::Embedding);
    batch.embedding_lookup_q8_0_into(&weights.embed, token as usize, &scratch.hidden_ping)?;
    let mut current = PingPong::Ping;

    for layer in 0..layers {
        let (input, output) = hidden_pair(scratch, current);
        encode_decode_layer(
            &batch,
            scratch,
            kv_cache,
            weights.layer(layer)?,
            layer,
            position,
            input,
            output,
        )?;
        current = current.next();
    }

    let hidden = current_hidden(scratch, current);
    batch.set_stage(GpuStage::LmHead);
    batch.rms_norm_with_partials_into(
        hidden,
        &weights.final_norm,
        &scratch.rms_partials,
        &scratch.final_hidden,
    )?;
    batch.q8_0_matrix_top1_into(
        &weights.lm_head,
        &scratch.final_hidden,
        &scratch.lm_logits,
        &scratch.lm_top1_block_indices,
        &scratch.lm_top1_block_values,
        &scratch.lm_top_indices,
        &scratch.lm_top_values,
        &scratch.lm_sample_result,
    )?;
    let timing = finish_generation_batch(
        ctx,
        batch,
        stage_marker(output_position, TokenStage::HotToken),
    );
    let token = sample_from_greedy_result(ctx, scratch)?;

    Ok(ScoredToken {
        token,
        gpu_ns: timing.gpu_ns,
        counters: timing.counters,
        readback_calls: 1,
        readback_bytes: 4 * size_of::<u32>(),
    })
}

fn sample_from_hidden(
    ctx: &MetalRuntime,
    weights: &GptOssWeights,
    scratch: &GenerationScratch,
    hidden: &platform::F32VectorBuffer,
    output_position: usize,
) -> Result<u32> {
    let batch = ctx.platform.begin_labeled_batch("generation.score_greedy");
    batch.set_stage(GpuStage::LmHead);
    batch.rms_norm_with_partials_into(
        hidden,
        &weights.final_norm,
        &scratch.rms_partials,
        &scratch.final_hidden,
    )?;
    batch.q8_0_matrix_top1_into(
        &weights.lm_head,
        &scratch.final_hidden,
        &scratch.lm_logits,
        &scratch.lm_top1_block_indices,
        &scratch.lm_top1_block_values,
        &scratch.lm_top_indices,
        &scratch.lm_top_values,
        &scratch.lm_sample_result,
    )?;
    finish_generation_batch(
        ctx,
        batch,
        stage_marker(output_position, TokenStage::LmHead),
    );
    sample_from_greedy_result(ctx, scratch)
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
    batch.embedding_lookup_q8_0_batch_into(
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
    kv_cache: &mut KvCache,
    layers: usize,
    start_position: usize,
    token_count: usize,
) -> Result<PingPong> {
    let mut current = PingPong::Ping;

    for layer in 0..layers {
        let (input, output) = prefill_hidden_pair(scratch, current);
        prefill_layer(
            ctx,
            scratch,
            kv_cache,
            weights.layer(layer)?,
            layer,
            start_position,
            token_count,
            input,
            output,
        )?;
        current = current.next();
    }

    let final_prompt_hidden = if current.is_ping() {
        &scratch.prefill_hidden_ping
    } else {
        &scratch.prefill_hidden_pong
    };
    let batch = ctx
        .platform
        .begin_labeled_batch("generation.prefill_final_hidden");
    batch.copy_f32_slot_into(
        final_prompt_hidden,
        token_count - 1,
        HIDDEN_SIZE,
        &scratch.hidden_ping,
    )?;
    finish_generation_batch(ctx, batch, no_stage_marker());
    Ok(PingPong::Ping)
}

#[allow(clippy::too_many_arguments)]
fn prefill_layer(
    ctx: &MetalRuntime,
    scratch: &GenerationScratch,
    kv_cache: &mut KvCache,
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
    batch.q8_0_qkv_matvec_batch_into(
        &weights.attn.q.weight,
        &weights.attn.k.weight,
        &weights.attn.v.weight,
        &scratch.prefill_normed,
        &weights.attn.q.bias,
        &weights.attn.k.bias,
        &weights.attn.v.bias,
        &scratch.prefill_q,
        &scratch.prefill_k,
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
    batch.q8_0_matrix_matvec_add_batch_into(
        &weights.attn.o.weight,
        &scratch.prefill_attn,
        &weights.attn.o.bias,
        input,
        &scratch.prefill_residual,
        token_count,
    )?;
    batch.set_stage(GpuStage::RouterTop4);
    batch.rms_norm_batch_into(
        &scratch.prefill_residual,
        &weights.post_attn_norm,
        &scratch.prefill_router_input,
        token_count,
        HIDDEN_SIZE,
    )?;
    batch.f32_matrix_matvec_batch_into(
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
        batch.mxfp4_gguf_top4_gate_swiglu_batch_into(
            &weights.sparse_mlp.experts_carousel.gate,
            &weights.sparse_mlp.experts_carousel.up,
            &weights.sparse_mlp.experts_carousel.gate_bias,
            &weights.sparse_mlp.experts_carousel.up_bias,
            &scratch.prefill_router_input,
            &scratch.prefill_router_indices,
            &scratch.prefill_expert_acts_packed,
            row_offset,
            rows,
        )?;
        batch.set_stage(GpuStage::ExpertsDown);
        batch.mxfp4_gguf_top4_down_weighted_batch_into(
            &weights.sparse_mlp.experts_carousel.down,
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

fn sample_from_greedy_result(ctx: &MetalRuntime, scratch: &GenerationScratch) -> Result<u32> {
    let result = ctx
        .platform
        .read_u32_array::<4>(&scratch.lm_sample_result)?;
    let token = result[0];
    let status = result[2];
    if status != 0 {
        return Err(eyre!("greedy sample result returned status {status}"));
    }
    Ok(token)
}

fn send_generated_token(sender: &Sender<Generated>, token: u32) -> bool {
    sender.send(Generated::Token(token)).is_ok()
}

fn is_gpt_oss_stop_token(token: u32) -> bool {
    GPT_OSS_STOP_TOKENS.contains(&token)
}

fn encode_decode_layer(
    batch: &platform::MetalBatch<'_>,
    scratch: &GenerationScratch,
    kv_cache: &mut KvCache,
    weights: &GptOssLayerWeights,
    layer: usize,
    position: usize,
    input: &platform::F32VectorBuffer,
    output: &platform::F32VectorBuffer,
) -> Result<()> {
    if position >= kv_cache.capacity {
        return Err(eyre!(
            "decode position {position} exceeds KV capacity {}",
            kv_cache.capacity
        ));
    }

    let layer_cache = kv_cache.layer(layer)?;
    batch.set_stage(GpuStage::InputNormQkv);
    batch.rms_norm_with_partials_into(
        input,
        &weights.input_norm,
        &scratch.rms_partials,
        &scratch.normed,
    )?;
    batch.q8_0_qkv_matvec_into(
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
    batch.qk_rope_write_cache_into(
        &scratch.q,
        &scratch.k,
        &scratch.v,
        &scratch.q_rope,
        &layer_cache.k,
        &layer_cache.v,
        position,
    )?;
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
    batch.q8_0_matrix_matvec_add_into(
        &weights.attn.o.weight,
        &scratch.attn,
        &weights.attn.o.bias,
        input,
        &scratch.residual,
    )?;
    batch.set_stage(GpuStage::RouterTop4);
    batch.rms_norm_with_partials_into(
        &scratch.residual,
        &weights.post_attn_norm,
        &scratch.rms_partials,
        &scratch.router_input,
    )?;
    batch.f32_matrix_matvec_into(
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
    batch.mxfp4_gguf_top4_gate_swiglu_into(
        &weights.sparse_mlp.experts_carousel.gate,
        &weights.sparse_mlp.experts_carousel.up,
        &weights.sparse_mlp.experts_carousel.gate_bias,
        &weights.sparse_mlp.experts_carousel.up_bias,
        &scratch.router_input,
        &scratch.router_indices,
        &scratch.expert_acts_packed,
    )?;
    batch.set_stage(GpuStage::ExpertsDown);
    batch.mxfp4_gguf_top4_down_slots_into(
        &weights.sparse_mlp.experts_carousel.down,
        &weights.sparse_mlp.experts_carousel.down_bias,
        &scratch.expert_acts_packed,
        &scratch.router_indices,
        &scratch.expert_downs_packed,
    )?;
    batch.weighted_sum4_residual_into(
        &scratch.expert_downs_packed,
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
) -> platform::BatchTiming {
    let timing = batch.finish();
    #[cfg(not(feature = "profile"))]
    {
        let _ = ctx;
        let _ = stage;
        timing
    }
    #[cfg(feature = "profile")]
    {
        let mut timing = timing;
        let gpu_ns = timing.gpu_ns;
        ctx.record_profile(
            "phase.generate_resident",
            ProfileDelta {
                command_buffers: 1,
                ..ProfileDelta::default()
            },
        );
        if let Some((token_position, stage)) = stage {
            ctx.record_token_stage(token_position, stage, gpu_ns);
            ctx.record_gpu_stages(token_position, std::mem::take(&mut timing.gpu_stages));
        }
        timing
    }
}

fn no_stage_marker() -> StageMarker {
    #[cfg(feature = "profile")]
    {
        None
    }
    #[cfg(not(feature = "profile"))]
    {}
}

fn format_average_duration(duration: Duration, count: usize) -> String {
    if count == 0 {
        return "n/a".to_string();
    }
    format_duration(duration / count as u32)
}

fn format_duration_ns_average(ns: u128, count: usize) -> String {
    if count == 0 {
        return "n/a".to_string();
    }
    format_duration(duration_from_ns_lossy(ns / count as u128))
}

fn tokens_per_second(tokens: usize, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if tokens == 0 || seconds == 0.0 {
        return 0.0;
    }
    tokens as f64 / seconds
}

fn average_count(total: usize, count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    total as f64 / count as f64
}

fn format_bytes_average(total: usize, count: usize) -> String {
    if count == 0 {
        return "n/a".to_string();
    }
    format_bytes(total / count)
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

fn format_duration(duration: Duration) -> String {
    let ns = duration.as_nanos();
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

fn duration_from_ns_lossy(ns: u128) -> Duration {
    Duration::from_nanos(ns.min(u64::MAX as u128) as u64)
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
