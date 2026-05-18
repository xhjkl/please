use serde::{Deserialize, Serialize};
use std::sync::mpsc::Receiver;

pub mod backend_metal;
pub mod harmony_adapter;
pub mod model_store;

pub use backend_metal::{MetalModel, MetalTimings, TimedGenerationStream};
pub use harmony_adapter::{HarmonyAdapter, Message, Role};

// The inference core owns a token tape, a KV cache, and generated token ids.
// Prompt rendering and text decoding stay above that boundary, which keeps the
// Metal path numeric while letting app layers compose Harmony, ASCII, or other
// codecs over the same generation stream.
//
// ```ignore
// let harmony = HarmonyAdapter::gpt_oss()?;
// let tokens = harmony.render_completion_tokens(&messages)?;
// let model = MetalModel::load_canonical()?;
// let episode = model.episode(tokens.len() + 32)?;
// episode.splice_tokens(0..0, &tokens)?;
// for event in episode.generate(32)? { /* compose */ }
// ```
pub type GenerationStream = Receiver<Generated>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Generated {
    Token(u32),
    Stop,
    LimitReached,
    ExpertMiss { layer: u16, expert: u16 },
    Error(String),
}
