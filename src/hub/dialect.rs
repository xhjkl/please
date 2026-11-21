//! Hub-private model dialect identity; no dialect is carried over the probe protocol.

/// Conversation format a checkpoint was trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Dialect {
    Qwenish,
    Harmony,
}

impl Dialect {
    /// Select the dialect from the checkpoint architecture, never its filename.
    pub(super) fn from_architecture(architecture: &str) -> eyre::Result<Self> {
        match architecture {
            "qwen35" => Ok(Self::Qwenish),
            "gpt-oss" => Ok(Self::Harmony),
            _ => Err(eyre::eyre!(
                "unsupported model architecture: {architecture}"
            )),
        }
    }
}

mod adapter;
mod conversation;
mod harmony;
mod piece;
mod prompting;
mod qwenish;
pub(super) use adapter::{Adapter, Delta, OutputParser, Rendered, ToolCall};
