//! Hub-private model dialect identity; no dialect is carried over the probe protocol.

/// Conversation format a checkpoint was trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Dialect {
    Inkling,
    Qwenish,
    Harmony,
}

impl Dialect {
    /// Select the dialect from the checkpoint architecture, never its filename.
    pub(super) fn from_architecture(architecture: &str) -> eyre::Result<Self> {
        match architecture {
            "inkling2_loop" => Ok(Self::Inkling),
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
mod inkling;
mod piece;
pub(super) mod prompting;
mod qwenish;
#[cfg(test)]
mod tests;
pub(super) use adapter::{Adapter, Delta, FunctionCall, OutputParser, Rendered};
