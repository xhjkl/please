//! Model dialects: how each family of checkpoints frames a prompt and spells its reply.
//! Hub-private; no dialect is carried over the probe protocol.
use crate::protocol::{Message, Setting};
use eyre::Result;
use oco::Tokenizer;

/// Prompt format a checkpoint was trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Dialect {
    Inkling,
    Qwenish,
    Harmony,
}

/// Incremental conversion from model tokens to dialect-neutral content and tool calls.
pub(super) trait OutputParser: Send {
    /// Consume one generated token and return newly visible text.
    fn push_token(&mut self, token: u32) -> Result<Vec<Delta>>;
    /// Return the tool calls of the finished response.
    fn finish(&mut self) -> Result<Vec<FunctionCall>>;
}

impl Dialect {
    /// Select the dialect from the checkpoint architecture, never its filename.
    pub(super) fn from_architecture(architecture: &str) -> Result<Self> {
        match architecture {
            "inkling2_loop" => Ok(Self::Inkling),
            "qwen35" => Ok(Self::Qwenish),
            "gpt-oss" => Ok(Self::Harmony),
            _ => Err(eyre::eyre!(
                "unsupported model architecture: {architecture}"
            )),
        }
    }

    /// Word the setting and history, fit command results to `tail_budget`, then encode them
    /// in this dialect.
    pub(super) fn render(
        self,
        tokenizer: &Tokenizer,
        setting: &Setting,
        history: &[Message],
        tail_budget: usize,
    ) -> Result<Rendered> {
        let preamble = prompt::Preamble::new(setting);
        let mut messages = prompt::prepare(setting, history)?;
        prompt::fit_command_results(&mut messages, tail_budget, |text| {
            Ok(tokenizer.encode(text)?.len())
        })?;
        match self {
            Self::Inkling => inkling::encode(tokenizer, &preamble, &messages),
            Self::Qwenish => qwenish::encode(tokenizer, &preamble, &messages),
            Self::Harmony => harmony::encode(tokenizer, &preamble, &messages),
        }
    }

    /// Start decoding one response that continues the rendered prompt.
    pub(super) fn output_parser<'a>(
        self,
        tokenizer: &'a Tokenizer,
    ) -> Result<Box<dyn OutputParser + 'a>> {
        match self {
            Self::Inkling => inkling::output_parser(tokenizer),
            Self::Qwenish => qwenish::output_parser(tokenizer),
            Self::Harmony => harmony::output_parser(tokenizer),
        }
    }

    /// Whether generation ends at this token.
    pub(super) fn is_stop_token(self, tokenizer: &Tokenizer, token: u32) -> bool {
        match self {
            Self::Inkling => inkling::is_stop_token(tokenizer, token),
            Self::Qwenish => qwenish::is_stop_token(tokenizer, token),
            Self::Harmony => harmony::is_stop_token(tokenizer, token),
        }
    }
}

mod harmony;
mod inkling;
mod piece;
mod prompt;
mod qwenish;
#[cfg(test)]
mod tests;
pub(super) mod wording;
pub(super) use prompt::{Delta, FunctionCall, Rendered};
