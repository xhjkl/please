//! Model dialects: how each family of checkpoints frames a prompt and spells its reply.
//! Hub-private; no dialect is carried over the probe protocol.
use crate::protocol::{Message, Setting};
use eyre::Result;
use oco::Tokenizer;

/// Prompt format a checkpoint was trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Inkling,
    Qwenish,
    Harmony,
}

/// One response being decoded in its checkpoint's dialect: model tokens in, dialect-neutral
/// content and tool calls out.
pub enum OutputParser<'a> {
    Inkling(inkling::Parser<'a>),
    Qwenish(qwenish::Parser<'a>),
    Harmony(harmony::Parser<'a>),
}

impl OutputParser<'_> {
    /// Consume one generated token and return newly visible text.
    pub fn push_token(&mut self, token: u32) -> Result<Vec<Delta>> {
        match self {
            Self::Inkling(parser) => parser.push_token(token),
            Self::Qwenish(parser) => parser.push_token(token),
            Self::Harmony(parser) => parser.push_token(token),
        }
    }

    /// Return the tool calls of the finished response.
    pub fn finish(&mut self) -> Result<Vec<FunctionCall>> {
        match self {
            Self::Inkling(parser) => parser.finish(),
            Self::Qwenish(parser) => parser.finish(),
            Self::Harmony(parser) => parser.finish(),
        }
    }
}

impl Dialect {
    /// Select the dialect from the checkpoint architecture, never its filename.
    pub fn from_architecture(architecture: &str) -> Result<Self> {
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
    pub fn render(
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
    pub fn output_parser(self, tokenizer: &Tokenizer) -> Result<OutputParser<'_>> {
        Ok(match self {
            Self::Inkling => OutputParser::Inkling(inkling::Parser::new(tokenizer)?),
            Self::Qwenish => OutputParser::Qwenish(qwenish::Parser::new(tokenizer)?),
            Self::Harmony => OutputParser::Harmony(harmony::Parser::new(tokenizer)?),
        })
    }

    /// Whether generation ends at this token.
    pub fn is_stop_token(self, tokenizer: &Tokenizer, token: u32) -> bool {
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
pub use prompt::{Delta, FunctionCall, Rendered};
