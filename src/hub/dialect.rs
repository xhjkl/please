//! Model dialects: how each family of checkpoints frames a prompt and spells its reply.
//! Hub-private; no dialect is carried over the probe protocol.
use crate::protocol::{Message, Setting, ToolCall};
use crate::tools::spec::Spec;
use eyre::Result;
use oco::Tokenizer;
use serde_json::Value;

/// Prompt format a checkpoint was trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Inkling,
    Qwenish,
    Harmony,
}

/// A call as dialects read and write it, in decoded output and replayed history alike: a name
/// and JSON arguments, with no ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
}

impl FunctionCall {
    /// This call as the protocol carries it and a room keeps it, under `id`.
    pub fn with_id(&self, id: u64) -> ToolCall {
        ToolCall {
            id,
            name: self.name.clone(),
            arguments: self.arguments.to_string(),
        }
    }
}

/// Encoded prompt and how many of its leading tokens make its cacheable prefix: the part
/// that is the same for every invocation, at this effort where the family states it there,
/// which a hub may prefill once and save.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub tokens: Vec<u32>,
    pub cacheable: usize,
}

/// Newly visible generated text, by the section it belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delta {
    Answer(String),
    Thinking(String),
}

/// Every tool the model may call, the probe's and the hub's own, by name.
pub fn tool_specs() -> Vec<Spec> {
    let probe = crate::tools::probe_tools().into_values();
    let probe = probe.map(|tool| tool.spec);
    let mut specs: Vec<_> = probe.chain([crate::hub::transcript::RECALL_SPEC]).collect();
    specs.sort_by_key(|spec| spec.name);
    specs
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
