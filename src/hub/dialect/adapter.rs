//! Translate neutral conversations and generated tokens at the hub's dialect boundary.
use super::{Dialect, conversation, harmony, inkling, qwenish};
use crate::protocol::{Message, Setting};
use eyre::Result;
use oco::Tokenizer;

use conversation::ConversationMessage;
pub use conversation::{Delta, Rendered, ToolCall};

/// Stateful codec selected once when the hub loads its checkpoint.
pub trait Adapter: Send + Sync {
    /// Word the setting and conversation, fit command results to `tail_budget`, then encode
    /// them in the selected dialect.
    fn render(
        &self,
        tokenizer: &Tokenizer,
        setting: &Setting,
        history: &[Message],
        tail_budget: usize,
    ) -> Result<Rendered> {
        let mut messages = conversation::prepare(setting, history)?;
        conversation::fit_command_results(&mut messages, tail_budget, |text| {
            Ok(tokenizer.encode(text)?.len())
        })?;
        self.encode(tokenizer, &messages)
    }

    /// Encode prepared history in this dialect.
    fn encode(&self, tokenizer: &Tokenizer, messages: &[ConversationMessage]) -> Result<Rendered>;
    /// Start decoding one response that continues the rendered prompt.
    fn output_parser<'a>(&self, tokenizer: &'a Tokenizer) -> Result<Box<dyn OutputParser + 'a>>;
    /// Whether generation ends at this token.
    fn is_stop_token(&self, tokenizer: &Tokenizer, token: u32) -> bool;
}

/// Incremental conversion from model tokens to dialect-neutral content and tool calls.
pub trait OutputParser: Send {
    /// Consume one generated token and return newly visible text.
    fn push_token(&mut self, token: u32) -> Result<Vec<Delta>>;
    /// Return the tool calls of the finished response.
    fn finish(&mut self) -> Result<Vec<ToolCall>>;
}

impl Dialect {
    /// Select the codec without exposing it to the probe or inference loop.
    pub(in crate::hub) fn adapter(self) -> Box<dyn Adapter> {
        match self {
            Self::Inkling => Box::new(inkling::Adapter),
            Self::Qwenish => Box::new(qwenish::Adapter),
            Self::Harmony => Box::new(harmony::Adapter),
        }
    }
}
