//! Translate neutral probe history and generated tokens at the hub's dialect boundary.
use super::{Dialect, conversation, harmony, qwenish};
use crate::protocol::Message;
use eyre::Result;
use oco::Tokenizer;

use conversation::ConversationMessage;
pub use conversation::{Delta, Rendered, ToolCall};

/// Stateful codec selected once when the hub loads its checkpoint.
pub trait Adapter: Send + Sync {
    /// Expand application templates and tool exchanges, then encode the selected dialect.
    fn render(
        &self,
        tokenizer: &Tokenizer,
        history: &[Message],
        tail_budget: usize,
    ) -> Result<Rendered> {
        let messages = conversation::prepare(history)?;
        let mut rendered = self.encode(tokenizer, messages, tail_budget)?;
        // Without session context the leading messages are whatever the probe sent, not
        // static instructions, so nothing may be cached for other invocations.
        if !matches!(history.first(), Some(Message::Context(_))) {
            rendered.preamble = 0;
        }
        Ok(rendered)
    }

    /// Encode prepared history, trimming the latest command result to fit `tail_budget`.
    fn encode(
        &self,
        tokenizer: &Tokenizer,
        messages: Vec<ConversationMessage>,
        tail_budget: usize,
    ) -> Result<Rendered>;
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
            Self::Qwenish => Box::new(qwenish::Adapter),
            Self::Harmony => Box::new(harmony::Adapter),
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
