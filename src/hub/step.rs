//! One step of a turn: a single response from the model, generated, streamed, and parsed.
use super::dialect::{Delta, FunctionCall};
use super::inference::{self, Engine};
use crate::protocol::{Frame, Message, Setting, write_frame};
use eyre::Result;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

/// What the model produced in one response.
pub struct Step {
    pub thinking: String,
    pub answer: String,
    pub calls: Result<Vec<FunctionCall>>,
    /// The response ended by itself rather than at its token limit.
    pub complete: bool,
}

/// Generate one response to `history`, streaming its text to `sink` when given.
pub async fn generate(
    mut sink: Option<&mut (impl AsyncWriteExt + Unpin)>,
    engine: &Arc<Engine>,
    owner: u64,
    setting: &Setting,
    history: &[Message],
    max_tokens: Option<usize>,
) -> Result<Step> {
    let mut parser = engine.output_parser()?;
    let (sender, mut generated) = tokio::sync::mpsc::unbounded_channel();
    let (engine, setting, history) = (engine.clone(), setting.clone(), history.to_vec());
    let task = tokio::spawn(async move {
        inference::generate(&engine, owner, &setting, &history, sender, max_tokens).await
    });
    let (mut thinking, mut answer) = (String::new(), String::new());
    while let Some(token) = generated.recv().await {
        for delta in parser.push_token(token)? {
            let frame = match delta {
                Delta::Answer(text) => {
                    answer.push_str(&text);
                    Frame::Answer(text)
                }
                Delta::Thinking(text) => {
                    thinking.push_str(&text);
                    Frame::Thinking(text)
                }
            };
            if let Some(sink) = sink.as_mut() {
                write_frame(sink, &frame).await?;
            }
        }
    }
    let generation = task.await??;
    let calls = parser.finish();
    let complete = generation.is_some_and(|generation| generation.complete);
    Ok(Step {
        thinking,
        answer,
        calls,
        complete,
    })
}

/// A step's reasoning and answer as the messages a room keeps. A response of calls alone
/// still opens its own step, with empty reasoning: that is what tells it from the step before.
pub fn prose(thinking: String, answer: String) -> Vec<Message> {
    let mut kept = Vec::new();
    if !thinking.is_empty() || answer.is_empty() {
        kept.push(Message::Thinking(thinking));
    }
    if !answer.is_empty() {
        kept.push(Message::Answer(answer));
    }
    kept
}
