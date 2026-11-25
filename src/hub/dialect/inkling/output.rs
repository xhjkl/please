//! Decode Inkling messages incrementally, recognizing framing only by token ID.
use super::{CALL, END, MODEL, SYSTEM, TEXT, THINKING, TOOL, USER};
use crate::hub::dialect::{Delta, OutputParser, ToolCall, piece::special};
use eyre::{Result, WrapErr, bail, eyre};
use oco::Tokenizer;
use serde_json::Value;

#[derive(Clone, Copy)]
enum Marker {
    Model,
    Text,
    Thinking,
    Invoke,
    End,
    Reset,
}

pub struct Parser<'a> {
    tokenizer: &'a Tokenizer,
    decoder: oco::Decoder<'a>,
    markers: [(u32, Marker); 8],
    stream: Stream,
}

impl<'a> Parser<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Result<Self> {
        Ok(Self {
            tokenizer,
            decoder: tokenizer.decoder(),
            markers: [
                (special(tokenizer, MODEL)?, Marker::Model),
                (special(tokenizer, TEXT)?, Marker::Text),
                (special(tokenizer, THINKING)?, Marker::Thinking),
                (special(tokenizer, CALL)?, Marker::Invoke),
                (special(tokenizer, END)?, Marker::End),
                (special(tokenizer, SYSTEM)?, Marker::Reset),
                (special(tokenizer, USER)?, Marker::Reset),
                (special(tokenizer, TOOL)?, Marker::Reset),
            ],
            stream: Stream::default(),
        })
    }
}

impl OutputParser for Parser<'_> {
    fn push_token(&mut self, token: u32) -> Result<Vec<Delta>> {
        let marker = self
            .markers
            .iter()
            .find_map(|&(id, marker)| (token == id).then_some(marker))
            .or_else(|| (self.tokenizer.eos_token() == Some(token)).then_some(Marker::Reset));
        if let Some(marker) = marker {
            // Finish pending UTF-8 in the message it came from.
            let decoder = std::mem::replace(&mut self.decoder, self.tokenizer.decoder());
            let deltas = self.stream.text(&decoder.finish()).into_iter().collect();
            self.stream.marker(marker);
            return Ok(deltas);
        }
        let text = self.decoder.push(token)?;
        Ok(self.stream.text(&text).into_iter().collect())
    }

    fn finish(&mut self) -> Result<Vec<ToolCall>> {
        self.stream.finish()
    }
}

#[derive(Default)]
enum Part {
    #[default]
    BetweenMessages,
    Header(String),
    Thinking,
    Answer,
    Call {
        name: String,
        payload: String,
    },
}

#[derive(Default)]
struct Stream {
    part: Part,
    calls: Vec<Result<ToolCall>>,
}

impl Stream {
    fn text(&mut self, text: &str) -> Option<Delta> {
        if text.is_empty() {
            return None;
        }
        match &mut self.part {
            Part::BetweenMessages => {}
            Part::Header(name) => name.push_str(text),
            Part::Thinking => return Some(Delta::Thinking(text.to_owned())),
            Part::Answer => return Some(Delta::Answer(text.to_owned())),
            Part::Call { payload, .. } => payload.push_str(text),
        }
        None
    }

    fn marker(&mut self, marker: Marker) {
        let part = std::mem::take(&mut self.part);
        let part = match part {
            Part::Call { name, payload } => {
                let call = match marker {
                    Marker::End => parse_call(&name, &payload),
                    _ => Err(eyre!("unterminated Inkling tool call {name}")),
                };
                self.calls.push(call);
                Part::BetweenMessages
            }
            part => part,
        };
        self.part = match (marker, part) {
            (Marker::Model, _) => Part::Header(String::new()),
            (Marker::Text, Part::Header(_)) => Part::Answer,
            (Marker::Thinking, Part::Header(_)) => Part::Thinking,
            (Marker::Invoke, Part::Header(name)) if !name.is_empty() => Part::Call {
                name,
                payload: String::new(),
            },
            (Marker::Invoke, _) => {
                self.calls
                    .push(Err(eyre!("missing Inkling tool call name")));
                Part::BetweenMessages
            }
            _ => Part::BetweenMessages,
        };
    }

    fn finish(&mut self) -> Result<Vec<ToolCall>> {
        self.marker(Marker::Reset);
        let calls = std::mem::take(&mut self.calls);
        calls.into_iter().collect()
    }
}

fn parse_call(name: &str, payload: &str) -> Result<ToolCall> {
    let mut payload: Value = serde_json::from_str(payload).wrap_err("invalid Inkling tool JSON")?;
    if payload["name"].as_str() != Some(name) {
        bail!("Inkling tool call name differs from its message header {name}");
    }
    let arguments = payload["args"].take();
    if !arguments.is_object() {
        bail!("Inkling tool call {name} requires a JSON argument object");
    }
    Ok(ToolCall {
        name: name.to_owned(),
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_thinking_and_answer_and_collects_multiple_native_calls() -> Result<()> {
        let mut stream = Stream::default();
        stream.marker(Marker::Model);
        stream.marker(Marker::Thinking);
        let text = stream.text(" plan");
        assert_eq!(text, Some(Delta::Thinking(" plan".into())));
        stream.marker(Marker::End);
        stream.marker(Marker::Model);
        stream.marker(Marker::Text);
        let text = stream.text("Working <|end_message|>");
        assert_eq!(text, Some(Delta::Answer("Working <|end_message|>".into())));
        stream.marker(Marker::End);
        for path in ["one.txt", "two.txt"] {
            stream.marker(Marker::Model);
            stream.text("read_");
            stream.text("file");
            stream.marker(Marker::Invoke);
            let text = stream.text(r#"{"name":"read_file","args":{"path":""#);
            assert!(text.is_none());
            stream.text(path);
            stream.text(r#""}}"#);
            stream.marker(Marker::End);
        }
        let calls = stream.finish()?;
        let paths: Vec<_> = calls.iter().map(|call| &call.arguments["path"]).collect();
        assert_eq!(paths, ["one.txt", "two.txt"]);
        Ok(())
    }

    #[test]
    fn fails_the_step_if_a_call_is_truncated_or_misframed() {
        for (payload, end) in [
            (r#"{"name":"read_file","args":{}}"#, Marker::Reset),
            (r#"{"name":"run_command","args":{}}"#, Marker::End),
            (r#"{"name":"read_file","args":[]}"#, Marker::End),
        ] {
            let mut stream = Stream::default();
            stream.marker(Marker::Model);
            stream.text("read_file");
            stream.marker(Marker::Invoke);
            stream.text(payload);
            stream.marker(end);
            let result = stream.finish();
            assert!(result.is_err(), "{payload}");
        }
    }
}
