//! Streaming split of generated Harmony into thinking, answer, and tool calls.
//!
//! The reference parser's behaviour in its lenient mode, resuming after the prompt's open
//! assistant header: structure comes from control tokens by ID, a header is read from its
//! spelled-out text the way the reference reads it, and content arrives as decoded deltas.
//! Only invalid UTF-8 is treated differently, by the checkpoint's decoder: it is replaced in
//! headers instead of failing them, bytes after a replacement show at once instead of one
//! token later, and an unfinished character before a marker stays unshown in its message.
use super::super::conversation::{Delta, FunctionCall};
use super::super::piece::special;
use super::{CALL, CHANNEL, CONSTRAIN, END, MESSAGE, RETURN, START};
use eyre::{Result, bail, eyre};
use oco::Tokenizer;

/// Token-level front end: recognises control tokens by ID and decodes the rest.
pub struct OutputParser<'a> {
    tokenizer: &'a Tokenizer,
    decoder: oco::Decoder<'a>,
    markers: Vec<(u32, Marker)>,
    stream: OutputStream,
}

impl<'a> OutputParser<'a> {
    /// Resolve every marker up front, so a checkpoint without one fails before generating
    /// rather than streaming its markers as text.
    pub fn new(tokenizer: &'a Tokenizer) -> Result<Self> {
        let mut markers = Vec::new();
        for marker in Marker::ALL {
            markers.push((special(tokenizer, marker.spelling())?, marker));
        }
        let start = markers[0].0;
        Ok(Self {
            tokenizer,
            decoder: tokenizer.decoder(),
            markers,
            stream: OutputStream::new(start),
        })
    }

    /// Hand the stream whatever an unfinished character left in the decoder, as the reference
    /// keeps undecodable bytes in the message they ended.
    fn flush(&mut self) {
        let decoder = std::mem::replace(&mut self.decoder, self.tokenizer.decoder());
        self.stream.tail(&decoder.finish());
    }
}

impl super::DialectOutputParser for OutputParser<'_> {
    fn push_token(&mut self, token: u32) -> Result<Vec<Delta>> {
        let marker = self.markers.iter().find(|(id, _)| *id == token);
        if let Some(&(_, marker)) = marker {
            self.flush();
            return self.stream.push(token, Event::Marker(marker));
        }
        let text = self.decoder.push(token)?;
        self.stream.push(token, Event::Text(&text))
    }

    fn finish(&mut self) -> Result<Vec<FunctionCall>> {
        self.flush();
        self.stream.finish()
    }
}

/// A control token that frames Harmony messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Marker {
    /// Opens a message.
    Start,
    /// Introduces the header's channel.
    Channel,
    /// Introduces a constrained content type.
    Constrain,
    /// Ends the header.
    Message,
    /// Ends a message; another must start.
    End,
    /// Ends a tool call; generation stops.
    Call,
    /// Ends the final answer; generation stops.
    Return,
}

impl Marker {
    /// Every marker, `Start` first.
    pub(super) const ALL: [Self; 7] = [
        Self::Start,
        Self::Channel,
        Self::Constrain,
        Self::Message,
        Self::End,
        Self::Call,
        Self::Return,
    ];

    /// The control token's spelling in the vocabulary, and in headers read as text.
    pub(super) fn spelling(self) -> &'static str {
        match self {
            Self::Start => START,
            Self::Channel => CHANNEL,
            Self::Constrain => CONSTRAIN,
            Self::Message => MESSAGE,
            Self::End => END,
            Self::Call => CALL,
            Self::Return => RETURN,
        }
    }

    /// Whether the token closes the message being read.
    fn ends(self) -> bool {
        matches!(self, Self::End | Self::Call | Self::Return)
    }
}

/// One generated token: decoded content, which may be empty mid-character, or a marker.
#[derive(Debug, Clone, Copy)]
pub(super) enum Event<'a> {
    Text(&'a str),
    Marker(Marker),
}

/// What the stream is reading.
#[derive(Debug)]
enum State {
    /// Between messages, where only `<|start|>` may come.
    Idle,
    /// A header, spelled out with its markers; `first` is the prompt's own assistant header.
    Header { text: String, first: bool },
    /// A message's content after its parsed header, with the text decoded so far.
    Content { header: Header, text: String },
}

/// The header fields please acts on.
#[derive(Debug, Clone, Default)]
struct Header {
    /// Only the assistant's messages can call tools.
    assistant: bool,
    /// A tool the message calls, which keeps its content off the screen.
    recipient: Option<String>,
    /// The analysis channel streams as thinking; every other channel as answer.
    channel: Option<String>,
}

/// Tokenizer-free Harmony message state machine.
#[derive(Debug)]
pub(super) struct OutputStream {
    /// The `<|start|>` ID, named when anything else arrives between messages.
    start: u32,
    state: State,
    /// Messages completed so far, which tells consecutive analysis messages apart.
    completed: usize,
    /// Index of the last message that streamed thinking, so later analysis starts a paragraph.
    thinking: Option<usize>,
    /// Completed assistant messages addressed to a tool, with their raw arguments.
    calls: Vec<(String, String)>,
}

impl OutputStream {
    /// Resume inside the assistant header that the prompt opened.
    pub(super) fn new(start: u32) -> Self {
        Self {
            start,
            state: State::Header {
                text: String::new(),
                first: true,
            },
            completed: 0,
            thinking: None,
            calls: Vec::new(),
        }
    }

    /// Consume one token and return the content it made visible.
    pub(super) fn push(&mut self, token: u32, event: Event) -> Result<Vec<Delta>> {
        match (&mut self.state, event) {
            (State::Idle, Event::Marker(Marker::Start)) => {
                self.state = State::Header {
                    text: String::new(),
                    first: false,
                };
            }
            (State::Idle, _) => {
                let start = self.start;
                bail!("Unexpected token {token} while expecting start token {start}");
            }
            (State::Header { text, first }, Event::Marker(Marker::Message)) => {
                let header = parse_header(text, *first)?;
                self.state = State::Content {
                    header,
                    text: String::new(),
                };
            }
            // A message cut off inside its header. The reference keeps the prompt's own one,
            // without a recipient, so it counts as a message that calls nothing; reading its
            // channel still fails on a bare `<|channel|>`, as there.
            (State::Header { text, first }, Event::Marker(marker)) if marker.ends() => {
                if *first && !text.is_empty() {
                    take_channel(text)?;
                    self.completed += 1;
                }
                self.state = State::Idle;
            }
            (State::Header { text, .. }, event) => text.push_str(spell(event)),
            (State::Content { .. }, Event::Marker(marker)) if marker.ends() => self.close(),
            (State::Content { header, text }, event) => {
                let delta = spell(event);
                text.push_str(delta);
                if delta.is_empty() || header.recipient.is_some() {
                    return Ok(vec![]);
                }
                if header.channel.as_deref() == Some("analysis") {
                    let message = self.completed;
                    let separate = self.thinking.is_some_and(|last| last != message);
                    self.thinking = Some(message);
                    let delta = if separate {
                        format!("\n\n{delta}")
                    } else {
                        delta.to_owned()
                    };
                    return Ok(vec![Delta::Thinking(delta)]);
                }
                return Ok(vec![Delta::Answer(delta.to_owned())]);
            }
        }
        Ok(vec![])
    }

    /// Keep text that never formed a whole character in the message being read, unshown.
    pub(super) fn tail(&mut self, rest: &str) {
        match &mut self.state {
            State::Header { text, .. } | State::Content { text, .. } => text.push_str(rest),
            State::Idle => {}
        }
    }

    /// End the response, closing a message the output limit cut off, and return its calls.
    pub(super) fn finish(&mut self) -> Result<Vec<FunctionCall>> {
        match self.state {
            State::Header { .. } => {
                bail!("Unexpected EOS while waiting for message header to complete");
            }
            State::Content { .. } => self.close(),
            State::Idle => {}
        }
        let mut calls = Vec::new();
        for (recipient, text) in &self.calls {
            let name = recipient.strip_prefix("functions.").unwrap_or(recipient);
            let arguments = serde_json::from_str(text)
                .map_err(|error| eyre!("error parsing tool call: raw='{text}', err={error}"))?;
            calls.push(FunctionCall {
                name: name.to_string(),
                arguments,
            });
        }
        Ok(calls)
    }

    /// Complete the message being read, remembering it if it calls a tool.
    fn close(&mut self) {
        let state = std::mem::replace(&mut self.state, State::Idle);
        if let State::Content { header, text } = state
            && header.assistant
            && let Some(recipient) = header.recipient
        {
            self.calls.push((recipient, text));
        }
        self.completed += 1;
    }
}

/// A token's text where the reference parser reads decoded text: markers by their spelling.
fn spell(event: Event<'_>) -> &str {
    match event {
        Event::Text(text) => text,
        Event::Marker(marker) => marker.spelling(),
    }
}

/// Cut the channel out of a spelled header: the word after `<|channel|>`, up to whitespace
/// or the next marker. Returns it and the header without it.
fn take_channel(header: &str) -> Result<(Option<String>, String)> {
    let Some(at) = header.find(CHANNEL) else {
        return Ok((None, header.to_string()));
    };
    let after = &header[at + CHANNEL.len()..];
    let length = after
        .find(|c: char| c.is_whitespace() || c == '<')
        .unwrap_or(after.len());
    if length == 0 {
        bail!("channel marker present but no channel value found in header");
    }
    let rest = format!("{}{}", &header[..at], &after[length..]);
    Ok((Some(after[..length].to_string()), rest))
}

/// Read a header as the reference does: the channel first, then whitespace-separated words of
/// role, recipient, and content type. The prompt's own header has already named the role.
fn parse_header(header: &str, first: bool) -> Result<Header> {
    let (channel, rest) = take_channel(header)?;
    let mut rest = rest.trim().to_string();
    if rest.contains(CONSTRAIN) {
        rest = rest
            .replace(CONSTRAIN, &format!(" {CONSTRAIN}"))
            .trim()
            .to_string();
    }
    let mut parts: Vec<&str> = rest.split_ascii_whitespace().collect();
    let role = if first {
        "assistant"
    } else {
        let Some(&role) = parts.first() else {
            bail!("message header did not contain a role");
        };
        if ["system", "developer", "user", "assistant", "tool"].contains(&role) {
            role
        } else if parts.len() > 1 || role.starts_with("to=") {
            // An unknown role followed by more words is a tool, named by that word.
            parts.remove(0);
            "tool"
        } else {
            bail!("Unknown role: {role}");
        }
    };
    if parts.first() == Some(&role) {
        parts.remove(0);
    }
    let mut recipient = None;
    if let Some(last) = parts.pop() {
        if let Some(name) = last.strip_prefix("to=") {
            recipient = Some(name);
        } else if parts.is_empty() {
            recipient = Some(last);
        } else if let Some(raw) = parts.pop() {
            // The last word was a content type, so the one before names the recipient.
            recipient = Some(raw.strip_prefix("to=").unwrap_or(raw));
        }
    }
    if !parts.is_empty() {
        bail!(
            "unexpected tokens remaining in message header: {:?}",
            Some(parts.join(" "))
        );
    }
    Ok(Header {
        assistant: role == "assistant",
        recipient: recipient.map(str::to_string),
        channel,
    })
}
