//! Streaming split of generated Qwen text into thinking, answer, and native tool calls.
//!
//! Structure comes only from the checkpoint's added tokens, which is how the model emits it
//! under the template's own instructions; marker spellings the model writes out as ordinary
//! text, for example when quoting documentation, stay content.
use super::super::conversation::{Delta, FunctionCall};
use super::super::piece::special;
use super::super::{Adapter as _, OutputParser as DialectOutputParser};
use super::Adapter;
use crate::tools::{Param, ParamType};
use eyre::{Result, eyre};
use oco::Tokenizer;
use serde_json::{Map, Value};

/// Token-level front end: recognises stop and structure tokens by ID and decodes the rest.
pub struct OutputParser<'a> {
    tokenizer: &'a Tokenizer,
    decoder: oco::Decoder<'a>,
    markers: [(u32, Event<'static>); 3],
    stream: OutputStream,
}

impl<'a> OutputParser<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Result<Self> {
        Ok(Self {
            tokenizer,
            decoder: tokenizer.decoder(),
            markers: [
                (special(tokenizer, "</think>")?, Event::ThinkEnd),
                (special(tokenizer, "<tool_call>")?, Event::CallStart),
                (special(tokenizer, "</tool_call>")?, Event::CallEnd),
            ],
            stream: OutputStream::default(),
        })
    }
}

impl DialectOutputParser for OutputParser<'_> {
    fn push_token(&mut self, token: u32) -> Result<Vec<Delta>> {
        let event = if Adapter.is_stop_token(self.tokenizer, token) {
            Some(Event::End)
        } else {
            self.markers
                .iter()
                .find(|(id, _)| *id == token)
                .map(|&(_, event)| event)
        };
        if let Some(event) = event {
            // Finishing partial UTF-8 before the marker keeps it in the section it came from.
            let decoder = std::mem::replace(&mut self.decoder, self.tokenizer.decoder());
            let rest = decoder.finish();
            let mut deltas = self.stream.push(Event::Text(&rest));
            deltas.extend(self.stream.push(event));
            return Ok(deltas);
        }
        let text = self.decoder.push(token)?;
        Ok(self.stream.push(Event::Text(&text)))
    }

    fn finish(&mut self) -> Result<Vec<FunctionCall>> {
        self.stream.finish()
    }
}

/// One step of generated output: decoded content, or a structure token.
#[derive(Debug, Clone, Copy)]
pub(super) enum Event<'a> {
    Text(&'a str),
    ThinkEnd,
    CallStart,
    CallEnd,
    End,
}

/// Whitespace that the template puts around boundaries is structure, not content: it is
/// dropped at the start of each section and held back until later text proves it inner.
#[derive(Default)]
pub(super) struct OutputStream {
    answering: bool,
    /// Visible text has been emitted since the last boundary.
    inner: bool,
    /// Trailing whitespace not yet known to be content.
    held: String,
    current_call: Option<String>,
    calls: Vec<Result<FunctionCall>>,
}

impl OutputStream {
    pub(super) fn push(&mut self, event: Event) -> Vec<Delta> {
        let mut deltas = Vec::new();
        match event {
            Event::Text(text) => self.text(text, &mut deltas),
            Event::ThinkEnd if !self.answering => self.boundary(),
            Event::ThinkEnd => self.text("</think>", &mut deltas),
            Event::CallStart if self.current_call.is_none() => {
                self.boundary();
                self.current_call = Some(String::new());
            }
            Event::CallStart => self.text("<tool_call>", &mut deltas),
            Event::CallEnd => match self.current_call.take() {
                Some(mut call) if call.trim_end().ends_with("</function>") => {
                    let parsed = parse_call(&call);
                    if parsed.as_ref().is_err_and(|error| {
                        error
                            .downcast_ref::<serde_json::Error>()
                            .is_some_and(serde_json::Error::is_eof)
                    }) {
                        // An unfinished JSON argument can contain both closing spellings.
                        call.push_str("</tool_call>");
                        self.current_call = Some(call);
                    } else {
                        self.calls.push(parsed);
                        self.boundary();
                    }
                }
                // A closing token inside an unfinished function belongs to its arguments.
                Some(mut call) if is_function(&call) => {
                    call.push_str("</tool_call>");
                    self.current_call = Some(call);
                }
                Some(call) => self.quoted(&call, "</tool_call>", &mut deltas),
                None => self.text("</tool_call>", &mut deltas),
            },
            Event::End => {
                if let Some(call) = self.current_call.take_if(|call| !is_function(call)) {
                    self.quoted(&call, "", &mut deltas);
                }
                self.held.clear();
            }
        }
        deltas
    }

    /// Return a "call" that never began a function to the answer: the model emits the same
    /// structure tokens when it merely quotes the tags, for instance while explaining them.
    fn quoted(&mut self, call: &str, closing: &str, deltas: &mut Vec<Delta>) {
        self.text(&format!("<tool_call>{call}{closing}"), deltas);
    }

    /// Start a new answer section, discarding the whitespace that separated it from the last one.
    fn boundary(&mut self) {
        self.answering = true;
        self.inner = false;
        self.held.clear();
    }

    fn text(&mut self, text: &str, deltas: &mut Vec<Delta>) {
        if let Some(call) = &mut self.current_call {
            call.push_str(text);
            return;
        }
        let mut text = std::mem::take(&mut self.held) + text;
        if !self.inner {
            text = text.trim_start().to_owned();
        }
        let content = text.trim_end().len();
        self.held = text.split_off(content);
        if text.is_empty() {
            return;
        }
        self.inner = true;
        deltas.push(if self.answering {
            Delta::Answer(text)
        } else {
            Delta::Thinking(text)
        });
    }

    pub(super) fn finish(&mut self) -> Result<Vec<FunctionCall>> {
        if self.current_call.is_some() {
            eyre::bail!("incomplete Qwen tool call");
        }
        std::mem::take(&mut self.calls).into_iter().collect()
    }
}

/// Whether a call body has begun a native function block, so it is structure rather than a quote.
fn is_function(call: &str) -> bool {
    let call = call.trim_start();
    !call.is_empty() && (call.starts_with("<function=") || "<function=".starts_with(call))
}

/// Find the end of a plain value before the next declared parameter or the end of the function.
fn value_end(rest: &str, parameters: &[Param]) -> Option<(usize, usize)> {
    rest.match_indices("</parameter>")
        .find_map(|(index, closing)| {
            let after = rest[index + closing.len()..].trim_start();
            let next = parameters
                .iter()
                .any(|parameter| after.starts_with(&format!("<parameter={}>", parameter.name)));
            (after.is_empty() || next).then_some((index, rest.len() - after.len()))
        })
}

/// Parse Qwen's native function/parameter blocks using the declared parameter types.
pub(super) fn parse_call(source: &str) -> Result<FunctionCall> {
    let source = source.trim();
    let source = source
        .strip_prefix("<function=")
        .ok_or_else(|| eyre!("missing <function=...> in Qwen tool call"))?;
    let (name, body) = source
        .split_once('>')
        .ok_or_else(|| eyre!("unterminated function name"))?;
    let tools = crate::hub::dialect::conversation::tool_specs();
    let (_, _, parameters) = tools
        .into_iter()
        .find(|(known, ..)| *known == name)
        .ok_or_else(|| eyre!("unknown tool {name}"))?;
    let mut body = body
        .trim()
        .strip_suffix("</function>")
        .ok_or_else(|| eyre!("missing </function>"))?
        .trim();
    let mut arguments = Map::new();
    while !body.is_empty() {
        let rest = body
            .strip_prefix("<parameter=")
            .ok_or_else(|| eyre!("expected parameter block"))?;
        let (key, rest) = rest
            .split_once('>')
            .ok_or_else(|| eyre!("unterminated parameter name"))?;
        let parameter = parameters
            .iter()
            .find(|parameter| parameter.name == key)
            .ok_or_else(|| eyre!("unknown {name} parameter {key}"))?;
        let (value, remaining) = match &parameter.param_type {
            ParamType::StringArray => {
                // Reading the JSON first; quoted strings can contain parameter delimiters.
                let mut values =
                    serde_json::Deserializer::from_str(rest).into_iter::<Vec<String>>();
                let value = values
                    .next()
                    .transpose()?
                    .ok_or_else(|| eyre!("missing {key} value"))?;
                let after = rest[values.byte_offset()..].trim_start();
                let after = after
                    .strip_prefix("</parameter>")
                    .ok_or_else(|| eyre!("unterminated parameter {key}"))?;
                (Value::from(value), after.trim_start())
            }
            kind => {
                let (end, next) = value_end(rest, parameters)
                    .ok_or_else(|| eyre!("unterminated parameter {key}"))?;
                let value = &rest[..end];
                // Removing only the template's enclosing newlines preserves patch whitespace.
                let value = value.strip_prefix('\n').unwrap_or(value);
                let value = value.strip_suffix('\n').unwrap_or(value);
                let value = match kind {
                    ParamType::Choice(choices) => {
                        let value = value.trim();
                        if !choices.contains(&value) {
                            eyre::bail!("invalid {key} choice");
                        }
                        Value::String(value.to_owned())
                    }
                    ParamType::Number => Value::Number(
                        serde_json::from_str(value)
                            .map_err(|error| eyre!("invalid {key} number: {error}"))?,
                    ),
                    ParamType::String => Value::String(value.to_owned()),
                    ParamType::StringArray => unreachable!(),
                };
                (value, &rest[next..])
            }
        };
        let previous = arguments.insert(key.to_string(), value);
        if previous.is_some() {
            eyre::bail!("duplicate parameter {key}");
        }
        body = remaining;
    }
    for parameter in parameters {
        if parameter.required && !arguments.contains_key(parameter.name) {
            eyre::bail!("missing parameter {}", parameter.name);
        }
    }
    Ok(FunctionCall {
        name: name.to_string(),
        arguments: Value::Object(arguments),
    })
}
