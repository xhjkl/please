//! Dense Qwen chat framing, rendered exactly as the checkpoint's chat template would.
//!
//! The template is reproduced piece by piece rather than evaluated: control tokens are
//! inserted by ID, so marker spellings inside user, stdin, or tool content stay ordinary text.
//! Deliberate departures from the template: such literal markers are never promoted to
//! control tokens; instructions that arrive mid-prompt become user messages instead of
//! an error; tool exchanges with no intervening assistant entry share one assistant step; and
//! the template's effort sentences follow the instructions, after the date, rather than open
//! the system message, so one cacheable prefix serves every effort.
use super::piece::{Piece, encode_pieces};
use super::prompt::{FunctionCall, Grade, Preamble, PromptMessage, Rendered};
use crate::hub::wording;
use crate::protocol::Effort;
use eyre::{Result, eyre};
use oco::Tokenizer;
use serde_json::Value;

mod output;
pub use output::Parser;

/// Tool-calling rules, quoted from the checkpoint's chat template.
const CALL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Control tokens that the template's own instruction text spells out.
const MARKERS: [&str; 2] = ["<tool_call>", "</tool_call>"];

/// The requested effort in the template's own sentences, at the nearest effort it has any
/// for; medium, which is also please's default, has none.
fn effort(requested: Effort) -> Option<&'static str> {
    match Grade::of(requested) {
        Grade::High => Some(
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
        ),
        Grade::Low => Some(
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
        ),
        Grade::Medium => None,
    }
}

/// Serialize like the template's `tojson` filter, which is Python's `json.dumps` with its
/// default `", "` and `": "` separators; keys keep their insertion order.
fn template_json(value: &Value) -> String {
    struct Spaced;
    impl serde_json::ser::Formatter for Spaced {
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
        ) -> std::io::Result<()> {
            writer.write_all(b": ")
        }
    }
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, Spaced);
    serde::Serialize::serialize(value, &mut serializer).expect("a JSON value serializes to memory");
    String::from_utf8(bytes).expect("serde_json writes UTF-8")
}

/// Rendered prompt whose adjacent text merges into one run, because the reference
/// tokenizer splits only at control tokens and BPE merges can span our concatenations.
#[derive(Debug, Default)]
struct Prompt {
    pieces: Vec<Piece>,
    /// End of the invocation-independent prefix: whole pieces, then bytes of the next text.
    prefix: (usize, usize),
}

impl Prompt {
    fn text(&mut self, text: &str) {
        match self.pieces.last_mut() {
            Some(Piece::Text(last)) => last.push_str(text),
            _ if text.is_empty() => {}
            _ => self.pieces.push(Piece::Text(text.to_owned())),
        }
    }

    fn marker(&mut self, spelling: &'static str) {
        self.pieces.push(Piece::Marker(spelling));
    }

    /// Add template-owned text whose marker spellings are structure, unlike message content.
    fn markup(&mut self, mut text: &str) {
        while let Some((index, marker)) = MARKERS
            .iter()
            .filter_map(|marker| Some((text.find(marker)?, *marker)))
            .min()
        {
            self.text(&text[..index]);
            self.marker(marker);
            text = &text[index + marker.len()..];
        }
        self.text(text);
    }

    /// Mark the end of the cacheable prefix; the next text starts the per-invocation facts.
    fn boundary(&mut self) {
        self.prefix = match self.pieces.last() {
            Some(Piece::Text(text)) => (self.pieces.len() - 1, text.len()),
            _ => (self.pieces.len(), 0),
        };
    }

    fn encode(&self, tokenizer: &Tokenizer) -> Result<Rendered> {
        let tokens = encode_pieces(tokenizer, &self.pieces)?;
        let (whole, bytes) = self.prefix;
        let mut prefix = encode_pieces(tokenizer, &self.pieces[..whole])?;
        if let Some(Piece::Text(text)) = self.pieces.get(whole) {
            prefix.extend(tokenizer.encode(&text[..bytes])?);
        }
        // A BPE merge across the boundary would change the prefix's last tokens; cache nothing then.
        let cacheable = if tokens.starts_with(&prefix) {
            prefix.len()
        } else {
            0
        };
        Ok(Rendered { tokens, cacheable })
    }

    fn message(&mut self, role: &str, content: &str) {
        self.marker("<|im_start|>");
        self.text(role);
        self.text("\n");
        self.text(content.trim());
        self.marker("<|im_end|>");
        self.text("\n");
    }

    /// Lead with tools, then the merged leading system and developer messages, as the template does.
    /// Static instructions precede per-invocation facts, so everything before the facts is cacheable.
    fn system(&mut self, instructions: &str, facts: &str) {
        self.marker("<|im_start|>");
        self.text("system\n");
        self.text("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for schema in super::prompt::tool_schemas() {
            self.text("\n");
            self.text(&template_json(&schema));
        }
        self.text("\n</tools>");
        self.markup(CALL_INSTRUCTIONS);
        if !instructions.is_empty() || !facts.is_empty() {
            self.text("\n\n");
        }
        self.text(instructions);
        if !instructions.is_empty() && !facts.is_empty() {
            self.text("\n");
        }
        self.boundary();
        self.text(facts);
        self.marker("<|im_end|>");
        self.text("\n");
    }

    /// Render one step; earlier reasoning stays, as the template preserves thinking by default.
    fn assistant(&mut self, reasoning: &str, content: &str, calls: &[&FunctionCall]) -> Result<()> {
        let content = content.trim();
        self.marker("<|im_start|>");
        self.text("assistant\n");
        self.marker("<think>");
        self.text("\n");
        self.text(reasoning.trim());
        self.text("\n");
        self.marker("</think>");
        self.text("\n\n");
        self.text(content);
        for (index, FunctionCall { name, arguments }) in calls.iter().enumerate() {
            match index {
                0 if content.is_empty() => {}
                0 => self.text("\n\n"),
                _ => self.text("\n"),
            }
            let arguments = arguments
                .as_object()
                .ok_or_else(|| eyre!("Qwen tool arguments must be an object"))?;
            self.marker("<tool_call>");
            self.text(&format!("\n<function={name}>\n"));
            for (name, value) in arguments {
                let value = match value {
                    Value::String(value) => value.clone(),
                    value => template_json(value),
                };
                self.text(&format!("<parameter={name}>\n{value}\n</parameter>\n"));
            }
            self.text("</function>\n");
            self.marker("</tool_call>");
        }
        self.marker("<|im_end|>");
        self.text("\n");
        Ok(())
    }

    /// Answer a step's calls with one user message holding every response in order.
    fn tool_responses(&mut self, results: &[&str]) {
        self.marker("<|im_start|>");
        self.text("user");
        for result in results {
            self.text("\n");
            self.marker("<tool_response>");
            self.text("\n");
            self.text(result.trim());
            self.text("\n");
            self.marker("</tool_response>");
        }
        self.marker("<|im_end|>");
        self.text("\n");
    }
}

/// Render the preamble and worded history into template pieces, ending with an open assistant
/// turn in thinking.
fn render(preamble: &Preamble, messages: &[PromptMessage]) -> Result<Prompt> {
    use PromptMessage::*;
    let mut prompt = Prompt::default();
    let instructions: Vec<_> = preamble
        .instructions
        .iter()
        .map(|text| text.trim())
        .collect();
    let mut facts = vec![wording::date(&preamble.today)];
    facts.extend(effort(preamble.effort).map(String::from));
    prompt.system(&instructions.join("\n"), &facts.join("\n"));
    let mut rest = messages;
    while let Some(first) = rest.first() {
        match first {
            Note(text) | User(text) => {
                prompt.message("user", text);
                rest = &rest[1..];
            }
            ToolResult { .. } => {
                let length = rest
                    .iter()
                    .take_while(|message| matches!(message, ToolResult { .. }))
                    .count();
                prompt.tool_responses(&results(&rest[..length]));
                rest = &rest[length..];
            }
            Thinking(_) | Answer(_) | ToolCall(_) => {
                let mut reasoning = String::new();
                let mut content = String::new();
                let mut length = 0;
                for message in rest {
                    match message {
                        Thinking(text) => reasoning.push_str(text),
                        Answer(text) => content.push_str(text),
                        _ => break,
                    }
                    length += 1;
                }
                let exchange = rest[length..]
                    .iter()
                    .take_while(|message| matches!(message, ToolCall(_) | ToolResult { .. }))
                    .count();
                let exchange = &rest[length..length + exchange];
                let calls: Vec<_> = exchange
                    .iter()
                    .filter_map(|message| match message {
                        ToolCall(call) => Some(call),
                        _ => None,
                    })
                    .collect();
                prompt.assistant(&reasoning, &content, &calls)?;
                let results = results(exchange);
                if !results.is_empty() {
                    prompt.tool_responses(&results);
                }
                rest = &rest[length + exchange.len()..];
            }
        }
    }
    prompt.marker("<|im_start|>");
    prompt.text("assistant\n");
    prompt.marker("<think>");
    prompt.text("\n");
    Ok(prompt)
}

/// Collect the tool results of one exchange in order.
fn results(messages: &[PromptMessage]) -> Vec<&str> {
    messages
        .iter()
        .filter_map(|message| match message {
            PromptMessage::ToolResult { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect()
}

/// Encode the preamble and worded messages as dense Qwen's chat template would.
pub fn encode(
    tokenizer: &Tokenizer,
    preamble: &Preamble,
    messages: &[PromptMessage],
) -> Result<Rendered> {
    render(preamble, messages)?.encode(tokenizer)
}

/// Stop at the turn terminator or end of text, the checkpoint's generation stop set.
pub fn is_stop_token(tokenizer: &Tokenizer, token: u32) -> bool {
    ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .any(|spelling| tokenizer.special_token(spelling) == Some(token))
        || tokenizer.eos_token() == Some(token)
}

#[cfg(test)]
mod tests;
