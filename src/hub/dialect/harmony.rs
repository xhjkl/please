//! Harmony framing for gpt-oss, vendored from OpenAI's `openai-harmony` 0.0.8 (Apache-2.0)
//! and cut down to the messages please renders and the shapes the model emits.
//!
//! Prompts are rendered piece by piece: control tokens go in by ID, so marker spellings inside
//! user, stdin, or tool content stay ordinary text, and every text run is encoded on its own,
//! as the reference encoder does, so token boundaries match it exactly. The checkpoint's
//! tokenizer encodes the text; its vocabulary is the encoding's, so nothing is downloaded.
//! The system message is the encoder's canonical one; please's instructions and tool schemas
//! form the developer message, as gpt-oss's chat template places them. Unlike that template,
//! the "Current date" line comes in a second developer message rather than in the system
//! message, so the head outlasts the day. The reasoning line stays in the system message, the
//! one place gpt-oss heeds it, so its head differs by effort.
use super::OutputParser;
use super::piece::{Piece, encode_prompt};
use super::prompt::{FunctionCall, Grade, Preamble, PromptMessage, Rendered, tool_schemas};
use super::wording;
use crate::protocol::Effort;
use eyre::Result;
use oco::Tokenizer;
use serde_json::Value;

mod output;

/// Opens a message, whose header follows.
const START: &str = "<|start|>";
/// Introduces the header's channel.
const CHANNEL: &str = "<|channel|>";
/// Introduces a constrained content type in the header.
const CONSTRAIN: &str = "<|constrain|>";
/// Ends the header; the content follows.
const MESSAGE: &str = "<|message|>";
/// Ends a message that another follows.
const END: &str = "<|end|>";
/// Ends an assistant message addressed to a tool; generation stops for the call.
const CALL: &str = "<|call|>";
/// Ends the assistant's final message; generation stops.
const RETURN: &str = "<|return|>";

/// The encoder's default identity and knowledge cutoff, which gpt-oss was trained under.
const IDENTITY: &str =
    "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06";
/// Channel rules. The last sentence is the encoder's addition for prompts declaring
/// function tools, which please's always do.
const CHANNELS: &str = "# Valid channels: analysis, commentary, final. Channel must be included for every message.\nCalls to these tools must go to the commentary channel: 'functions'.";

/// A message header's fields, which the encoder writes in this order.
#[derive(Default)]
struct Header<'a> {
    /// The role, or a tool's name for its result.
    author: &'a str,
    /// Whom the message addresses: a tool for a call, the assistant for a result.
    recipient: Option<&'a str>,
    /// The assistant's channel: analysis, commentary, or final.
    channel: Option<&'a str>,
    /// The content is constrained to JSON, as tool-call arguments are.
    json: bool,
}

impl<'a> Header<'a> {
    /// A header naming only its author, as instructions and user turns have.
    fn of(author: &'a str) -> Self {
        Self {
            author,
            ..Self::default()
        }
    }
}

/// Rendered prompt. Text runs stay separate: the encoder tokenizes each on its own.
#[derive(Debug, Default)]
struct Prompt {
    pieces: Vec<Piece>,
}

impl Prompt {
    /// Append a text run, encoded apart from its neighbours.
    fn text(&mut self, text: impl Into<String>) {
        self.pieces.push(Piece::Text(text.into()));
    }

    /// Append a control token by its spelling.
    fn marker(&mut self, spelling: &'static str) {
        self.pieces.push(Piece::Marker(spelling));
    }

    /// Append one message: its header, content, and closing token.
    fn message(&mut self, header: Header, content: &str) {
        self.marker(START);
        self.text(header.author);
        if let Some(recipient) = header.recipient {
            self.text(format!(" to={recipient}"));
        }
        if let Some(channel) = header.channel {
            self.marker(CHANNEL);
            self.text(channel);
        }
        if header.json {
            self.text(" ");
            self.marker(CONSTRAIN);
            self.text("json");
        }
        self.marker(MESSAGE);
        self.text(content);
        let call = header.author == "assistant" && header.recipient.is_some();
        self.marker(if call { CALL } else { END });
    }
}

/// Render the preamble and worded history, opening with the canonical system and developer
/// messages and ending in the assistant's open turn; also return how many pieces form the
/// static head.
fn render(preamble: &Preamble, messages: &[PromptMessage]) -> (Prompt, usize) {
    let mut prompt = Prompt::default();
    let head = header(&mut prompt, preamble);
    for message in messages {
        frame(&mut prompt, message);
    }
    prompt.marker(START);
    prompt.text("assistant");
    (prompt, head)
}

/// Append the canonical system message with the requested effort, then please's instructions
/// and tools, then a second developer message with the date. Keeping the date out of the
/// first two leaves the tool declarations in the cacheable head while the date keeps
/// developer authority. Returns how many pieces the first two messages took.
fn header(prompt: &mut Prompt, preamble: &Preamble) -> usize {
    let instructions: Vec<_> = preamble
        .instructions
        .iter()
        .map(|text| text.trim())
        .collect();
    let reasoning = reasoning(preamble.effort);
    let system = format!("{IDENTITY}\n\n{reasoning}\n\n{CHANNELS}");
    let developer = format!(
        "# Instructions\n\n{}\n\n{}",
        instructions.join("\n\n"),
        tools_section()
    );
    prompt.message(Header::of("system"), &system);
    prompt.message(Header::of("developer"), &developer);
    let head = prompt.pieces.len();
    let today = wording::date(&preamble.today);
    let facts = format!("# Instructions\n\n{today}");
    prompt.message(Header::of("developer"), &facts);
    head
}

/// The requested effort as the encoder's reasoning line, at the nearest of the three levels
/// it names. The line works only in the system message: in a developer message it changed
/// how long gpt-oss thought by about a tenth, against more than fivefold where it was trained.
fn reasoning(effort: Effort) -> &'static str {
    match Grade::of(effort) {
        Grade::Low => "Reasoning: low",
        Grade::Medium => "Reasoning: medium",
        Grade::High => "Reasoning: high",
    }
}

/// Declare please's tools as the encoder's TypeScript `functions` namespace.
fn tools_section() -> String {
    let mut lines = vec![
        "## functions\n".to_string(),
        "namespace functions {\n".into(),
    ];
    for schema in tool_schemas() {
        let function = &schema["function"];
        let description = function["description"].as_str().unwrap_or_default();
        for line in description.lines() {
            lines.push(format!("// {line}"));
        }
        let name = function["name"].as_str().unwrap_or_default();
        let parameters = parameters(&function["parameters"]);
        lines.push(format!("type {name} = (_: {parameters}) => any;\n"));
    }
    lines.push("} // namespace functions".into());
    format!("# Tools\n\n{}", lines.join("\n"))
}

/// Spell a tool's parameters as the encoder's TypeScript object type, for what please's tool
/// schemas hold: described properties whose types are strings, string enums, numbers, or
/// arrays of those.
fn parameters(schema: &Value) -> String {
    let required = schema["required"].as_array();
    let required = |key: &str| required.is_some_and(|keys| keys.iter().any(|k| k == key));
    let mut out = String::from("{\n");
    for (key, property) in schema["properties"].as_object().into_iter().flatten() {
        if let Some(description) = property["description"].as_str() {
            out.push_str(&format!("// {description}\n"));
        }
        let optional = if required(key) { "" } else { "?" };
        out.push_str(&format!("{key}{optional}: {},\n", typescript(property)));
    }
    out.push('}');
    out
}

/// Spell a property's type as the encoder does; anything else becomes `any`, as there.
fn typescript(schema: &Value) -> String {
    match schema["type"].as_str() {
        Some("string") => {
            let choices: Vec<_> = schema["enum"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(|choice| format!("\"{choice}\""))
                .collect();
            if choices.is_empty() {
                "string".into()
            } else {
                choices.join(" | ")
            }
        }
        Some("number") => "number".into(),
        Some("array") => format!("{}[]", typescript(&schema["items"])),
        _ => "any".into(),
    }
}

/// Frame one worded message; the harness's word mid-turn is developer text.
fn frame(prompt: &mut Prompt, message: &PromptMessage) {
    let assistant = |channel| Header {
        author: "assistant",
        channel: Some(channel),
        ..Header::default()
    };
    match message {
        PromptMessage::Note(content) => prompt.message(Header::of("developer"), content),
        PromptMessage::User(content) => prompt.message(Header::of("user"), content),
        PromptMessage::Answer(content) => prompt.message(assistant("final"), content),
        PromptMessage::Thinking(content) if content.is_empty() => {}
        PromptMessage::Thinking(content) => prompt.message(assistant("analysis"), content),
        PromptMessage::ToolCall(FunctionCall { name, arguments }) => {
            let recipient = format!("functions.{name}");
            let header = Header {
                recipient: Some(&recipient),
                json: true,
                ..assistant("commentary")
            };
            prompt.message(header, &arguments.to_string());
        }
        // Naming the author by its namespace, as gpt-oss's template and the call recipient do.
        PromptMessage::ToolResult { name, content } => {
            let author = format!("functions.{name}");
            let header = Header {
                author: &author,
                recipient: Some("assistant"),
                channel: Some("commentary"),
                json: false,
            };
            prompt.message(header, content);
        }
    }
}

/// Encode the preamble and worded messages as Harmony, over the checkpoint's own vocabulary.
pub fn encode(
    tokenizer: &Tokenizer,
    preamble: &Preamble,
    messages: &[PromptMessage],
) -> Result<Rendered> {
    let (prompt, head) = render(preamble, messages);
    encode_prompt(tokenizer, &prompt.pieces, head)
}

/// Start decoding a response that resumes the prompt's open assistant header.
pub fn output_parser(tokenizer: &Tokenizer) -> Result<Box<dyn OutputParser + '_>> {
    Ok(Box::new(output::Parser::new(tokenizer)?))
}

pub fn is_stop_token(tokenizer: &Tokenizer, token: u32) -> bool {
    [RETURN, CALL]
        .iter()
        .any(|spelling| tokenizer.special_token(spelling) == Some(token))
}

#[cfg(test)]
mod tests;
