//! Harmony parser: collects OpenAI's markup format into consumable events.
//! Taken from Ollama's [harmony/harmonyparser.go].
//! [https://cookbook.openai.com/articles/openai-harmony]

use std::collections::HashMap;

/// Minimal header describing who is speaking, through which channel, to whom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarmonyHeader {
    pub role: String,
    pub channel: String,
    pub recipient: String,
}

/// Streamed parser events. Consumers build UX/state from these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarmonyEvent {
    MessageStart,
    HeaderComplete { header: HarmonyHeader },
    ContentEmitted { content: String },
    MessageEnd,
}

/// Internal parser phases. Single message lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserState {
    LookingForMessageStart,
    ParsingHeader,
    ParsingContent,
}

/// Streaming Harmony tag parser. Feed bytes, get `HarmonyEvent`s.
#[derive(Debug)]
pub struct HarmonyParser {
    state: ParserState,
    pub message_start_tag: String,
    pub message_end_tag: String,
    pub header_end_tag: String,
    acc: String,
}

impl Default for HarmonyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonyParser {
    /// Construct with default tags.
    pub fn new() -> Self {
        Self {
            state: ParserState::LookingForMessageStart,
            message_start_tag: "<|start|>".to_string(),
            message_end_tag: "<|end|>".to_string(),
            header_end_tag: "<|message|>".to_string(),
            acc: String::new(),
        }
    }

    /// Insert a synthetic assistant start when upstream omits it.
    pub fn add_implicit_start(&mut self) {
        self.acc.push_str("<|start|>assistant");
    }

    /// Start message or prefill channel to resume a partial assistant turn.
    pub fn add_implicit_start_or_prefill(&mut self, last_message: Option<&LastMessage>) {
        if let Some(m) = last_message
            && m.role == "assistant"
        {
            if !m.content.is_empty() {
                self.acc
                    .push_str("<|start|>assistant<|channel|>final<|message|>");
                return;
            } else if !m.thinking.is_empty() {
                self.acc
                    .push_str("<|start|>assistant<|channel|>analysis<|message|>");
                return;
            }
        }
        self.add_implicit_start();
    }

    /// Feed additional content into the parser. Emits zero or more events.
    pub fn add_content(&mut self, content: &str) -> Vec<HarmonyEvent> {
        self.acc.push_str(content);

        let mut events: Vec<HarmonyEvent> = Vec::new();
        let mut keep_parsing = true;
        while keep_parsing {
            let (new_events, should_continue) = Self::advance(self);
            events.extend(new_events);
            keep_parsing = should_continue;
        }
        events
    }

    /// Advance the state machine at most one step; return events and whether to continue.
    fn advance(&mut self) -> (Vec<HarmonyEvent>, bool) {
        match self.state {
            ParserState::LookingForMessageStart => {
                let acc = self.acc.clone();
                if let Some(idx) = acc.find(&self.message_start_tag) {
                    let before = &acc[..idx];
                    if !before.is_empty() {
                        // parity: warn in upstream; we silently ignore extra text
                    }
                    let after_start = &acc[idx + self.message_start_tag.len()..];
                    self.acc.clear();
                    self.acc.push_str(after_start);
                    // Found start → begin header
                    self.state = ParserState::ParsingHeader;
                    return (vec![HarmonyEvent::MessageStart], true);
                }
                (vec![], false)
            }
            ParserState::ParsingHeader => {
                let acc = self.acc.clone();
                if let Some(idx) = acc.find(&self.header_end_tag) {
                    let header = acc[..idx].to_string();
                    let after = acc[idx + self.header_end_tag.len()..].to_string();
                    self.acc.clear();
                    self.acc.push_str(&after);
                    // Header closed → stream content
                    self.state = ParserState::ParsingContent;
                    let parsed = Self::parse_header(&header);
                    return (vec![HarmonyEvent::HeaderComplete { header: parsed }], true);
                }
                (vec![], false)
            }
            ParserState::ParsingContent => {
                let acc = self.acc.clone();
                if let Some(idx) = acc.find(&self.message_end_tag) {
                    let content = acc[..idx].to_string();
                    let after = acc[idx + self.message_end_tag.len()..].to_string();
                    self.acc.clear();
                    self.acc.push_str(&after);
                    // Message closed → reset for next
                    self.state = ParserState::LookingForMessageStart;
                    let mut events = Vec::new();
                    if !content.is_empty() {
                        events.push(HarmonyEvent::ContentEmitted { content });
                    }
                    events.push(HarmonyEvent::MessageEnd);
                    return (events, true);
                }

                // Avoid cutting a potential end tag in half across chunks
                let overlap_len = overlap(&self.acc, &self.message_end_tag);
                if overlap_len > 0 {
                    let emit = self.acc[..self.acc.len() - overlap_len].to_string();
                    let remaining = self.acc[self.acc.len() - overlap_len..].to_string();
                    self.acc.clear();
                    self.acc.push_str(&remaining);
                    if emit.is_empty() {
                        return (vec![], false);
                    }
                    return (vec![HarmonyEvent::ContentEmitted { content: emit }], false);
                }

                let content = std::mem::take(&mut self.acc);
                if content.is_empty() {
                    return (vec![], false);
                }
                (vec![HarmonyEvent::ContentEmitted { content }], false)
            }
        }
    }

    /// Parse role/channel/recipient from a header slice.
    pub fn parse_header(raw_in: &str) -> HarmonyHeader {
        let mut header = HarmonyHeader {
            role: String::new(),
            channel: String::new(),
            recipient: String::new(),
        };
        let mut raw = raw_in.to_string();

        if raw.contains("<|constrain|>") {
            raw = raw
                .replacen("<|constrain|>", " <|constrain|>", 1)
                .trim()
                .to_string();
        }

        if let Some(idx) = raw.find("<|channel|>") {
            let before = &raw[..idx];
            let after = &raw[idx + "<|channel|>".len()..];
            let mut end = after.len();
            for (i, ch) in after.char_indices() {
                if ch.is_whitespace() {
                    end = i;
                    break;
                }
            }
            header.channel = after[..end].to_string();
            let after_rest = &after[end..];
            raw = format!("{}{}", before, after_rest).trim().to_string();
        }

        let mut tokens = raw.split_whitespace();
        let first = match tokens.next() {
            Some(t) => t,
            None => {
                return header; // parity: missing role
            }
        };

        if let Some(rest) = first.strip_prefix("to=") {
            header.recipient = rest.to_string();
            header.role = "tool".to_string();
        } else {
            header.role = first.to_string();
        }

        if header.recipient.is_empty()
            && let Some(next) = tokens.next()
            && let Some(rest) = next.strip_prefix("to=")
        {
            header.recipient = rest.to_string();
        }

        header
    }
}

/// Longest suffix of `s` that is a prefix of `delim`.
fn overlap(s: &str, delim: &str) -> usize {
    let max = std::cmp::min(delim.len(), s.len());
    for i in (1..=max).rev() {
        if s.ends_with(&delim[..i]) {
            return i;
        }
    }
    0
}

/// Snapshot of the prior assistant turn to prefill continuation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LastMessage {
    pub role: String,
    pub content: String,
    pub thinking: String,
}

/// High-level content routing while handling a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageState {
    Answering,
    Thinking,
    ToolCalling,
}

/// Orchestrates parsing, channel routing, and tool-call assembly.
#[derive(Debug)]
pub struct HarmonyMessageHandler {
    state: MessageState,
    pub parser: HarmonyParser,
    pub function_name_map: FunctionNameMap,
    tool_accumulator: HarmonyToolCallAccumulator,
    converted_tools: HashMap<String, ()>,
}

impl HarmonyMessageHandler {
    /// Construct a handler with a fresh parser and empty tool maps.
    pub fn new() -> Self {
        Self {
            state: MessageState::Answering,
            parser: HarmonyParser::new(),
            function_name_map: FunctionNameMap::new(),
            tool_accumulator: HarmonyToolCallAccumulator::new(),
            converted_tools: HashMap::new(),
        }
    }
}

impl Default for HarmonyMessageHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonyMessageHandler {
    /// Prepare for a new completion: prefill, reset tool state, normalize tool names.
    pub fn init(&mut self, tools: &[Tool], last_message: Option<&LastMessage>) -> Vec<Tool> {
        if let Some(m) = last_message {
            self.parser.add_implicit_start_or_prefill(Some(m));
        } else {
            self.parser.add_implicit_start();
        }
        self.tool_accumulator = HarmonyToolCallAccumulator::new();
        if tools.is_empty() {
            return tools.to_vec();
        }
        let mut processed = tools.to_vec();
        for t in &mut processed {
            if let Some(name) = t.function.name.as_ref() {
                let converted = self.function_name_map.convert_and_add(name);
                self.converted_tools.insert(name.clone(), ());
                t.function.name = Some(converted);
            }
        }
        processed
    }

    /// Consume new model output and return (answer, thinking, tool calls).
    pub fn add(&mut self, s: &str, done: bool) -> Result<(String, String, Vec<ToolCall>), String> {
        let mut content = String::new();
        let mut thinking = String::new();
        let mut tool_payload = String::new();
        let events = self.parser.add_content(s);
        for ev in events {
            match ev {
                HarmonyEvent::HeaderComplete { header } => match header.channel.as_str() {
                    "analysis" => {
                        if !header.recipient.is_empty() {
                            self.state = MessageState::ToolCalling;
                            self.tool_accumulator.set_tool_name(header.recipient);
                        } else {
                            self.state = MessageState::Thinking;
                        }
                    }
                    "commentary" => {
                        if !header.recipient.is_empty() {
                            self.state = MessageState::ToolCalling;
                            self.tool_accumulator.set_tool_name(header.recipient);
                        } else {
                            // Route to final answer stream
                            self.state = MessageState::Answering;
                        }
                    }
                    "final" => {
                        // Explicit final channel → answer stream
                        self.state = MessageState::Answering;
                    }
                    _ => {}
                },
                HarmonyEvent::ContentEmitted { content: c } => match self.state {
                    MessageState::Answering => content.push_str(&c),
                    MessageState::Thinking => thinking.push_str(&c),
                    MessageState::ToolCalling => tool_payload.push_str(&c),
                },
                HarmonyEvent::MessageEnd => {
                    // Always reset to answering-mode default after a message
                    self.state = MessageState::Answering;
                }
                HarmonyEvent::MessageStart => {}
            }
        }
        if !tool_payload.is_empty() {
            // Accumulate tool args until completion boundary
            self.tool_accumulator.add(&tool_payload);
        }

        let mut calls: Vec<ToolCall> = Vec::new();
        if done {
            let (tool_name, raw) = self.tool_accumulator.drain();
            if let Some(mut name) = tool_name {
                if let Some(stripped) = name.strip_prefix("functions.") {
                    name = stripped.to_string();
                }
                name = self.function_name_map.original_from_converted(&name);
                // JSON parse: return error message on failure
                let args: serde_json::Value = serde_json::from_str(&raw)
                    .map_err(|e| format!("error parsing tool call: raw='{}', err={}", raw, e))?;
                calls.push(ToolCall {
                    function: ToolCallFunction {
                        name,
                        arguments: args,
                    },
                });
            }
        }

        Ok((content, thinking, calls))
    }
}

/// Collects tool call JSON and current tool name across chunks.
#[derive(Debug, Clone, Default)]
pub struct HarmonyToolCallAccumulator {
    acc: String,
    current_tool_name: Option<String>,
}

impl HarmonyToolCallAccumulator {
    /// New, empty accumulator.
    pub fn new() -> Self {
        Self {
            acc: String::new(),
            current_tool_name: None,
        }
    }

    /// Set active tool name as seen in header.
    pub fn set_tool_name(&mut self, tool_name: String) {
        self.current_tool_name = Some(tool_name);
    }

    /// Append raw JSON chunk.
    pub fn add(&mut self, content: &str) {
        self.acc.push_str(content);
    }

    /// Take accumulated JSON and clear state.
    pub fn drain(&mut self) -> (Option<String>, String) {
        let raw = std::mem::take(&mut self.acc);
        let name = self.current_tool_name.take();
        (name, raw)
    }

    /// Peek at current buffer.
    pub fn content(&self) -> &str {
        &self.acc
    }
}

/// Bidirectional map between user tool names and Harmony-safe names.
#[derive(Debug, Clone, Default)]
pub struct FunctionNameMap {
    user_to_harmony: HashMap<String, String>,
    harmony_to_user: HashMap<String, String>,
}

impl FunctionNameMap {
    /// Create an empty mapping.
    pub fn new() -> Self {
        Self {
            user_to_harmony: HashMap::new(),
            harmony_to_user: HashMap::new(),
        }
    }

    /// Normalize and dedupe user function name; record both directions.
    pub fn convert_and_add(&mut self, user_function_name: &str) -> String {
        let mut harmony = self.derive_name(user_function_name);
        if matches!(
            user_function_name,
            "browser.open" | "browser.search" | "browser.find" | "python"
        ) {
            harmony = user_function_name.to_string();
        }
        self.user_to_harmony
            .insert(user_function_name.to_string(), harmony.clone());
        self.harmony_to_user
            .insert(harmony.clone(), user_function_name.to_string());
        harmony
    }

    /// Get the original user name or fall back to the Harmony one.
    pub fn original_from_converted(&self, harmony_function_name: &str) -> String {
        if let Some(user) = self.harmony_to_user.get(harmony_function_name) {
            return user.clone();
        }
        // parity: warn upstream; use harmony name as fallback
        harmony_function_name.to_string()
    }

    /// Best-effort identifier-safe conversion (ASCII-ish, unicode identifier aware).
    pub fn convert_to_valid_chars(&self, user_function_name: &str) -> String {
        let mut out = String::with_capacity(user_function_name.len());
        for ch in user_function_name.chars() {
            if ch == ' ' || ch == '-' || ch == '.' {
                out.push('_');
                continue;
            }
            if ch == '_'
                || ch == '$'
                || ch.is_ascii_alphanumeric()
                || unicode_ident::is_xid_start(ch)
                || unicode_ident::is_xid_continue(ch)
            {
                // Permit a broad set of identifier-ish characters
                out.push(ch);
            } else {
                // drop other characters
            }
        }
        if out.is_empty() {
            return "unnamed".to_string();
        }
        if out
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            let mut with = String::with_capacity(out.len() + 1);
            with.push('_');
            with.push_str(&out);
            return with;
        }
        out
    }

    /// Stable, unique candidate generation.
    fn derive_name(&mut self, user_function_name: &str) -> String {
        let original_candidate = self.convert_to_valid_chars(user_function_name);
        let mut candidate = original_candidate.clone();
        let mut count = 2;
        loop {
            if !self.harmony_to_user.contains_key(&candidate) {
                break;
            }
            candidate = format!("{original_candidate}_{count}");
            count += 1;
        }
        candidate
    }
}

// Minimal tool-related structs compatible with handler API
#[derive(Debug, Clone, Default)]
pub struct Tool {
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Default)]
pub struct ToolFunction {
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

pub mod templating;

mod tests;
