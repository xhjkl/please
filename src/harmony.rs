//! Canonical Harmony boundary for gpt-oss.
//!
//! Prompt history is rendered straight to token ids through OpenAI's Harmony
//! encoder. Generated token ids are parsed back incrementally for display and
//! tool execution, so literal marker strings inside JSON stay ordinary content.

use eyre::{Result, eyre};
use openai_harmony::chat::{
    Author, Content, Message as OpenAiMessage, Role as OpenAiRole, TextContent,
};
use openai_harmony::{
    HarmonyEncoding, HarmonyEncodingName, ParseOptions, StreamableParser, load_harmony_encoding,
};
use serde_json::Value;

use crate::protocol::Message;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarmonyMessage {
    System(String),
    Developer(String),
    User(String),
    AssistantFinal(String),
    AssistantAnalysis(String),
    AssistantToolCall {
        recipient: String,
        arguments_json: String,
    },
    ToolResult {
        name: String,
        content: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarmonyDelta {
    Answer(String),
    Thinking(String),
}

#[derive(Debug, Clone)]
pub struct HarmonyAdapter {
    encoding: HarmonyEncoding,
    stop_tokens: Vec<u32>,
}

impl HarmonyAdapter {
    pub fn gpt_oss() -> Result<Self> {
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .map_err(|error| eyre!(error.to_string()))?;
        let stop_tokens = sorted_tokens(
            encoding
                .stop_tokens_for_assistant_actions()
                .map_err(|error| eyre!(error.to_string()))?,
        );
        Ok(Self {
            encoding,
            stop_tokens,
        })
    }

    pub fn render_completion_tokens(&self, messages: &[HarmonyMessage]) -> Result<Vec<u32>> {
        let messages = messages.iter().map(to_openai_message).collect::<Vec<_>>();
        self.encoding
            .render_conversation_for_completion(&messages, OpenAiRole::Assistant, None)
            .map_err(|error| eyre!(error.to_string()))
    }

    pub fn render_protocol_tokens(&self, history: &[Message]) -> Result<Vec<u32>> {
        let messages = protocol_to_harmony(history)?;
        self.render_completion_tokens(&messages)
    }

    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }

    pub fn is_stop_token(&self, token: u32) -> bool {
        self.stop_tokens.binary_search(&token).is_ok()
    }

    pub fn output_parser(&self) -> Result<HarmonyOutputParser> {
        HarmonyOutputParser::new(self.encoding.clone())
    }
}

pub struct HarmonyOutputParser {
    parser: StreamableParser,
}

impl HarmonyOutputParser {
    fn new(encoding: HarmonyEncoding) -> Result<Self> {
        let options = ParseOptions { strict: false };
        let parser =
            StreamableParser::new_with_options(encoding, Some(OpenAiRole::Assistant), options)
                .map_err(|error| eyre!(error.to_string()))?;
        Ok(Self { parser })
    }

    pub fn push_token(&mut self, token: u32) -> Result<Option<HarmonyDelta>> {
        self.parser
            .process(token)
            .map_err(|error| eyre!(error.to_string()))?;
        let delta = self
            .parser
            .last_content_delta()
            .map_err(|error| eyre!(error.to_string()))?;
        let Some(delta) = delta else {
            return Ok(None);
        };
        if self.parser.current_recipient().is_some() {
            return Ok(None);
        }
        if self.parser.current_channel().as_deref() == Some("analysis") {
            return Ok(Some(HarmonyDelta::Thinking(delta)));
        }
        Ok(Some(HarmonyDelta::Answer(delta)))
    }

    pub fn finish(&mut self) -> Result<Vec<ToolCall>> {
        self.parser
            .process_eos()
            .map_err(|error| eyre!(error.to_string()))?;
        let mut calls = Vec::new();
        for message in self.parser.messages() {
            if message.author.role != OpenAiRole::Assistant {
                continue;
            }
            let Some(recipient) = message.recipient.as_deref() else {
                continue;
            };
            let name = recipient
                .strip_prefix("functions.")
                .unwrap_or(recipient)
                .to_string();
            let text = message_text(message)?;
            let arguments = serde_json::from_str(&text)
                .map_err(|error| eyre!("error parsing tool call: raw='{text}', err={error}"))?;
            calls.push(ToolCall { name, arguments });
        }
        Ok(calls)
    }
}

pub fn protocol_to_harmony(history: &[Message]) -> Result<Vec<HarmonyMessage>> {
    let mut out = Vec::new();
    for message in history {
        match message {
            Message::System(content) => out.push(HarmonyMessage::System(content.clone())),
            Message::Developer(content) => out.push(HarmonyMessage::Developer(content.clone())),
            Message::User(content) => out.push(HarmonyMessage::User(content.clone())),
            Message::Reasoning(content) => {
                out.push(HarmonyMessage::AssistantAnalysis(content.clone()))
            }
            Message::Assistant(content) => {
                out.push(HarmonyMessage::AssistantFinal(content.clone()))
            }
            Message::Tool(content) => push_tool_history(&mut out, content)?,
        }
    }
    Ok(out)
}

fn push_tool_history(out: &mut Vec<HarmonyMessage>, content: &str) -> Result<()> {
    let value = serde_json::from_str::<Value>(content);
    let Ok(value) = value else {
        out.push(HarmonyMessage::ToolResult {
            name: "tool".to_string(),
            content: content.to_string(),
        });
        return Ok(());
    };

    let name = value
        .get("tool")
        .and_then(Value::as_str)
        .unwrap_or("tool")
        .to_string();
    if let Some(arguments) = value.get("arguments") {
        out.push(HarmonyMessage::AssistantToolCall {
            recipient: format!("functions.{name}"),
            arguments_json: serde_json::to_string(arguments)?,
        });
    }
    if let Some(result) = value.get("result") {
        let content = result
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| serde_json::to_string(result).unwrap_or_else(|_| "null".into()));
        out.push(HarmonyMessage::ToolResult {
            name: name.clone(),
            content,
        });
    }
    if value.get("arguments").is_none() && value.get("result").is_none() {
        out.push(HarmonyMessage::ToolResult {
            name,
            content: content.to_string(),
        });
    }
    Ok(())
}

fn to_openai_message(source: &HarmonyMessage) -> OpenAiMessage {
    match source {
        HarmonyMessage::System(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::System), content)
        }
        HarmonyMessage::Developer(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::Developer), content)
        }
        HarmonyMessage::User(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::User), content)
        }
        HarmonyMessage::AssistantFinal(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::Assistant), content)
                .with_channel("final")
        }
        HarmonyMessage::AssistantAnalysis(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::Assistant), content)
                .with_channel("analysis")
        }
        HarmonyMessage::AssistantToolCall {
            recipient,
            arguments_json,
        } => OpenAiMessage::from_author_and_content(
            Author::from(OpenAiRole::Assistant),
            arguments_json,
        )
        .with_channel("commentary")
        .with_recipient(recipient)
        .with_content_type("<|constrain|>json"),
        HarmonyMessage::ToolResult { name, content } => {
            OpenAiMessage::from_author_and_content(Author::new(OpenAiRole::Tool, name), content)
                .with_channel("commentary")
                .with_recipient("assistant")
        }
    }
}

fn message_text(message: &OpenAiMessage) -> Result<String> {
    let mut text = String::new();
    for content in &message.content {
        match content {
            Content::Text(TextContent { text: content }) => text.push_str(content),
            other => text.push_str(&serde_json::to_string(other)?),
        }
    }
    Ok(text)
}

fn sorted_tokens(tokens: std::collections::HashSet<u32>) -> Vec<u32> {
    let mut tokens = tokens.into_iter().collect::<Vec<_>>();
    tokens.sort_unstable();
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    const START_TOKEN: u32 = 200006;

    #[test]
    fn literal_start_marker_in_user_json_is_content_not_structure() -> Result<()> {
        let harmony = HarmonyAdapter::gpt_oss()?;
        let messages = [HarmonyMessage::User(
            r#"{"literal":"<|start|>","nested":{"still":"content"}}"#.to_string(),
        )];

        let tokens = harmony.render_completion_tokens(&messages)?;
        assert_eq!(count_token(&tokens, START_TOKEN), 2);
        assert!(
            !harmony
                .encoding
                .tokenizer()
                .encode_ordinary("<|start|>")
                .contains(&START_TOKEN)
        );
        Ok(())
    }

    #[test]
    fn literal_start_marker_in_tool_call_json_is_content_not_structure() -> Result<()> {
        let harmony = HarmonyAdapter::gpt_oss()?;
        let messages = [HarmonyMessage::AssistantToolCall {
            recipient: "functions.echo".to_string(),
            arguments_json: r#"{"argument":"<|start|>"}"#.to_string(),
        }];

        let tokens = harmony.render_completion_tokens(&messages)?;
        assert_eq!(count_token(&tokens, START_TOKEN), 2);
        Ok(())
    }

    fn count_token(tokens: &[u32], needle: u32) -> usize {
        tokens.iter().filter(|token| **token == needle).count()
    }
}
