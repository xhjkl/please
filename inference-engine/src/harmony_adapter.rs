use eyre::{Result, eyre};
use openai_harmony::chat::{Author, Message as OpenAiMessage, Role as OpenAiRole};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};

/// App-facing prompt unit. Each variant maps to one canonical Harmony shape, so
/// invalid role/channel/recipient combinations are not representable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
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

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System(content.into())
    }

    pub fn developer(content: impl Into<String>) -> Self {
        Self::Developer(content.into())
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User(content.into())
    }

    pub fn assistant_final(content: impl Into<String>) -> Self {
        Self::AssistantFinal(content.into())
    }

    pub fn assistant_analysis(content: impl Into<String>) -> Self {
        Self::AssistantAnalysis(content.into())
    }

    pub fn assistant_tool_call(
        recipient: impl Into<String>,
        arguments_json: impl Into<String>,
    ) -> Self {
        Self::AssistantToolCall {
            recipient: recipient.into(),
            arguments_json: arguments_json.into(),
        }
    }

    pub fn tool_result(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            name: name.into(),
            content: content.into(),
        }
    }
}

/// Codec boundary around OpenAI's canonical gpt-oss
/// rendering and tokenizer. It turns structured messages into token ids, and
/// generated token ids back into text.
#[derive(Debug, Clone)]
pub struct HarmonyAdapter {
    encoding: HarmonyEncoding,
}

impl HarmonyAdapter {
    pub fn gpt_oss() -> Result<Self> {
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .map_err(|error| eyre!(error.to_string()))?;
        Ok(Self { encoding })
    }

    pub fn render_completion_tokens(&self, messages: &[Message]) -> Result<Vec<u32>> {
        let messages = messages.iter().map(to_openai_message).collect::<Vec<_>>();
        self.encoding
            .render_conversation_for_completion(&messages, OpenAiRole::Assistant, None)
            .map_err(|error| eyre!(error.to_string()))
    }

    pub fn decode_utf8(&self, tokens: &[u32]) -> Result<String> {
        self.encoding
            .tokenizer()
            .decode_utf8(tokens.iter().copied())
            .map_err(|error| eyre!(error))
    }

    pub fn stop_tokens(&self) -> Result<Vec<u32>> {
        let tokens = self
            .encoding
            .stop_tokens()
            .map_err(|error| eyre!(error.to_string()))?;
        Ok(sorted_tokens(tokens))
    }
}

fn to_openai_message(source: &Message) -> OpenAiMessage {
    match source {
        Message::System(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::System), content)
        }
        Message::Developer(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::Developer), content)
        }
        Message::User(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::User), content)
        }
        Message::AssistantFinal(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::Assistant), content)
                .with_channel("final")
        }
        Message::AssistantAnalysis(content) => {
            OpenAiMessage::from_author_and_content(Author::from(OpenAiRole::Assistant), content)
                .with_channel("analysis")
        }
        Message::AssistantToolCall {
            recipient,
            arguments_json,
        } => OpenAiMessage::from_author_and_content(
            Author::from(OpenAiRole::Assistant),
            arguments_json,
        )
        .with_channel("commentary")
        .with_recipient(recipient)
        .with_content_type("<|constrain|>json"),
        Message::ToolResult { name, content } => {
            OpenAiMessage::from_author_and_content(Author::new(OpenAiRole::Tool, name), content)
                .with_channel("commentary")
                .with_recipient("assistant")
        }
    }
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
        let messages = [Message::user(
            r#"{"literal":"<|start|>","nested":{"still":"content"}}"#,
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
        let messages = [Message::assistant_tool_call(
            "functions.echo",
            r#"{"argument":"<|start|>"}"#,
        )];

        let tokens = harmony.render_completion_tokens(&messages)?;
        assert_eq!(count_token(&tokens, START_TOKEN), 2);
        Ok(())
    }

    fn count_token(tokens: &[u32], needle: u32) -> usize {
        tokens.iter().filter(|token| **token == needle).count()
    }
}
