use eyre::{Result, eyre};
use openai_harmony::chat::{Author, Content, Message as OpenAiMessage, Role as OpenAiRole};
use openai_harmony::{
    HarmonyEncoding, HarmonyEncodingName, StreamableParser, load_harmony_encoding,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    Developer,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub name: Option<String>,
    pub recipient: Option<String>,
    pub channel: Option<String>,
    pub content_type: Option<String>,
}

impl Message {
    pub fn text((role, content): (Role, String)) -> Self {
        Self {
            role,
            content,
            name: None,
            recipient: None,
            channel: None,
            content_type: None,
        }
    }

    pub fn assistant_final(content: impl Into<String>) -> Self {
        Self::text((Role::Assistant, content.into())).with_channel("final")
    }

    pub fn assistant_analysis(content: impl Into<String>) -> Self {
        Self::text((Role::Assistant, content.into())).with_channel("analysis")
    }

    pub fn assistant_tool_call(name: impl Into<String>, arguments_json: impl Into<String>) -> Self {
        Self::text((Role::Assistant, arguments_json.into()))
            .with_channel("commentary")
            .with_recipient(name)
            .with_content_type("<|constrain|>json")
    }

    pub fn tool_result(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self::text((Role::Tool, content.into()))
            .with_name(name)
            .with_channel("commentary")
            .with_recipient("assistant")
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_recipient(mut self, recipient: impl Into<String>) -> Self {
        self.recipient = Some(recipient.into());
        self
    }

    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = Some(channel.into());
        self
    }

    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }
}

impl From<(Role, String)> for Message {
    fn from(value: (Role, String)) -> Self {
        Self::text(value)
    }
}

impl From<(Role, &str)> for Message {
    fn from((role, content): (Role, &str)) -> Self {
        Self::text((role, content.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct CompletionPlan {
    pub tokens: Vec<u32>,
    pub stop_tokens: Vec<u32>,
    pub assistant_action_stop_tokens: Vec<u32>,
}

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

    pub fn completion_plan(&self, messages: &[Message]) -> Result<CompletionPlan> {
        Ok(CompletionPlan {
            tokens: self.render_completion_tokens(messages)?,
            stop_tokens: self.stop_tokens()?,
            assistant_action_stop_tokens: self.assistant_action_stop_tokens()?,
        })
    }

    pub fn decode_utf8(&self, tokens: &[u32]) -> Result<String> {
        self.encoding
            .tokenizer()
            .decode_utf8(tokens.iter().copied())
            .map_err(|error| eyre!(error))
    }

    pub fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>> {
        self.encoding
            .tokenizer()
            .decode_bytes(tokens.iter().copied())
            .map_err(|error| eyre!(error))
    }

    pub fn stop_tokens(&self) -> Result<Vec<u32>> {
        let tokens = self
            .encoding
            .stop_tokens()
            .map_err(|error| eyre!(error.to_string()))?;
        Ok(sorted_tokens(tokens))
    }

    pub fn assistant_action_stop_tokens(&self) -> Result<Vec<u32>> {
        let tokens = self
            .encoding
            .stop_tokens_for_assistant_actions()
            .map_err(|error| eyre!(error.to_string()))?;
        Ok(sorted_tokens(tokens))
    }

    pub fn stream_parser(&self, role: Option<Role>) -> Result<TokenStreamParser> {
        let role = role.map(to_openai_role);
        let parser = StreamableParser::new(self.encoding.clone(), role)
            .map_err(|error| eyre!(error.to_string()))?;
        Ok(TokenStreamParser { parser })
    }
}

pub struct TokenStreamParser {
    parser: StreamableParser,
}

impl TokenStreamParser {
    pub fn push(&mut self, token: u32) -> Result<Option<String>> {
        self.parser
            .process(token)
            .map_err(|error| eyre!(error.to_string()))?;
        self.parser
            .last_content_delta()
            .map_err(|error| eyre!(error.to_string()))
    }

    pub fn finish(mut self) -> Result<Vec<Message>> {
        self.parser
            .process_eos()
            .map_err(|error| eyre!(error.to_string()))?;
        self.parser
            .into_messages()
            .into_iter()
            .map(from_openai_message)
            .collect()
    }
}

fn to_openai_message(source: &Message) -> OpenAiMessage {
    let author = if source.role == Role::Tool {
        let name = source.name.clone().unwrap_or_default();
        Author::new(OpenAiRole::Tool, name)
    } else {
        Author::from(to_openai_role(source.role))
    };

    let mut message = OpenAiMessage::from_author_and_content(author, source.content.clone());
    if let Some(recipient) = &source.recipient {
        message = message.with_recipient(recipient.clone());
    }
    if let Some(channel) = &source.channel {
        message = message.with_channel(channel.clone());
    }
    if let Some(content_type) = &source.content_type {
        message = message.with_content_type(content_type.clone());
    }
    message
}

fn from_openai_message(message: OpenAiMessage) -> Result<Message> {
    let content = message
        .content
        .into_iter()
        .map(content_text)
        .collect::<Result<Vec<_>>>()?
        .join("");
    Ok(Message {
        role: from_openai_role(message.author.role),
        content,
        name: message.author.name,
        recipient: message.recipient,
        channel: message.channel,
        content_type: message.content_type,
    })
}

fn content_text(content: Content) -> Result<String> {
    match content {
        Content::Text(text) => Ok(text.text),
        other => Err(eyre!("unsupported parsed Harmony content: {other:?}")),
    }
}

fn to_openai_role(role: Role) -> OpenAiRole {
    match role {
        Role::User => OpenAiRole::User,
        Role::Assistant => OpenAiRole::Assistant,
        Role::System => OpenAiRole::System,
        Role::Developer => OpenAiRole::Developer,
        Role::Tool => OpenAiRole::Tool,
    }
}

fn from_openai_role(role: OpenAiRole) -> Role {
    match role {
        OpenAiRole::User => Role::User,
        OpenAiRole::Assistant => Role::Assistant,
        OpenAiRole::System => Role::System,
        OpenAiRole::Developer => Role::Developer,
        OpenAiRole::Tool => Role::Tool,
    }
}

fn sorted_tokens(tokens: std::collections::HashSet<u32>) -> Vec<u32> {
    let mut tokens = tokens.into_iter().collect::<Vec<_>>();
    tokens.sort_unstable();
    tokens
}
