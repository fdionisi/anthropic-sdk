use std::pin::Pin;

use anyhow::{anyhow, Result};
use async_stream::stream;
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::RequestBuilder;
use reqwest_eventsource::{Event as SsrEvent, EventSource};
use serde_json::Value;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::User => write!(f, "User"),
            Role::Assistant => write!(f, "Assistant"),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    pub fn user(content: Vec<Content>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaType {
    #[serde(rename = "image/jpeg")]
    ImageJpeg,
    #[serde(rename = "image/png")]
    ImagePng,
    #[serde(rename = "image/gif")]
    ImageGif,
    #[serde(rename = "image/webp")]
    ImageWebp,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(rename = "media_type")]
    pub media_type: MediaType,
    pub data: String,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    TextDelta {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    InputJsonDelta {
        partial: String,
    },
}

impl<S> From<S> for Content
where
    S: AsRef<str>,
{
    fn from(text: S) -> Self {
        Self::Text {
            text: text.as_ref().to_string(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Metadata {
    #[serde(rename = "user_id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceKind {
    Auto,
    Any,
    Tool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolChoice {
    #[serde(rename = "type")]
    pub kind: ToolChoiceKind,
}

impl From<ToolChoiceKind> for ToolChoice {
    fn from(kind: ToolChoiceKind) -> Self {
        Self { kind }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolInputSchema {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Value>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Tool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub name: String,
    #[serde(rename = "input_schema")]
    pub input_schema: ToolInputSchema,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct CreateMessageRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(rename = "max_tokens")]
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "stop_sequences")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "tool_choice")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "top_k")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "top_p")]
    pub top_p: Option<f32>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct CreateMessageRequestWithStream {
    #[serde(flatten)]
    pub create_message_request: CreateMessageRequest,
    pub stream: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct CreateMessageRequest_ {
    #[serde(flatten)]
    pub create_message_request: CreateMessageRequest,
    pub stream: Option<bool>,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageResponseKind {
    Message,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StopReason {
    #[serde(rename = "end_turn")]
    EndTurn,
    #[serde(rename = "max_tokens")]
    MaxTokens,
    #[serde(rename = "stop_sequence")]
    StopSequence,
    #[serde(rename = "tool_use")]
    ToolUse,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Usage {
    #[serde(rename = "input_tokens")]
    pub input_tokens: Option<u32>,
    #[serde(rename = "output_tokens")]
    pub output_tokens: u32,
}

#[derive(Debug, serde::Deserialize)]
pub struct MessageResponse {
    pub id: String,
    pub model: String,
    pub role: String,
    pub content: Vec<Content>,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CreateMessageResponse {
    Message(MessageResponse),
    Error { error: ErrorDetails },
}

#[derive(Debug, serde::Deserialize)]
pub struct ErrorDetails {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}

pub struct CreateMessageRequestBuilder {
    model: Option<String>,
    messages: Option<Vec<Message>>,
    max_tokens: Option<u32>,
    metadata: Option<Metadata>,
    stop_sequences: Option<Vec<String>>,
    system: Option<String>,
    temperature: Option<f32>,
    tool_choice: Option<ToolChoice>,
    tools: Option<Vec<Tool>>,
    top_k: Option<u32>,
    top_p: Option<f32>,
}

impl CreateMessageRequest {
    pub fn builder() -> CreateMessageRequestBuilder {
        CreateMessageRequestBuilder {
            model: None,
            messages: None,
            max_tokens: None,
            metadata: None,
            stop_sequences: None,
            system: None,
            temperature: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
        }
    }
}

impl CreateMessageRequestBuilder {
    pub fn model<S>(mut self, model: S) -> Self
    where
        S: ToString,
    {
        self.model = Some(model.to_string());
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    pub fn system(mut self, system: String) -> Self {
        self.system = Some(system);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn build(self) -> Result<CreateMessageRequest> {
        Ok(CreateMessageRequest {
            model: self.model.ok_or_else(|| anyhow!("model is required"))?,
            messages: self
                .messages
                .ok_or_else(|| anyhow!("messages is required"))?,
            max_tokens: self
                .max_tokens
                .ok_or_else(|| anyhow!("max_tokens is required"))?,
            metadata: self.metadata,
            stop_sequences: self.stop_sequences,
            system: self.system,
            temperature: self.temperature,
            tool_choice: self.tool_choice,
            tools: self.tools,
            top_k: self.top_k,
            top_p: self.top_p,
        })
    }
}

#[async_trait]
pub trait Requester {
    fn base_url(&self) -> String;

    fn endpoint_url(&self, body: &CreateMessageRequestWithStream) -> String;

    async fn request_builder(
        &self,
        url: String,
        body: CreateMessageRequestWithStream,
    ) -> Result<RequestBuilder>;
}

#[async_trait]
pub trait Messages {
    async fn messages(&self, request: CreateMessageRequest) -> Result<CreateMessageResponse>;
}

#[derive(Debug, serde::Deserialize)]
pub struct EventMessageDelta {
    pub stop_reason: StopReason,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Ping,
    MessageStart {
        message: MessageResponse,
    },
    ContentBlockStart {
        index: u64,
        content_block: Content,
    },
    ContentBlockDelta {
        index: u64,
        delta: Content,
    },
    ContentBlockStop {
        index: u64,
    },
    MessageDelta {
        delta: EventMessageDelta,
        usage: Usage,
    },
    MessageStop,
    Error(ErrorDetails),
}

#[async_trait]
pub trait MessagesStream {
    async fn messages_stream(
        &self,
        request: CreateMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Event>> + Send>>>;
}

#[async_trait]
impl<T> Messages for T
where
    T: Requester + Sync,
{
    async fn messages(&self, request: CreateMessageRequest) -> Result<CreateMessageResponse> {
        let create_message_request_with_stream = CreateMessageRequestWithStream {
            create_message_request: request,
            stream: false,
        };

        Ok(self
            .request_builder(
                format!(
                    "{}{}",
                    self.base_url(),
                    self.endpoint_url(&create_message_request_with_stream)
                ),
                create_message_request_with_stream,
            )
            .await?
            .send()
            .await?
            .json()
            .await?)
    }
}

#[async_trait]
impl<T> MessagesStream for T
where
    T: Requester + Sync,
{
    async fn messages_stream(
        &self,
        request: CreateMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Event>> + Send>>> {
        let create_message_request_with_stream = CreateMessageRequestWithStream {
            create_message_request: request,
            stream: true,
        };

        let mut es = EventSource::new(
            self.request_builder(
                format!(
                    "{}{}",
                    self.base_url(),
                    self.endpoint_url(&create_message_request_with_stream)
                ),
                create_message_request_with_stream,
            )
            .await?,
        )?;

        Ok(stream! {
            while let Some(event) = es.next().await {
                match event {
                    Ok(SsrEvent::Open) => continue,
                    Ok(SsrEvent::Message(message_event)) => {
                        yield Ok(serde_json::from_str::<Event>(&message_event.data,)?)
                    },
                    Err(err) => {
                        es.close();
                        match err {
                            reqwest_eventsource::Error::StreamEnded => continue,
                            _ => yield Err(anyhow!("Unknown {}", err)),
                        }
                    }
                }
            }
        }
        .boxed())
    }
}
