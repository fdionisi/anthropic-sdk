use anthropic::messages::{
    CreateMessageRequestWithStream, Message, Metadata, Requester, Tool, ToolChoice,
};
use anyhow::Result;
use reqwest::{Client, IntoUrl, RequestBuilder};
use secrecy::{ExposeSecret, SecretString};

pub use anthropic::messages;

const DEFAULT_API_VERSION: &str = "vertex-2023-10-16";

pub enum Model {
    ClaudeThreeDotFiveSonnet,
    ClaudeThreeSonnet,
    ClaudeThreeOpus,
    ClaudeThreeHaiku,
}

impl ToString for Model {
    fn to_string(&self) -> String {
        match self {
            Model::ClaudeThreeDotFiveSonnet => "claude-3-5-sonnet@20240620".to_string(),
            Model::ClaudeThreeSonnet => "claude-3-sonnet@20240229".to_string(),
            Model::ClaudeThreeOpus => "claude-3-opus@20240229".to_string(),
            Model::ClaudeThreeHaiku => "claude-3-haiku@20240307".to_string(),
        }
    }
}

pub struct AnthropicVertexAi {
    client: Client,
    project: String,
    region: String,
    api_key: SecretString,
}

pub struct AnthropicVertexAiBuilder {
    project: Option<String>,
    region: Option<String>,
    api_key: Option<SecretString>,
}

impl AnthropicVertexAi {
    pub fn builder() -> AnthropicVertexAiBuilder {
        AnthropicVertexAiBuilder {
            project: None,
            region: None,
            api_key: None,
        }
    }
}

impl AnthropicVertexAiBuilder {
    pub fn project(mut self, project: String) -> Self {
        self.project = Some(project);
        self
    }

    pub fn region(mut self, region: String) -> Self {
        self.region = Some(region);
        self
    }

    pub fn api_key<S>(mut self, api_key: S) -> Self
    where
        S: AsRef<str>,
    {
        self.api_key = Some(SecretString::new(api_key.as_ref().to_string()));
        self
    }

    pub fn build(&self) -> Result<AnthropicVertexAi> {
        Ok(AnthropicVertexAi {
            project: self.project.to_owned().unwrap(),
            region: self.region.to_owned().unwrap(),
            api_key: self
                .api_key
                .to_owned()
                .or_else(|| std::env::var("GOOGLECLOUD_API_KEY").ok().map(|s| s.into()))
                .ok_or_else(|| anyhow::anyhow!("API key is required"))?,
            client: Client::new(),
        })
    }
}

#[derive(Debug, serde::Serialize)]
struct VertexAiCreateMessageRequest {
    messages: Vec<Message>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Metadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    stream: bool,
    anthropic_version: String,
}

impl From<CreateMessageRequestWithStream> for VertexAiCreateMessageRequest {
    fn from(value: CreateMessageRequestWithStream) -> Self {
        Self {
            messages: value.create_message_request.messages,
            max_tokens: value.create_message_request.max_tokens,
            metadata: value.create_message_request.metadata,
            stop_sequences: value.create_message_request.stop_sequences,
            system: value.create_message_request.system,
            temperature: value.create_message_request.temperature,
            tool_choice: value.create_message_request.tool_choice,
            tools: value.create_message_request.tools,
            top_k: value.create_message_request.top_k,
            top_p: value.create_message_request.top_p,
            stream: value.stream,
            anthropic_version: DEFAULT_API_VERSION.into(),
        }
    }
}

impl Requester for AnthropicVertexAi {
    fn base_url(&self) -> String {
        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/anthropic",
            self.region, self.project, self.region
        )
    }

    fn endpoint_url(&self, body: &CreateMessageRequestWithStream) -> String {
        let return_type = if body.stream {
            "streamRawPredict"
        } else {
            "rawPredict"
        };
        format!(
            "/models/{}:{}",
            body.create_message_request.model.to_string(),
            return_type
        )
    }

    fn request_builder<U>(
        &self,
        url: U,
        body: CreateMessageRequestWithStream,
    ) -> Result<RequestBuilder>
    where
        U: IntoUrl,
    {
        let mut req = self.client.post(url);
        if body.stream {
            req = req.header("X-Stainless-Helper-Method", "stream");
        }

        Ok(req
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("x-goog-user-project", &self.project)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(dbg!(serde_json::to_string(
                &VertexAiCreateMessageRequest::from(body)
            ))?))
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use crate::messages::{CreateMessageRequest, Message, Messages, MessagesStream};

    use super::*;

    #[tokio::test]
    async fn test_messages() -> Result<()> {
        let client = AnthropicVertexAi::builder()
            .api_key(std::env::var("GCLOUD_API_KEY")?)
            .project(std::env::var("GCLOUD_PROJECT_ID")?)
            .region(std::env::var("GCLOUD_REGION")?)
            .build()?;

        let response = client
            .messages(
                CreateMessageRequest::builder()
                    .model(Model::ClaudeThreeDotFiveSonnet)
                    .messages(vec![Message::user(vec!["Hi!".into()])])
                    .max_tokens(1024)
                    .build()?,
            )
            .await?;

        dbg!(response);

        Ok(())
    }

    #[tokio::test]
    async fn test_messages_stream() -> Result<()> {
        let client = AnthropicVertexAi::builder()
            .api_key(std::env::var("GCLOUD_API_KEY")?)
            .project(std::env::var("GCLOUD_PROJECT_ID")?)
            .region(std::env::var("GCLOUD_REGION")?)
            .build()?;

        let mut s = client.messages_stream(
            CreateMessageRequest::builder()
                .model(Model::ClaudeThreeDotFiveSonnet)
                .messages(vec![Message::user(vec!["Hi!".into()])])
                .max_tokens(100)
                .build()?,
        )?;

        while let Some(response) = s.next().await {
            dbg!(response)?;
        }

        Ok(())
    }
}
