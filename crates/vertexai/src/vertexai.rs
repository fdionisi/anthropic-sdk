use std::{str::FromStr, sync::Arc};

use anthropic::messages::{
    Content, CreateMessageRequestWithStream, Message, Metadata, Requester, Tool, ToolChoice,
};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use google_cloud_auth::{project::Config, token::DefaultTokenSourceProvider};
use google_cloud_token::{TokenSource, TokenSourceProvider as _};
use http_client::{
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE},
        Method, Request,
    },
    AsyncBody, HttpClient, RequestBuilderExt,
};

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
            Model::ClaudeThreeDotFiveSonnet => "claude-3-5-sonnet-v2@20241022".to_string(),
            Model::ClaudeThreeSonnet => "claude-3-sonnet@20240229".to_string(),
            Model::ClaudeThreeOpus => "claude-3-opus@20240229".to_string(),
            Model::ClaudeThreeHaiku => "claude-3-haiku@20240307".to_string(),
        }
    }
}

impl FromStr for Model {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "claude-3-5-sonnet@20240620" | "claude-3-5-sonnet-v2@20241022" => {
                Ok(Model::ClaudeThreeDotFiveSonnet)
            }
            "claude-3-sonnet@20240229" => Ok(Model::ClaudeThreeSonnet),
            "claude-3-opus@20240229" => Ok(Model::ClaudeThreeOpus),
            "claude-3-haiku@20240307" => Ok(Model::ClaudeThreeHaiku),
            _ => Err(anyhow::anyhow!("model not supported: {}", s)),
        }
    }
}

pub struct AnthropicVertexAi {
    http_client: Arc<dyn HttpClient>,
    project: String,
    region: String,
    token_source: Arc<dyn TokenSource>,
}

pub struct AnthropicVertexAiBuilder {
    project: Option<String>,
    region: Option<String>,
    http_client: Option<Arc<dyn HttpClient>>,
}

impl AnthropicVertexAi {
    pub fn builder() -> AnthropicVertexAiBuilder {
        AnthropicVertexAiBuilder {
            project: None,
            region: None,
            http_client: None,
        }
    }
}

impl AnthropicVertexAiBuilder {
    pub fn with_project(mut self, project: String) -> Self {
        self.project = Some(project);
        self
    }

    pub fn with_region(mut self, region: String) -> Self {
        self.region = Some(region);
        self
    }

    pub fn with_http_client(mut self, http_client: Arc<dyn HttpClient>) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub async fn build(&self) -> Result<AnthropicVertexAi> {
        let config = Config {
            audience: None,
            scopes: Some(&["https://www.googleapis.com/auth/cloud-platform"]),
            sub: None,
        };

        let tsp = DefaultTokenSourceProvider::new(config).await?;
        let ts = tsp.token_source();

        Ok(AnthropicVertexAi {
            project: self.project.to_owned().unwrap(),
            region: self.region.to_owned().unwrap(),
            token_source: ts,
            http_client: self.http_client.to_owned().unwrap(),
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
    system: Option<Content>,
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
            system: value.create_message_request.system.map(|s| s.into()),
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

#[async_trait]
impl Requester for AnthropicVertexAi {
    fn http_client(&self) -> Arc<dyn HttpClient> {
        self.http_client.clone()
    }

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

    async fn request_builder(
        &self,
        url: String,
        body: CreateMessageRequestWithStream,
    ) -> Result<Request<AsyncBody>> {
        let mut req = Request::builder().method(Method::POST).uri(url);

        if body.stream {
            req = req.header("X-Stainless-Helper-Method", "stream");
        }

        Ok(req
            .header("x-goog-user-project", &self.project)
            .header(
                AUTHORIZATION,
                self.token_source
                    .token()
                    .await
                    .map_err(|err| anyhow!("{:?}", err))?,
            )
            .header(CONTENT_TYPE, "application/json")
            .json(dbg!(VertexAiCreateMessageRequest::from(body)))?)
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use http_client_reqwest::HttpClientReqwest;

    use crate::messages::{CreateMessageRequest, Message, Messages, MessagesStream};

    use super::*;

    #[tokio::test]
    async fn test_messages() -> Result<()> {
        let client = AnthropicVertexAi::builder()
            .with_project(std::env::var("GCLOUD_PROJECT_ID")?)
            .with_region(std::env::var("GCLOUD_REGION")?)
            .with_http_client(Arc::new(HttpClientReqwest::default()))
            .build()
            .await?;

        let _ = dbg!(
            client
                .messages(
                    CreateMessageRequest::builder()
                        .model(Model::ClaudeThreeDotFiveSonnet)
                        .messages(vec![Message::user("Hi!".into())])
                        .max_tokens(1024)
                        .build()?,
                )
                .await?
        );

        assert!(false);

        Ok(())
    }

    #[tokio::test]
    async fn test_messages_stream() -> Result<()> {
        let client = AnthropicVertexAi::builder()
            .with_project(std::env::var("GCLOUD_PROJECT_ID")?)
            .with_region(std::env::var("GCLOUD_REGION")?)
            .with_http_client(Arc::new(HttpClientReqwest::default()))
            .build()
            .await?;

        let mut s = client
            .messages_stream(
                CreateMessageRequest::builder()
                    .model(Model::ClaudeThreeDotFiveSonnet)
                    .messages(vec![Message::user("Hi!".into())])
                    .max_tokens(100)
                    .build()?,
            )
            .await?;

        while let Some(response) = s.next().await {
            response?;
        }

        assert!(false);

        Ok(())
    }
}
