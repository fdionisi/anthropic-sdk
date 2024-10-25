pub mod messages;

use std::{str::FromStr, sync::Arc};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use http_client::{
    http::{header::CONTENT_TYPE, Method, Request},
    AsyncBody, HttpClient, RequestBuilderExt,
};
use messages::{CreateMessageRequestWithStream, Requester};
use secrecy::{ExposeSecret, SecretString};

const DEFAULT_API_ENDPOINT: &str = "https://api.anthropic.com";
const DEFAULT_API_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub enum Model {
    ClaudeThreeDotFiveSonnet,
    ClaudeThreeSonnet,
    ClaudeThreeOpus,
    ClaudeThreeHaiku,
}

impl ToString for Model {
    fn to_string(&self) -> String {
        match self {
            Model::ClaudeThreeDotFiveSonnet => "claude-3-5-sonnet-latest".to_string(),
            Model::ClaudeThreeSonnet => "claude-3-sonnet-20240229".to_string(),
            Model::ClaudeThreeOpus => "claude-3-opus-20240229".to_string(),
            Model::ClaudeThreeHaiku => "claude-3-haiku-20240307".to_string(),
        }
    }
}

impl FromStr for Model {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "claude-3-5-sonnet-latest" | "claude-3-5-sonnet-20240620" => {
                Ok(Model::ClaudeThreeDotFiveSonnet)
            }
            "claude-3-sonnet-20240229" => Ok(Model::ClaudeThreeSonnet),
            "claude-3-opus-20240229" => Ok(Model::ClaudeThreeOpus),
            "claude-3-haiku-20240307" => Ok(Model::ClaudeThreeHaiku),
            _ => Err(anyhow::anyhow!("model not supported: {}", s)),
        }
    }
}

pub struct Anthropic {
    api_key: SecretString,
    base_url: String,
    http_client: Arc<dyn HttpClient>,
}

#[derive(Clone)]
pub struct AnthropicBuilder {
    api_key: Option<SecretString>,
    base_url: Option<String>,
    http_client: Option<Arc<dyn HttpClient>>,
}

impl Anthropic {
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder {
            api_key: None,
            base_url: None,
            http_client: None,
        }
    }
}

impl AnthropicBuilder {
    pub fn with_api_key<S>(&mut self, api_key: S) -> &mut Self
    where
        S: AsRef<str>,
    {
        self.api_key = Some(SecretString::new(api_key.as_ref().to_string()));
        self
    }

    pub fn with_base_url<S>(&mut self, base_url: S) -> &mut Self
    where
        S: AsRef<str>,
    {
        self.base_url = Some(base_url.as_ref().to_string());
        self
    }

    pub fn with_http_client(&mut self, http_client: Arc<dyn HttpClient>) -> &mut Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn build(&self) -> Result<Anthropic> {
        Ok(Anthropic {
            api_key: self
                .api_key
                .to_owned()
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok().map(|s| s.into()))
                .ok_or_else(|| anyhow::anyhow!("API key is required"))?,
            base_url: self
                .base_url
                .to_owned()
                .or_else(|| std::env::var("ANTHROPIC_BASE_URL").ok().map(|s| s.into()))
                .unwrap_or_else(|| DEFAULT_API_ENDPOINT.into()),
            http_client: self
                .http_client
                .to_owned()
                .ok_or_else(|| anyhow!("http client is required"))?,
        })
    }
}

#[async_trait]
impl Requester for Anthropic {
    fn http_client(&self) -> Arc<dyn HttpClient> {
        self.http_client.clone()
    }

    fn base_url(&self) -> String {
        self.base_url.to_owned()
    }

    fn endpoint_url(&self, _: &CreateMessageRequestWithStream) -> String {
        "/v1/messages".into()
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
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", DEFAULT_API_VERSION)
            .header(CONTENT_TYPE, "application/json")
            .json(body)?)
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use http_client_reqwest::HttpClientReqwest;
    use messages::MessagesStream;

    use crate::messages::{CreateMessageRequest, Message, Messages};

    use super::*;

    #[tokio::test]
    async fn test_messages() -> Result<()> {
        let client = Anthropic::builder()
            .with_http_client(Arc::new(HttpClientReqwest::default()))
            .build()?;

        let _ = dbg!(
            client
                .messages(
                    CreateMessageRequest::builder()
                        .model(Model::ClaudeThreeHaiku)
                        .messages(vec![Message::user("Hi!".into())])
                        .max_tokens(100)
                        .build()?,
                )
                .await?
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_messages_stream() -> Result<()> {
        let client = Anthropic::builder()
            .with_http_client(Arc::new(HttpClientReqwest::default()))
            .build()?;

        let mut s = client
            .messages_stream(
                CreateMessageRequest::builder()
                    .model(Model::ClaudeThreeHaiku)
                    .messages(vec![Message::user("Hi!".into())])
                    .max_tokens(100)
                    .build()?,
            )
            .await?;

        dbg!(s.next().await);
        while let Some(response) = s.next().await {
            dbg!(response)?;
        }

        Ok(())
    }
}
