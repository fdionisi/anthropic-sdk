pub mod messages;

use std::str::FromStr;

use anyhow::Result;
use async_trait::async_trait;
use messages::{CreateMessageRequestWithStream, Requester};
use reqwest::{Client, RequestBuilder};
use secrecy::{ExposeSecret, SecretString};

const DEFAULT_API_ENDPOINT: &str = "https://api.anthropic.com";
const DEFAULT_API_VERSION: &str = "2023-06-01";

pub enum Model {
    ClaudeThreeDotFiveSonnet,
    ClaudeThreeSonnet,
    ClaudeThreeOpus,
    ClaudeThreeHaiku,
}

impl ToString for Model {
    fn to_string(&self) -> String {
        match self {
            Model::ClaudeThreeDotFiveSonnet => "claude-3-5-sonnet-20240620".to_string(),
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
            "claude-3-5-sonnet-20240620" => Ok(Model::ClaudeThreeDotFiveSonnet),
            "claude-3-sonnet-20240229" => Ok(Model::ClaudeThreeSonnet),
            "claude-3-opus-20240229" => Ok(Model::ClaudeThreeOpus),
            "claude-3-haiku-20240307" => Ok(Model::ClaudeThreeHaiku),
            _ => Err(anyhow::anyhow!("model not supported: {}", s)),
        }
    }
}

pub struct Anthropic {
    api_key: SecretString,
    client: Client,
}

pub struct AnthropicBuilder {
    api_key: Option<SecretString>,
}

impl Anthropic {
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder { api_key: None }
    }
}

impl AnthropicBuilder {
    pub fn api_key<S>(&mut self, api_key: S) -> &mut Self
    where
        S: AsRef<str>,
    {
        self.api_key = Some(SecretString::new(api_key.as_ref().to_string()));
        self
    }

    pub fn build(&self) -> Result<Anthropic> {
        Ok(Anthropic {
            api_key: self
                .api_key
                .to_owned()
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok().map(|s| s.into()))
                .ok_or_else(|| anyhow::anyhow!("API key is required"))?,
            client: Client::new(),
        })
    }
}

#[async_trait]
impl Requester for Anthropic {
    fn base_url(&self) -> String {
        DEFAULT_API_ENDPOINT.into()
    }

    fn endpoint_url(&self, _: &CreateMessageRequestWithStream) -> String {
        "/v1/messages".into()
    }

    async fn request_builder(
        &self,
        url: String,
        body: CreateMessageRequestWithStream,
    ) -> Result<RequestBuilder> {
        let mut req = self.client.post(url);
        if body.stream {
            req = req.header("X-Stainless-Helper-Method", "stream");
        }

        Ok(req
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", DEFAULT_API_VERSION)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(serde_json::to_string(&body)?))
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use messages::MessagesStream;

    use crate::messages::{CreateMessageRequest, Message, Messages};

    use super::*;

    #[tokio::test]
    async fn test_messages() -> Result<()> {
        let client = Anthropic::builder().build()?;

        let _ = client
            .messages(
                CreateMessageRequest::builder()
                    .model(Model::ClaudeThreeHaiku)
                    .messages(vec![Message::user(vec!["Hi!".into()])])
                    .max_tokens(100)
                    .build()?,
            )
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_messages_stream() -> Result<()> {
        let client = Anthropic::builder().build()?;

        let mut s = client
            .messages_stream(
                CreateMessageRequest::builder()
                    .model(Model::ClaudeThreeHaiku)
                    .messages(vec![Message::user(vec!["Hi!".into()])])
                    .max_tokens(100)
                    .build()?,
            )
            .await?;

        while let Some(response) = s.next().await {
            dbg!(response)?;
        }

        Ok(())
    }
}
