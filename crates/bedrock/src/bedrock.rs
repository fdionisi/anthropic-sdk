use std::pin::Pin;

pub use anthropic::messages;
use anthropic::messages::{
    Content, ContentPart, CreateMessageRequest, CreateMessageRequestWithStream,
    CreateMessageResponse, Event, ImageSource, MediaType, Message, MessageResponse, Messages,
    MessagesStream, Metadata, Role, StopReason, Tool, ToolChoice, Usage,
};
use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use aws_config::SdkConfig;
use aws_sdk_bedrockruntime::types;
use aws_types::request_id::RequestId;
use futures::{Stream, StreamExt};

const DEFAULT_API_VERSION: &str = "bedrock-2023-05-31";

pub enum Model {
    ClaudeThreeDotFiveSonnet,
    ClaudeThreeSonnet,
    ClaudeThreeOpus,
    ClaudeThreeHaiku,
}

impl ToString for Model {
    fn to_string(&self) -> String {
        match self {
            Model::ClaudeThreeDotFiveSonnet => {
                "anthropic.claude-3-5-sonnet-20240620-v1:0".to_string()
            }
            Model::ClaudeThreeSonnet => "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
            Model::ClaudeThreeOpus => "anthropic.claude-3-opus-20240229-v1:0".to_string(),
            Model::ClaudeThreeHaiku => "anthropic.claude-3-5-sonnet-20240620-v1:0".to_string(),
        }
    }
}

pub struct AnthropicBedrock {
    client: aws_sdk_bedrockruntime::Client,
}

impl AnthropicBedrock {
    pub fn new(config: &SdkConfig) -> Self {
        Self {
            client: aws_sdk_bedrockruntime::Client::new(config),
        }
    }
}

#[derive(Debug, serde::Serialize)]
struct BedrockCreateMessageRequest {
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
    anthropic_version: String,
}

impl From<CreateMessageRequestWithStream> for BedrockCreateMessageRequest {
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
            anthropic_version: DEFAULT_API_VERSION.into(),
        }
    }
}

#[async_trait]
impl Messages for AnthropicBedrock {
    async fn messages(&self, request: CreateMessageRequest) -> Result<CreateMessageResponse> {
        let response = self
            .client
            .converse()
            .model_id(request.model.to_owned())
            .set_messages(Some(
                request
                    .messages
                    .iter()
                    .map(|m| {
                        types::Message::builder()
                            .role(match m.role {
                                messages::Role::User => types::ConversationRole::User,
                                messages::Role::Assistant => types::ConversationRole::Assistant,
                            })
                            .set_content(Some(match m.content.clone() {
                                Content::Single(text) => vec![types::ContentBlock::Text(text)],
                                Content::Multi(parts) => parts
                                    .iter()
                                    .map(|part| match part {
                                        messages::ContentPart::Text { text } => {
                                            types::ContentBlock::Text(text.clone())
                                        }
                                        messages::ContentPart::Image { source } => {
                                            types::ContentBlock::Image(
                                                types::ImageBlock::builder()
                                                    .format(match source.media_type {
                                                        MediaType::ImageJpeg => {
                                                            types::ImageFormat::Jpeg
                                                        }
                                                        MediaType::ImagePng => {
                                                            types::ImageFormat::Png
                                                        }
                                                        MediaType::ImageGif => {
                                                            types::ImageFormat::Gif
                                                        }
                                                        MediaType::ImageWebp => {
                                                            types::ImageFormat::Webp
                                                        }
                                                    })
                                                    .source(types::ImageSource::Bytes(
                                                        aws_smithy_types::Blob::new(
                                                            source.data.as_bytes().to_vec(),
                                                        ),
                                                    ))
                                                    .build()
                                                    .unwrap(),
                                            )
                                        }
                                        messages::ContentPart::ToolResult {
                                            tool_use_id,
                                            content,
                                        } => types::ContentBlock::ToolResult(
                                            types::ToolResultBlock::builder()
                                                .tool_use_id(tool_use_id)
                                                .content(types::ToolResultContentBlock::Text(
                                                    content.to_owned(),
                                                ))
                                                .build()
                                                .unwrap(),
                                        ),
                                        messages::ContentPart::ToolUse { id, name, input } => {
                                            types::ContentBlock::ToolUse(
                                                types::ToolUseBlock::builder()
                                                    .tool_use_id(id)
                                                    .name(name)
                                                    .input(aws_smithy_types::Document::String(
                                                        serde_json::to_string(input).unwrap(),
                                                    ))
                                                    .build()
                                                    .unwrap(),
                                            )
                                        }
                                        messages::ContentPart::InputJsonDelta { .. }
                                        | messages::ContentPart::TextDelta { .. } => {
                                            unreachable!()
                                        }
                                    })
                                    .collect(),
                            }))
                            .build()
                            .expect("failed to build Message")
                    })
                    .collect(),
            ))
            .set_system(
                request
                    .system
                    .map(|system| vec![types::SystemContentBlock::Text(system)]),
            )
            .inference_config(
                types::InferenceConfiguration::builder()
                    .set_max_tokens(Some(request.max_tokens as i32))
                    .set_stop_sequences(request.stop_sequences)
                    .set_temperature(request.temperature)
                    .set_top_p(request.top_p)
                    .build(),
            )
            .send()
            .await?;

        let message = response.output().unwrap();
        let message = message.as_message().unwrap();

        Ok(CreateMessageResponse::Message(MessageResponse {
            id: response.request_id().unwrap().to_string(),
            kind: "message".to_string(),
            model: request.model,
            role: Role::Assistant.to_string(),
            content: message
                .content()
                .iter()
                .map(|c| match c {
                    types::ContentBlock::Text(text) => ContentPart::Text {
                        text: text.to_owned(),
                    },
                    types::ContentBlock::Image(image_block) => ContentPart::Image {
                        source: ImageSource {
                            kind: "image".into(),
                            media_type: match image_block.format() {
                                types::ImageFormat::Jpeg => MediaType::ImageJpeg,
                                types::ImageFormat::Png => MediaType::ImagePng,
                                types::ImageFormat::Gif => MediaType::ImageGif,
                                types::ImageFormat::Webp => MediaType::ImageWebp,
                                _ => unreachable!(),
                            },
                            data: String::from_utf8_lossy(
                                image_block.source().unwrap().as_bytes().unwrap().as_ref(),
                            )
                            .to_string(),
                        },
                    },
                    types::ContentBlock::ToolResult(_) => todo!(),
                    types::ContentBlock::ToolUse(_) => todo!(),
                    _ => unreachable!(),
                })
                .collect(),
            stop_reason: Some(match response.stop_reason {
                types::StopReason::EndTurn => StopReason::EndTurn,
                types::StopReason::MaxTokens => StopReason::MaxTokens,
                types::StopReason::StopSequence => StopReason::StopSequence,
                types::StopReason::ToolUse => StopReason::ToolUse,
                _ => unreachable!(),
            }),
            stop_sequence: None,
            usage: Usage {
                input_tokens: None,
                output_tokens: 0,
            },
        }))
    }
}

#[async_trait]
impl MessagesStream for AnthropicBedrock {
    async fn messages_stream(
        &self,
        request: CreateMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Event>> + Send>>> {
        let response = self
            .client
            .converse_stream()
            .model_id(request.model.to_owned())
            .set_messages(Some(
                request
                    .messages
                    .iter()
                    .map(|m| {
                        types::Message::builder()
                            .role(match m.role {
                                messages::Role::User => types::ConversationRole::User,
                                messages::Role::Assistant => types::ConversationRole::Assistant,
                            })
                            .set_content(Some(match m.content.clone() {
                                Content::Single(text) => vec![types::ContentBlock::Text(text)],
                                Content::Multi(parts) => parts
                                    .iter()
                                    .map(|part| match part {
                                        messages::ContentPart::Text { text } => {
                                            types::ContentBlock::Text(text.clone())
                                        }
                                        messages::ContentPart::Image { source } => {
                                            types::ContentBlock::Image(
                                                types::ImageBlock::builder()
                                                    .format(match source.media_type {
                                                        MediaType::ImageJpeg => {
                                                            types::ImageFormat::Jpeg
                                                        }
                                                        MediaType::ImagePng => {
                                                            types::ImageFormat::Png
                                                        }
                                                        MediaType::ImageGif => {
                                                            types::ImageFormat::Gif
                                                        }
                                                        MediaType::ImageWebp => {
                                                            types::ImageFormat::Webp
                                                        }
                                                    })
                                                    .source(types::ImageSource::Bytes(
                                                        aws_smithy_types::Blob::new(
                                                            source.data.as_bytes().to_vec(),
                                                        ),
                                                    ))
                                                    .build()
                                                    .unwrap(),
                                            )
                                        }
                                        messages::ContentPart::ToolResult {
                                            tool_use_id,
                                            content,
                                        } => types::ContentBlock::ToolResult(
                                            types::ToolResultBlock::builder()
                                                .tool_use_id(tool_use_id)
                                                .content(types::ToolResultContentBlock::Text(
                                                    content.to_owned(),
                                                ))
                                                .build()
                                                .unwrap(),
                                        ),
                                        messages::ContentPart::ToolUse { id, name, input } => {
                                            types::ContentBlock::ToolUse(
                                                types::ToolUseBlock::builder()
                                                    .tool_use_id(id)
                                                    .name(name)
                                                    .input(aws_smithy_types::Document::String(
                                                        serde_json::to_string(input).unwrap(),
                                                    ))
                                                    .build()
                                                    .unwrap(),
                                            )
                                        }
                                        messages::ContentPart::InputJsonDelta { .. }
                                        | messages::ContentPart::TextDelta { .. } => {
                                            unreachable!()
                                        }
                                    })
                                    .collect(),
                            }))
                            .build()
                            .expect("failed to build Message")
                    })
                    .collect(),
            ))
            .set_system(
                request
                    .system
                    .map(|system| vec![types::SystemContentBlock::Text(system)]),
            )
            .inference_config(
                types::InferenceConfiguration::builder()
                    .set_max_tokens(Some(request.max_tokens as i32))
                    .set_stop_sequences(request.stop_sequences)
                    .set_temperature(request.temperature)
                    .set_top_p(request.top_p)
                    .build(),
            )
            .send()
            .await?;

        Ok(stream! {
            let mut s = response.stream;
            while let Ok(Some(event)) = s.recv().await {
                yield match event {
                    types::ConverseStreamOutput::ContentBlockDelta(block_delta) =>
                        Ok(Event::ContentBlockDelta {
                            index: block_delta.content_block_index() as u64,
                            delta: match block_delta.delta() {
                                Some(content_block_delta) => match content_block_delta {
                                    types::ContentBlockDelta::Text(text) =>
                                        ContentPart::TextDelta { text: text.to_owned() },
                                    types::ContentBlockDelta::ToolUse(tool_use) =>
                                        ContentPart::InputJsonDelta {
                                            partial: tool_use.input.to_owned()
                                        },
                                    _ => unreachable!(),
                                },
                                None => unreachable!(),
                            }
                        }),
                    types::ConverseStreamOutput::ContentBlockStart(block_start) => Ok(Event::ContentBlockStart {
                        index: block_start.content_block_index as u64,
                        content_block: match block_start.start {
                            Some(start) => match start {
                                types::ContentBlockStart::ToolUse(tool_use) => ContentPart::ToolUse {
                                    id: tool_use.tool_use_id,
                                    name: tool_use.name,
                                    input: "{}".into(),
                                },
                                _ => unreachable!()
                            },
                            None => ContentPart::Text { text: "".into() },
                        }
                    }),
                    types::ConverseStreamOutput::ContentBlockStop(block_stop) =>
                        Ok(Event::ContentBlockStop {
                            index: block_stop.content_block_index as u64,
                        }),
                    types::ConverseStreamOutput::MessageStart(_) => continue,
                    types::ConverseStreamOutput::MessageStop(_) => Ok(Event::MessageStop),
                    _ => unreachable!(),
                }
            }
        }
        .boxed())
    }
}

#[cfg(test)]
mod tests {
    use aws_config::BehaviorVersion;

    use super::*;

    #[tokio::test]
    async fn test_messages() -> Result<()> {
        let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
        let bedrock = AnthropicBedrock::new(&config);
        let request = CreateMessageRequest {
            model: Model::ClaudeThreeDotFiveSonnet.to_string(),
            messages: vec![Message {
                role: messages::Role::User,
                content: Content::Single("Hello".into()),
            }],
            system: Some("system".into()),
            max_tokens: 100,
            stop_sequences: None,
            temperature: Some(0.5),
            top_p: None,
            metadata: None,
            tool_choice: None,
            tools: None,
            top_k: None,
        };

        let _ = bedrock.messages(request).await?;

        Ok(())
    }
}
