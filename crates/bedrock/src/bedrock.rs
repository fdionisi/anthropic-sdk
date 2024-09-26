use std::{collections::HashSet, pin::Pin};

pub use anthropic::messages;
use anthropic::messages::{
    Content, ContentPart, CreateMessageRequest, CreateMessageRequestWithStream,
    CreateMessageResponse, Event, EventMessageDelta, ImageSource, MediaType, Message,
    MessageResponse, MessageResponseStream, Messages, MessagesStream, Metadata, StopReason, Tool,
    ToolChoice, Usage,
};
use anyhow::{anyhow, Result};
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

fn filter_content_blocks(content: Vec<ContentPart>) -> Vec<ContentPart> {
    if content
        .iter()
        .any(|part| matches!(part, ContentPart::ToolResult { .. }))
    {
        content
            .into_iter()
            .filter(|part| matches!(part, ContentPart::ToolResult { .. }))
            .collect()
    } else {
        content
    }
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

fn attach_tools(
    tool_config: types::builders::ToolConfigurationBuilder,
    tools: Vec<Tool>,
) -> types::builders::ToolConfigurationBuilder {
    tool_config.set_tools(Some(
        tools
            .iter()
            .map(|tool| {
                types::Tool::ToolSpec(
                    types::ToolSpecification::builder()
                        .name(tool.name.clone())
                        .set_description(tool.description.clone())
                        .input_schema(types::ToolInputSchema::Json(
                            serde_json::to_value(&tool.input_schema)
                                .and_then(|val| {
                                    serde_json::from_value::<aws_smithy_types::Document>(val)
                                })
                                .unwrap(),
                        ))
                        .build()
                        .unwrap(),
                )
            })
            .collect(),
    ))
}

fn attach_tool_choice(
    tool_config: types::builders::ToolConfigurationBuilder,
    tool_choice: ToolChoice,
) -> types::builders::ToolConfigurationBuilder {
    tool_config.set_tool_choice(Some(match tool_choice.kind {
        messages::ToolChoiceKind::Auto => {
            types::ToolChoice::Auto(types::AutoToolChoice::builder().build())
        }
        messages::ToolChoiceKind::Any => {
            types::ToolChoice::Any(types::AnyToolChoice::builder().build())
        }
        messages::ToolChoiceKind::Tool => unreachable!(),
    }))
}

fn parse_messages(message: &Message) -> types::Message {
    types::Message::builder()
        .role(match message.role {
            messages::Role::User => types::ConversationRole::User,
            messages::Role::Assistant => types::ConversationRole::Assistant,
        })
        .set_content(Some(match message.content.to_owned() {
            Content::Single(text) => vec![types::ContentBlock::Text(text)],
            Content::Multi(parts) => filter_content_blocks(parts)
                .iter()
                .map(|part| match part {
                    messages::ContentPart::Text { text } => types::ContentBlock::Text(text.clone()),
                    messages::ContentPart::Image { source } => types::ContentBlock::Image(
                        types::ImageBlock::builder()
                            .format(match source.media_type {
                                MediaType::ImageJpeg => types::ImageFormat::Jpeg,
                                MediaType::ImagePng => types::ImageFormat::Png,
                                MediaType::ImageGif => types::ImageFormat::Gif,
                                MediaType::ImageWebp => types::ImageFormat::Webp,
                            })
                            .source(types::ImageSource::Bytes(aws_smithy_types::Blob::new(
                                aws_smithy_types::base64::decode(&source.data)
                                    .map_err(|e| anyhow!("Failed to decode base64: {}", e))
                                    .unwrap(),
                            )))
                            .build()
                            .unwrap(),
                    ),
                    messages::ContentPart::ToolResult {
                        tool_use_id,
                        content,
                    } => types::ContentBlock::ToolResult(
                        types::ToolResultBlock::builder()
                            .tool_use_id(tool_use_id)
                            .content(types::ToolResultContentBlock::Text(content.to_owned()))
                            .build()
                            .unwrap(),
                    ),
                    messages::ContentPart::ToolUse { id, name, input } => {
                        types::ContentBlock::ToolUse(
                            types::ToolUseBlock::builder()
                                .tool_use_id(id)
                                .name(name)
                                .input(serde_json::from_value(input.to_owned()).unwrap())
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
}

fn parse_system(system: Content) -> Vec<types::SystemContentBlock> {
    match system {
        Content::Single(system) => vec![types::SystemContentBlock::Text(system)],
        Content::Multi(parts) => parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => {
                    Some(types::SystemContentBlock::Text(text.to_owned()))
                }
                ContentPart::TextDelta { .. }
                | ContentPart::ToolResult { .. }
                | ContentPart::ToolUse { .. }
                | ContentPart::Image { .. }
                | ContentPart::InputJsonDelta { .. } => None,
            })
            .collect(),
    }
}

#[async_trait]
impl Messages for AnthropicBedrock {
    async fn messages(&self, request: CreateMessageRequest) -> Result<CreateMessageResponse> {
        let mut test_config = types::ToolConfiguration::builder();

        if let Some(tools) = request.tools.to_owned() {
            test_config = attach_tools(test_config, tools);
        }

        if let Some(tool_choice) = request.tool_choice.to_owned() {
            test_config = attach_tool_choice(test_config, tool_choice);
        }

        let mut bd_request = self
            .client
            .converse()
            .model_id(request.model.to_owned())
            .set_messages(Some(request.messages.iter().map(parse_messages).collect()))
            .set_system(request.system.map(parse_system))
            .inference_config(
                types::InferenceConfiguration::builder()
                    .set_max_tokens(Some(request.max_tokens as i32))
                    .set_stop_sequences(request.stop_sequences)
                    .set_temperature(request.temperature)
                    .set_top_p(request.top_p)
                    .build(),
            );

        if request.tools.is_some() {
            bd_request = bd_request.set_tool_config(Some(test_config.build().unwrap()));
        }

        let response = bd_request.send().await?;

        let message = response.output().unwrap();
        let message = message.as_message().unwrap();

        Ok(CreateMessageResponse::Message(MessageResponse {
            id: response.request_id().unwrap().to_string(),
            model: request.model,
            role: "assistant".to_string(),
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
                    types::ContentBlock::ToolResult(tool_result) => ContentPart::ToolResult {
                        tool_use_id: tool_result.tool_use_id().to_string(),
                        content: match tool_result.content().first() {
                            Some(types::ToolResultContentBlock::Text(text)) => text.to_owned(),
                            _ => unreachable!(),
                        },
                    },
                    types::ContentBlock::ToolUse(tool_use) => ContentPart::ToolUse {
                        id: tool_use.tool_use_id().to_string(),
                        name: tool_use.name().to_string(),
                        input: serde_json::to_value(tool_use.input()).unwrap(),
                    },
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
            usage: response
                .usage
                .map(|usage| Usage {
                    input_tokens: Some(usage.input_tokens as u32),
                    output_tokens: usage.output_tokens as u32,
                })
                .unwrap(),
        }))
    }
}

#[async_trait]
impl MessagesStream for AnthropicBedrock {
    async fn messages_stream(
        &self,
        request: CreateMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Event>> + Send>>> {
        let mut test_config = types::ToolConfiguration::builder();

        if let Some(tools) = request.tools.to_owned() {
            test_config = attach_tools(test_config, tools);
        }

        if let Some(tool_choice) = request.tool_choice.to_owned() {
            test_config = attach_tool_choice(test_config, tool_choice);
        }

        let mut bd_request = self
            .client
            .converse_stream()
            .model_id(request.model.to_owned())
            .set_messages(Some(request.messages.iter().map(parse_messages).collect()))
            .set_system(request.system.map(parse_system))
            .inference_config(
                types::InferenceConfiguration::builder()
                    .set_max_tokens(Some(request.max_tokens as i32))
                    .set_stop_sequences(request.stop_sequences)
                    .set_temperature(request.temperature)
                    .set_top_p(request.top_p)
                    .build(),
            );

        if request.tools.is_some() {
            bd_request = bd_request.set_tool_config(Some(test_config.build().unwrap()));
        }

        let response = bd_request.send().await?;

        let model = request.model.clone();
        Ok(stream! {
            let request_id = response.request_id().unwrap().to_string();
            let mut s = response.stream;
            let mut event_message_delta: Option<EventMessageDelta> = None;
            let mut block_starts = HashSet::new();

            yield Ok(Event::MessageStart {
                message: MessageResponseStream {
                    kind: "message".into(),
                    message_response: MessageResponse {
                        id: request_id,
                        model,
                        role: "assistant".into(),
                        content: vec![],
                        stop_reason: None,
                        stop_sequence: None,
                        // FIXME: input_tokens should come from somewhere... Not sure where though
                        usage: Usage { input_tokens: Some(0), output_tokens: 0 },
                    }
                },
            });


            while let Ok(Some(event)) = s.recv().await {
                match event {
                    types::ConverseStreamOutput::ContentBlockDelta(block_delta) => {
                        let index = block_delta.content_block_index() as u64;
                        if !block_starts.contains(&index) {
                            block_starts.insert(index);
                            yield Ok(Event::ContentBlockStart {
                                index,
                                content_block: ContentPart::Text { text: "".into() },
                            });
                        }

                        yield Ok(Event::ContentBlockDelta {
                            index,
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
                        })},
                    types::ConverseStreamOutput::ContentBlockStart(block_start) => {
                        let index = block_start.content_block_index as u64;
                        if !block_starts.contains(&index) {
                            block_starts.insert(index);
                        }

                        yield Ok(Event::ContentBlockStart {
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
                    })},
                    types::ConverseStreamOutput::ContentBlockStop(block_stop) =>
                        yield Ok(Event::ContentBlockStop {
                            index: block_stop.content_block_index as u64,
                        }),
                    types::ConverseStreamOutput::MessageStart(_) => {
                        continue
                    },
                    types::ConverseStreamOutput::MessageStop(mess_stop) => {
                        if event_message_delta.is_some() {
                            yield Err(anyhow!("duplicated message delta"))
                        } else {
                            event_message_delta.replace(EventMessageDelta {
                                stop_reason: match mess_stop.stop_reason {
                                    types::StopReason::EndTurn => StopReason::EndTurn,
                                    types::StopReason::MaxTokens => StopReason::MaxTokens,
                                    types::StopReason::StopSequence => StopReason::StopSequence,
                                    types::StopReason::ToolUse => StopReason::ToolUse,
                                    _ => unreachable!(),
                                },
                                stop_sequence: None,
                            });
                            continue;
                        }
                    },
                    types::ConverseStreamOutput::Metadata(metadata) =>
                    if event_message_delta.is_none() {
                        yield Err(anyhow!("no message delta"))
                    } else {
                        let metadata = metadata.usage.unwrap();
                        yield Ok(Event::MessageDelta {
                            delta: event_message_delta.take().unwrap(),
                            usage: Usage {
                                input_tokens: Some(metadata.input_tokens as u32),
                                output_tokens: metadata.output_tokens as u32
                            }
                        });
                        yield Ok(Event::MessageStop)
                    }
                    e => {
                        dbg!(e);
                        unreachable!()
                    },
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
        let config = aws_config::defaults(BehaviorVersion::latest())
            .region("eu-central-1")
            .load()
            .await;
        let bedrock = AnthropicBedrock::new(&config);
        let request = CreateMessageRequest {
            model: Model::ClaudeThreeDotFiveSonnet.to_string(),
            messages: vec![Message {
                role: messages::Role::User,
                content: Content::Multi(vec![
                    "Hello".into(),
                    ContentPart::Image {
                        source: ImageSource {
                            kind: "base64".into(),
                            media_type: MediaType::ImageJpeg,
                            data: "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".into(),
                        },
                    },
                ]),
            }],
            system: Some("system".into()),
            max_tokens: 4096,
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
