pub use anthropic::messages;

pub enum Model {
    ClaudeThreeDotFiveSonnet,
    ClaudeThreeSonnet,
    ClaudeThreeOpus,
    ClaudeThreeHaiku,
}

impl ToString for Model {
    fn to_string(&self) -> String {
        match self {
            Model::ClaudeThreeDotFiveSonnet => "anthropic.claude-3-5-sonnet@20240620".to_string(),
            Model::ClaudeThreeSonnet => "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
            Model::ClaudeThreeOpus => "anthropic.claude-3-opus-20240229-v1:0".to_string(),
            Model::ClaudeThreeHaiku => "anthropic.claude-3-5-sonnet-20240620-v1:0".to_string(),
        }
    }
}

pub struct AnthropicBedrock;
