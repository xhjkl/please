use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GptOssVariant {
    GptOss20b,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GptOssConfig {
    pub variant: GptOssVariant,
    pub local_attention_window: usize,
    pub kv_heads: usize,
    pub expert_top_k: usize,
}

impl GptOssConfig {
    pub fn gpt_oss_20b() -> Self {
        Self {
            variant: GptOssVariant::GptOss20b,
            local_attention_window: 128,
            kv_heads: 8,
            expert_top_k: 4,
        }
    }
}
