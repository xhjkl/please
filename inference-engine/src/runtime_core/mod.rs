pub mod kv_cache;
pub mod limits;
pub mod planner;
pub mod sampler;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineRequest {
    pub prompt: PromptPlan,
    pub sampling: SamplingConfig,
    pub limits: GenerationLimits,
    pub fixture: Option<PromptFixture>,
}

impl EngineRequest {
    pub fn scaffold() -> Self {
        Self {
            prompt: PromptPlan::empty_scaffold(),
            sampling: SamplingConfig::default(),
            limits: GenerationLimits::default(),
            fixture: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptFixture {
    pub fixture_name: Option<String>,
    pub prompt_bytes: usize,
    pub prompt_token_count: usize,
    pub prompt_tokens: Vec<u32>,
    pub prompt_token_prefix: Vec<u32>,
    pub prompt_token_suffix: Vec<u32>,
    pub prefill_token_count: usize,
}

impl PromptFixture {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\nHarmony prompt fixture:\n");
        if let Some(name) = &self.fixture_name {
            out.push_str(&format!("- fixture: {name}\n"));
        }
        out.push_str(&format!("- rendered prompt bytes: {}\n", self.prompt_bytes));
        out.push_str(&format!("- prompt tokens: {}\n", self.prompt_token_count));
        out.push_str(&format!("- prefill tokens: {}\n", self.prefill_token_count));
        out.push_str(&format!("- token prefix: {:?}\n", self.prompt_token_prefix));
        out.push_str(&format!("- token suffix: {:?}\n", self.prompt_token_suffix));
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProbeReport {
    pub first_prompt_token: u32,
    pub embedding_values: usize,
    pub embedding_min: f32,
    pub embedding_max: f32,
    pub embedding_mean: f32,
    pub embedding_l2: f32,
    pub embedding_sample: Vec<f32>,
}

impl CpuProbeReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\ncpu direct SafeTensors probe:\n");
        out.push_str(&format!(
            "- first prompt token: {}\n",
            self.first_prompt_token
        ));
        out.push_str(&format!(
            "- embedding row values: {}\n",
            self.embedding_values
        ));
        out.push_str(&format!("- min: {:.7}\n", self.embedding_min));
        out.push_str(&format!("- max: {:.7}\n", self.embedding_max));
        out.push_str(&format!("- mean: {:.7}\n", self.embedding_mean));
        out.push_str(&format!("- l2: {:.7}\n", self.embedding_l2));
        out.push_str(&format!("- first 8 values: {:?}\n", self.embedding_sample));
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLayer0Report {
    pub token: u32,
    pub hidden_size: usize,
    pub q_values: usize,
    pub k_values: usize,
    pub v_values: usize,
    pub attention_values: usize,
    pub residual_values: usize,
    pub moe_values: usize,
    pub layer_output_values: usize,
    pub q_sample: Vec<f32>,
    pub k_sample: Vec<f32>,
    pub v_sample: Vec<f32>,
    pub attention_sample: Vec<f32>,
    pub residual_sample: Vec<f32>,
    pub moe_sample: Vec<f32>,
    pub layer_output_sample: Vec<f32>,
    pub top_experts: Vec<ExpertScore>,
}

impl CpuLayer0Report {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\ncpu layer-0 math probe:\n");
        out.push_str(&format!("- token: {}\n", self.token));
        out.push_str(&format!("- hidden values: {}\n", self.hidden_size));
        out.push_str(&format!("- q values: {}\n", self.q_values));
        out.push_str(&format!("- k values: {}\n", self.k_values));
        out.push_str(&format!("- v values: {}\n", self.v_values));
        out.push_str(&format!("- attention values: {}\n", self.attention_values));
        out.push_str(&format!("- residual values: {}\n", self.residual_values));
        out.push_str(&format!("- moe values: {}\n", self.moe_values));
        out.push_str(&format!(
            "- layer output values: {}\n",
            self.layer_output_values
        ));
        out.push_str(&format!("- q first 8: {:?}\n", self.q_sample));
        out.push_str(&format!("- k first 8: {:?}\n", self.k_sample));
        out.push_str(&format!("- v first 8: {:?}\n", self.v_sample));
        out.push_str(&format!(
            "- attention first 8: {:?}\n",
            self.attention_sample
        ));
        out.push_str(&format!("- residual first 8: {:?}\n", self.residual_sample));
        out.push_str(&format!("- moe first 8: {:?}\n", self.moe_sample));
        out.push_str(&format!(
            "- layer output first 8: {:?}\n",
            self.layer_output_sample
        ));
        out.push_str("- router top-4:\n");
        for expert in &self.top_experts {
            out.push_str(&format!(
                "  - expert {}: logit {:.7}, weight {:.7}\n",
                expert.index, expert.logit, expert.weight
            ));
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertScore {
    pub index: usize,
    pub logit: f32,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSingleTokenReport {
    pub token: u32,
    pub layers: usize,
    pub final_hidden_values: usize,
    pub final_hidden_sample: Vec<f32>,
    pub top_logits: Vec<LogitScore>,
}

impl CpuSingleTokenReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\ncpu single-token full-stack probe:\n");
        out.push_str(&format!("- token: {}\n", self.token));
        out.push_str(&format!("- layers: {}\n", self.layers));
        out.push_str(&format!(
            "- final hidden values: {}\n",
            self.final_hidden_values
        ));
        out.push_str(&format!(
            "- final hidden first 8: {:?}\n",
            self.final_hidden_sample
        ));
        out.push_str("- cpu top logits:\n");
        for logit in &self.top_logits {
            out.push_str(&format!("  - token {}: {:.7}\n", logit.token, logit.logit));
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuPromptPrefillReport {
    pub prompt_tokens: Vec<u32>,
    pub layers: usize,
    pub final_position: usize,
    pub final_hidden_values: usize,
    pub final_hidden_sample: Vec<f32>,
    pub top_logits: Vec<LogitScore>,
}

impl CpuPromptPrefillReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\ncpu prompt-prefill probe:\n");
        out.push_str(&format!("- prompt tokens: {:?}\n", self.prompt_tokens));
        out.push_str(&format!("- token count: {}\n", self.prompt_tokens.len()));
        out.push_str(&format!("- layers: {}\n", self.layers));
        out.push_str(&format!("- final position: {}\n", self.final_position));
        out.push_str(&format!(
            "- final hidden values: {}\n",
            self.final_hidden_values
        ));
        out.push_str(&format!(
            "- final hidden first 8: {:?}\n",
            self.final_hidden_sample
        ));
        out.push_str("- cpu top logits:\n");
        for logit in &self.top_logits {
            out.push_str(&format!("  - token {}: {:.7}\n", logit.token, logit.logit));
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitScore {
    pub token: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOracleReport {
    pub weights: String,
    pub tokens: Vec<u32>,
    pub layers: usize,
    pub embedding_final_first8: Vec<f32>,
    pub layer_checkpoints: Vec<LayerCheckpoint>,
    pub final_norm_first8: Vec<f32>,
    pub selected_logits: Vec<SelectedLogit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCheckpoint {
    pub layer: usize,
    pub final_l2: f32,
    pub final_mean: f32,
    pub final_first8: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedLogit {
    pub token: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalRmsNormProbeReport {
    pub token: u32,
    pub values: usize,
    pub max_abs_delta: f32,
    pub mean_abs_delta: f32,
    pub cpu_first8: Vec<f32>,
    pub metal_first8: Vec<f32>,
}

impl MetalRmsNormProbeReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\nmetal RMSNorm oracle probe:\n");
        out.push_str(&format!("- token: {}\n", self.token));
        out.push_str(&format!("- values: {}\n", self.values));
        out.push_str(&format!("- max_abs_delta: {:.9}\n", self.max_abs_delta));
        out.push_str(&format!("- mean_abs_delta: {:.9}\n", self.mean_abs_delta));
        out.push_str(&format!("- cpu first 8: {:?}\n", self.cpu_first8));
        out.push_str(&format!("- metal first 8: {:?}\n", self.metal_first8));
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalMatvecProbeReport {
    pub name: String,
    pub token: u32,
    pub rows: usize,
    pub cols: usize,
    pub max_abs_delta: f32,
    pub mean_abs_delta: f32,
    pub cpu_first8: Vec<f32>,
    pub metal_first8: Vec<f32>,
}

impl MetalMatvecProbeReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\nmetal BF16 matvec oracle probe:\n");
        out.push_str(&format!("- name: {}\n", self.name));
        out.push_str(&format!("- token: {}\n", self.token));
        out.push_str(&format!("- rows: {}\n", self.rows));
        out.push_str(&format!("- cols: {}\n", self.cols));
        out.push_str(&format!("- max_abs_delta: {:.9}\n", self.max_abs_delta));
        out.push_str(&format!("- mean_abs_delta: {:.9}\n", self.mean_abs_delta));
        out.push_str(&format!("- cpu first 8: {:?}\n", self.cpu_first8));
        out.push_str(&format!("- metal first 8: {:?}\n", self.metal_first8));
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalVectorProbeReport {
    pub name: String,
    pub token: u32,
    pub position: usize,
    pub values: usize,
    pub max_abs_delta: f32,
    pub mean_abs_delta: f32,
    pub cpu_first8: Vec<f32>,
    pub metal_first8: Vec<f32>,
}

impl MetalVectorProbeReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\nmetal vector oracle probe:\n");
        out.push_str(&format!("- name: {}\n", self.name));
        out.push_str(&format!("- token: {}\n", self.token));
        out.push_str(&format!("- position: {}\n", self.position));
        out.push_str(&format!("- values: {}\n", self.values));
        out.push_str(&format!("- max_abs_delta: {:.9}\n", self.max_abs_delta));
        out.push_str(&format!("- mean_abs_delta: {:.9}\n", self.mean_abs_delta));
        out.push_str(&format!("- cpu first 8: {:?}\n", self.cpu_first8));
        out.push_str(&format!("- metal first 8: {:?}\n", self.metal_first8));
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalTopKProbeReport {
    pub name: String,
    pub token: u32,
    pub indices_match: bool,
    pub max_logit_delta: f32,
    pub max_weight_delta: f32,
    pub cpu: Vec<ExpertScore>,
    pub metal: Vec<ExpertScore>,
}

impl MetalTopKProbeReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\nmetal top-k oracle probe:\n");
        out.push_str(&format!("- name: {}\n", self.name));
        out.push_str(&format!("- token: {}\n", self.token));
        out.push_str(&format!("- indices_match: {}\n", self.indices_match));
        out.push_str(&format!("- max_logit_delta: {:.9}\n", self.max_logit_delta));
        out.push_str(&format!(
            "- max_weight_delta: {:.9}\n",
            self.max_weight_delta
        ));
        out.push_str("- cpu top experts:\n");
        for expert in &self.cpu {
            out.push_str(&format!(
                "  - expert {}: logit {:.7}, weight {:.7}\n",
                expert.index, expert.logit, expert.weight
            ));
        }
        out.push_str("- metal top experts:\n");
        for expert in &self.metal {
            out.push_str(&format!(
                "  - expert {}: logit {:.7}, weight {:.7}\n",
                expert.index, expert.logit, expert.weight
            ));
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalSelectedLogitsProbeReport {
    pub name: String,
    pub token: u32,
    pub layers: usize,
    pub max_abs_delta: f32,
    pub mean_abs_delta: f32,
    pub cpu: Vec<SelectedLogit>,
    pub metal: Vec<SelectedLogit>,
}

impl MetalSelectedLogitsProbeReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\nmetal selected-logits oracle probe:\n");
        out.push_str(&format!("- name: {}\n", self.name));
        out.push_str(&format!("- token: {}\n", self.token));
        out.push_str(&format!("- layers: {}\n", self.layers));
        out.push_str(&format!("- max_abs_delta: {:.9}\n", self.max_abs_delta));
        out.push_str(&format!("- mean_abs_delta: {:.9}\n", self.mean_abs_delta));
        out.push_str("- cpu logits:\n");
        for logit in &self.cpu {
            out.push_str(&format!("  - token {}: {:.7}\n", logit.token, logit.logit));
        }
        out.push_str("- metal logits:\n");
        for logit in &self.metal {
            out.push_str(&format!("  - token {}: {:.7}\n", logit.token, logit.logit));
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreedyTokenReport {
    pub token: u32,
    pub logit: f32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreedyTextProbeReport {
    pub name: String,
    pub position: usize,
    pub layers: usize,
    pub scorer: String,
    pub token_match: bool,
    pub logit_delta: f32,
    pub cpu: GreedyTokenReport,
    pub metal: GreedyTokenReport,
}

impl GreedyTextProbeReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\ngreedy text oracle probe:\n");
        out.push_str(&format!("- name: {}\n", self.name));
        out.push_str(&format!("- position: {}\n", self.position));
        out.push_str(&format!("- layers: {}\n", self.layers));
        out.push_str(&format!("- scorer: {}\n", self.scorer));
        out.push_str(&format!("- token_match: {}\n", self.token_match));
        out.push_str(&format!("- logit_delta: {:.9}\n", self.logit_delta));
        out.push_str(&format!(
            "- cpu: token {}, logit {:.7}, text {:?}\n",
            self.cpu.token, self.cpu.logit, self.cpu.text
        ));
        out.push_str(&format!(
            "- metal: token {}, logit {:.7}, text {:?}\n",
            self.metal.token, self.metal.logit, self.metal.text
        ));
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmHeadTopKProbeReport {
    pub name: String,
    pub position: usize,
    pub layers: usize,
    pub k: usize,
    pub tokens_match: bool,
    pub max_abs_delta: f32,
    pub cpu: Vec<GreedyTokenReport>,
    pub metal: Vec<GreedyTokenReport>,
}

impl LmHeadTopKProbeReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\nmetal lm_head top-k oracle probe:\n");
        out.push_str(&format!("- name: {}\n", self.name));
        out.push_str(&format!("- position: {}\n", self.position));
        out.push_str(&format!("- layers: {}\n", self.layers));
        out.push_str(&format!("- k: {}\n", self.k));
        out.push_str("- scorer: Metal BF16 lm_head logits + Metal top-k\n");
        out.push_str(&format!("- tokens_match: {}\n", self.tokens_match));
        out.push_str(&format!("- max_abs_delta: {:.9}\n", self.max_abs_delta));
        out.push_str("- cpu top-k:\n");
        for token in &self.cpu {
            out.push_str(&format!(
                "  - token {}: logit {:.7}, text {:?}\n",
                token.token, token.logit, token.text
            ));
        }
        out.push_str("- metal top-k:\n");
        for token in &self.metal {
            out.push_str(&format!(
                "  - token {}: logit {:.7}, text {:?}\n",
                token.token, token.logit, token.text
            ));
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptPlan {
    pub tokens: Vec<u32>,
    pub pinned_prefix_len: usize,
    pub context_capacity: usize,
    pub notices: Vec<ContextNotice>,
}

impl PromptPlan {
    pub fn empty_scaffold() -> Self {
        Self {
            tokens: Vec::new(),
            pinned_prefix_len: 0,
            context_capacity: 0,
            notices: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreedyDecodeProbeReport {
    pub name: String,
    pub backend: String,
    pub scorer: String,
    pub layers: usize,
    pub prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub stop_reason: StopReason,
    pub generated: Vec<GreedyTokenReport>,
    pub text: String,
}

impl GreedyDecodeProbeReport {
    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("\n{} greedy decode probe:\n", self.backend));
        out.push_str(&format!("- name: {}\n", self.name));
        out.push_str(&format!("- layers: {}\n", self.layers));
        out.push_str(&format!("- prompt tokens: {}\n", self.prompt_tokens));
        out.push_str(&format!("- max new tokens: {}\n", self.max_new_tokens));
        out.push_str(&format!("- scorer: {}\n", self.scorer));
        out.push_str(&format!("- stop_reason: {:?}\n", self.stop_reason));
        out.push_str(&format!("- text: {:?}\n", self.text));
        out.push_str("- generated:\n");
        for token in &self.generated {
            out.push_str(&format!(
                "  - token {}: logit {:.7}, text {:?}\n",
                token.token, token.logit, token.text
            ));
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextNotice {
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationLimits {
    pub max_new_tokens: usize,
    pub max_output_bytes: usize,
}

impl Default for GenerationLimits {
    fn default() -> Self {
        Self {
            max_new_tokens: 4096,
            max_output_bytes: 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationEvent {
    Token(u32),
    Text(String),
    Notice(RuntimeNotice),
    Stop(StopReason),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeNotice {
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    EndOfGeneration,
    MaxGeneratedTokens,
    OutputByteLimit,
    ContextExhausted,
    Cancelled,
    NotImplemented,
}
