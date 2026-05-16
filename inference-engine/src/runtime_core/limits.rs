#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeLimits {
    pub context_tokens: usize,
    pub max_new_tokens: usize,
    pub max_output_bytes: usize,
}
