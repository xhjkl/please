use super::{ContextNotice, PromptPlan};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerConfig {
    pub context_capacity: usize,
}

pub fn empty_plan(config: PlannerConfig) -> PromptPlan {
    PromptPlan {
        tokens: Vec::new(),
        pinned_prefix_len: 0,
        context_capacity: config.context_capacity,
        notices: vec![ContextNotice {
            message: "context planner is scaffolded only".to_string(),
        }],
    }
}
