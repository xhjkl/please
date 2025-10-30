use super::{AnyDisplay, Spinner};

struct Plain;

#[async_trait::async_trait]
impl AnyDisplay for Plain {
    async fn start_spinning(&self) -> Spinner {
        Spinner::start_empty()
    }
    async fn show_log(&self, line: &str) {
        println!("log: {line:?}")
    }
    async fn start_thinking(&self) {
        println!("reasoning/start")
    }
    async fn end_thinking(&self) {
        println!("reasoning/done")
    }
    async fn end_answer(&self) {
        println!("answer/done")
    }
    async fn show_delta(&self, s: &str) {
        println!("delta: {s:?}")
    }

    async fn show_tool_call(&self, name: &str, args: &serde_json::Value) {
        let args = serde_json::to_string(args).unwrap_or_else(|_| args.to_string());
        println!("call: {name} {args}");
    }

    async fn confirm_run_command_execution(&self, argv: &[String]) -> bool {
        eprintln!(
            "rejecting run_command in plain/non-interactive mode: {:?}",
            argv
        );
        false
    }

    async fn confirm_apply_patch_edits(&self, _preview: &str) -> bool {
        eprintln!("rejecting apply_patch in plain/non-interactive mode");
        false
    }

    async fn show_onboarding(&self) {
        eprintln!("log: no weights found; pull weights and restart");
    }
}

/// Minimal stdout display: prints logs and answers linearly.
pub fn make_display() -> std::sync::Arc<dyn AnyDisplay> {
    std::sync::Arc::new(Plain)
}
