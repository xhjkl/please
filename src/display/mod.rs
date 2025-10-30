use std::sync::Arc;

mod plain;
mod rich;
mod spinner;

pub use spinner::Spinner;

/// Object-safe display interface used by CLI components.
#[async_trait::async_trait]
pub trait AnyDisplay: Send + Sync {
    /// Return a guard that will stop the spinner when dropped.
    async fn start_spinning(&self) -> Spinner;

    /// Append a text line to the technical readout.
    async fn show_log(&self, line: &str);

    /// Switch display mode to presenting the reasoning process.
    async fn start_thinking(&self);

    /// Switch display mode to presenting the final answer.
    async fn end_thinking(&self);

    /// Switch display mode to taking input from the user.
    async fn end_answer(&self);

    /// Append a text piece to the currently active inference output.
    async fn show_delta(&self, s: &str);

    /// Show a pretty-formatted tool/function call with its JSON arguments.
    async fn show_tool_call(&self, name: &str, args: &serde_json::Value);

    /// Ask the user to confirm executing a command represented by argv.
    /// Returns true only if approved.
    async fn confirm_run_command_execution(&self, argv: &[String]) -> bool;

    /// Ask the user to confirm applying edits using a diff/content preview.
    async fn confirm_apply_patch_edits(&self, preview: &str) -> bool;

    /// Explain to the user how to get weights.
    async fn show_onboarding(&self);
}

/// Dynamically chosen display backend used by CLI components.
pub type Display = Arc<dyn AnyDisplay>;

/// Choose between rich and plain displays.
/// Uses plain when stdout is not a TTY, or if rich initialization fails.
/// Create a streaming display. Prefer rich TTY UI; fallback to plain printing.
pub fn make_display() -> Display {
    // Prefer rich when stderr is a TTY (so we can render UI there).
    // Fall back to plain when stderr is redirected.
    if atty::is(atty::Stream::Stderr) {
        match rich::try_make_display() {
            Ok(display) => {
                return display;
            }
            Err(_e) => {
                // Rich failed (no TTY or init error); continue with plain.
            }
        }
    }
    let display = plain::make_display();
    #[allow(clippy::let_and_return)]
    display
}
