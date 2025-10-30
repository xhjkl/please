use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use eyre::Result;
use std::sync::{Arc, RwLock};

use super::{AnyDisplay, Spinner};

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Phase {
    #[default]
    Answering,
    Thinking,
    // ToolCalling,
}

/// Render logs and answer as colorful streams and show a spinner while thinking.
#[derive(Default)]
struct RichDisplay {
    phase: RwLock<Phase>,
}

#[async_trait::async_trait]
impl AnyDisplay for RichDisplay {
    async fn start_spinning(&self) -> Spinner {
        Spinner::start()
    }

    async fn show_log(&self, line: &str) {
        let we_are_hub = std::env::var("PLEASE_BECOME_HUB").is_ok();
        if !we_are_hub {
            return;
        }

        let line = line.trim_end();

        let _ = crossterm::execute!(
            std::io::stderr(),
            SetForegroundColor(Color::DarkCyan),
            Print("| "),
            Print(line),
            ResetColor,
            Print("\n"),
        );
    }

    async fn start_thinking(&self) {
        *self.phase.write().unwrap() = Phase::Thinking;
    }

    async fn end_thinking(&self) {
        let phase = { *self.phase.read().unwrap() };
        if phase == Phase::Thinking {
            let _ = crossterm::execute!(std::io::stderr(), Print("\n"));
        }
        *self.phase.write().unwrap() = Phase::Answering;
    }

    async fn end_answer(&self) {
        let _ = crossterm::execute!(std::io::stdout(), Print("\n"));
    }

    async fn show_delta(&self, delta: &str) {
        let phase = { *self.phase.read().unwrap() };
        match phase {
            Phase::Thinking => {
                let _ = crossterm::execute!(
                    std::io::stderr(),
                    SetForegroundColor(Color::DarkYellow),
                    Print(delta),
                    ResetColor,
                );
            }
            Phase::Answering => {
                // `stdout` should be free from control sequences so that it could be piped.
                let _ = crossterm::execute!(std::io::stdout(), Print(delta));
            }
        }
    }

    async fn show_tool_call(&self, name: &str, args: &serde_json::Value) {
        let args = serde_json::to_string(args).unwrap_or_else(|_| args.to_string());
        let _ = crossterm::execute!(
            std::io::stderr(),
            SetForegroundColor(Color::DarkCyan),
            Print(name),
            Print(args),
            ResetColor,
            Print("\n"),
            Print("\n"),
        );
    }

    async fn confirm_run_command_execution(&self, _argv: &[String]) -> bool {
        // Assuming `argv` has already been presented to the user by `show_tool_call`.
        let _ = crossterm::execute!(std::io::stderr(), Print("Proceed? [y/N] "));
        yes_or_no()
    }

    async fn confirm_apply_patch_edits(&self, preview: &str) -> bool {
        let _ = crossterm::execute!(
            std::io::stderr(),
            SetForegroundColor(Color::DarkYellow),
            Print("\n"),
            Print(preview),
            Print("\nProceed? [y/N] "),
        );
        yes_or_no()
    }

    async fn show_onboarding(&self) {
        use crossterm::style::{Attribute, SetAttribute};

        fn ollama_available() -> bool {
            std::process::Command::new("ollama")
                .arg("--version")
                .output()
                .is_ok()
        }

        let _ = crossterm::execute!(
            std::io::stderr(),
            SetAttribute(Attribute::Bold),
            Print(
                "To use please, you first need to load the model. This only needs to be done once."
            ),
            Print("\n"),
        );

        if ollama_available() {
            let _ = crossterm::execute!(
                std::io::stderr(),
                SetAttribute(Attribute::Bold),
                Print(
                    "Since you have ollama installed, you can just run: `ollama pull gpt-oss:20b`, wait until completion, and then run `please` again."
                ),
                Print("\n"),
            );
        };

        let _ = crossterm::execute!(
            std::io::stderr(),
            SetAttribute(Attribute::Bold),
            Print(
                "If you want to download the weights manually, you can run `please load`. It will place the weights to `~/.please/weights`."
            ),
            Print("\n"),
        );

        let _ = crossterm::execute!(std::io::stderr(), SetAttribute(Attribute::Reset));
    }
}

fn yes_or_no() -> bool {
    let mut buffer = String::new();
    let stdin = std::io::stdin();
    let Ok(_read) = stdin.read_line(&mut buffer) else {
        return false;
    };
    let first_char = buffer.trim().chars().next().unwrap_or('n');
    first_char.eq_ignore_ascii_case(&'y')
}

/// Try to construct the rich display; callers should fall back to plain on error.
pub fn try_make_display() -> Result<Arc<dyn AnyDisplay>> {
    Ok(Arc::new(RichDisplay::default()))
}
