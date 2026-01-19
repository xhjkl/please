mod pane;
mod spinner;

pub use pane::ExecutionPane;
pub use spinner::Spinner;

use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use std::sync::RwLock;

#[derive(Clone, Copy)]
struct Caps {
    /// We can emit ANSI color/UI sequences to stderr.
    colorful: bool,
    /// We can safely prompt and wait for stdin input.
    can_prompt_user: bool,
    /// Show hub technical readout when available.
    should_show_readout: bool,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Phase {
    #[default]
    Answering,
    Thinking,
    Executing,
}

/// Display interface used by CLI components.
pub struct Display {
    caps: Caps,
    phase: RwLock<Phase>,
}

impl Display {
    /// Return a guard that will stop the spinner when dropped.
    pub async fn start_spinning(&self) -> Spinner {
        if self.caps.colorful {
            Spinner::start()
        } else {
            Spinner::start_empty()
        }
    }

    /// Append a text line to the technical readout.
    pub async fn show_log(&self, line: &str) {
        if !self.caps.should_show_readout {
            return;
        }
        let line = line.trim_end();
        if self.caps.colorful {
            let _ = crossterm::execute!(
                std::io::stderr(),
                SetForegroundColor(Color::DarkCyan),
                Print("| "),
                Print(line),
                ResetColor,
                Print("\n"),
            );
        } else {
            eprintln!("| {line}");
        }
    }

    /// Switch display mode to presenting the reasoning process.
    pub async fn start_thinking(&self) {
        *self.phase.write().unwrap() = Phase::Thinking;
    }

    /// Switch display mode to presenting the final answer.
    pub async fn end_thinking(&self) {
        let phase = { *self.phase.read().unwrap() };
        if self.caps.colorful && phase == Phase::Thinking {
            let _ = crossterm::execute!(std::io::stderr(), Print("\n"));
        }
        *self.phase.write().unwrap() = Phase::Answering;
    }

    /// Switch display mode to taking user input.
    pub async fn end_answer(&self) {
        let _ = crossterm::execute!(std::io::stdout(), Print("\n"));
    }

    /// Append a text piece to the currently active inference output.
    pub async fn show_delta(&self, s: &str) {
        let phase = { *self.phase.read().unwrap() };
        match phase {
            Phase::Thinking => {
                if self.caps.colorful {
                    let _ = crossterm::execute!(
                        std::io::stderr(),
                        SetForegroundColor(Color::DarkYellow),
                        Print(s),
                        ResetColor,
                    );
                }
            }
            Phase::Answering => {
                // `stdout` should be free from control sequences so it can be piped.
                let _ = crossterm::execute!(std::io::stdout(), Print(s));
            }
            Phase::Executing => {
                // should never happen
            }
        }
    }

    /// Show a pretty-formatted tool/function call with its JSON arguments.
    pub async fn show_tool_call(&self, name: &str, args: &serde_json::Value) {
        let args = serde_json::to_string(args).unwrap_or_else(|_| args.to_string());
        if self.caps.colorful {
            let _ = crossterm::execute!(
                std::io::stderr(),
                SetForegroundColor(Color::DarkCyan),
                Print(name),
                Print(args),
                ResetColor,
                Print("\n"),
                Print("\n"),
            );
        } else {
            eprintln!("call: {name} {args}");
        }
    }

    /// Show stdout/stderr from a tool invocation.
    pub async fn show_tool_output(&self, name: &str, stdout: &str, stderr: &str) {
        if stdout.is_empty() && stderr.is_empty() {
            return;
        }
        if self.caps.colorful {
            let _ = crossterm::execute!(
                std::io::stderr(),
                SetForegroundColor(Color::DarkCyan),
                Print(format!("{name} output:")),
                ResetColor,
                Print("\n"),
            );
        } else {
            eprintln!("{name} output:");
        }
        if !stdout.is_empty() {
            eprintln!("stdout:\n{stdout}");
        }
        if !stderr.is_empty() {
            eprintln!("stderr:\n{stderr}");
        }
        eprintln!();
    }

    /// Ask the user to confirm executing a command represented by argv.
    /// Returns true only if approved.
    pub async fn confirm_run_command_execution(&self, _argv: &[String]) -> bool {
        if !self.caps.can_prompt_user {
            eprintln!("rejecting run_command in non-interactive mode");
            return false;
        }
        let _ = crossterm::execute!(std::io::stderr(), Print("Proceed? [y/N] "));
        yes_or_no()
    }

    /// Ask the user to confirm applying edits using a diff/content preview.
    pub async fn confirm_apply_patch_edits(&self, preview: &str) -> bool {
        if !self.caps.can_prompt_user {
            eprintln!("rejecting apply_patch in non-interactive mode");
            return false;
        }
        if self.caps.colorful {
            let _ = crossterm::execute!(
                std::io::stderr(),
                SetForegroundColor(Color::DarkYellow),
                Print("\n"),
                Print(preview),
                Print("\nProceed? [y/N] "),
            );
        } else {
            eprintln!("\n{preview}\nProceed? [y/N] ");
        }
        yes_or_no()
    }

    /// Explain to the user how to get weights.
    pub async fn show_onboarding(&self) {
        if self.caps.colorful {
            use crossterm::style::{Attribute, SetAttribute};
            let _ = crossterm::execute!(
                std::io::stderr(),
                Print("\rTo get started with please, load the model once by running:"),
                Print("\n"),
                Print("\n"),
                SetAttribute(Attribute::Bold),
                Print("$ please load"),
                SetAttribute(Attribute::Reset),
                Print("\n"),
                Print("\n"),
                Print("Wait until it finishes; the weights will be stored in `~/.please/weights`."),
                Print("\n"),
                Print("\n"),
            );
        } else {
            eprintln!("To get started with please, run: please load");
            eprintln!("Wait until it finishes; weights go to ~/.please/weights.");
        }
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

/// Create a streaming display. Prefer colorful UI on TTY stderr; fallback to plain printing.
pub fn make_display() -> Display {
    let stderr_is_tty = atty::is(atty::Stream::Stderr);
    let stdin_is_tty = atty::is(atty::Stream::Stdin);

    // CLI is the only consumer today; readout is enabled for foreground hub runs.
    let hub_runs_in_foreground =
        ["run", "start"].contains(&std::env::args().nth(1).unwrap_or_default().as_str());

    let caps = Caps {
        colorful: stderr_is_tty,
        can_prompt_user: stdin_is_tty && stderr_is_tty,
        should_show_readout: hub_runs_in_foreground
            || std::env::var("PLEASE_LOG_EVERYTHING").is_ok(),
    };
    Display {
        caps,
        phase: RwLock::new(Phase::Answering),
    }
}
