mod approval;
mod pane;
mod spinner;
mod terminal;

pub use approval::Approval;
pub use spinner::Spinner;
pub use terminal::read_line as read_terminal_line;

use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};

/// Display interface used by CLI components.
pub struct Display {
    stderr_is_tty: bool,
    stdin_is_tty: bool,
    /// Show hub technical readout when available.
    should_show_readout: bool,
}

/// A visible reasoning line that ends when the subturn moves on or is interrupted.
pub struct ThinkingLine(());

impl ThinkingLine {
    pub fn show(&self, text: &str) {
        let _ = crossterm::execute!(
            std::io::stderr(),
            SetForegroundColor(Color::DarkYellow),
            Print(text),
            ResetColor,
        );
    }
}

impl Drop for ThinkingLine {
    fn drop(&mut self) {
        let _ = crossterm::execute!(std::io::stderr(), Print("\n"));
    }
}

impl Display {
    /// Return a guard that will stop the spinner when dropped.
    pub async fn start_spinning(&self) -> Spinner {
        if self.stderr_is_tty {
            Spinner::start()
        } else {
            Spinner::start_empty()
        }
    }

    /// Append a text line to the technical readout.
    pub async fn show_log(&self, line: &str) {
        if !self.should_show_readout {
            return;
        }
        let line = line.trim_end();
        if self.stderr_is_tty {
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

    pub fn thinking_line(&self) -> Option<ThinkingLine> {
        if !self.stderr_is_tty {
            return None;
        }
        Some(ThinkingLine(()))
    }

    pub fn end_answer(&self) {
        let _ = crossterm::execute!(std::io::stdout(), Print("\n"));
    }

    /// Append answer text to stdout without adding terminal escape sequences.
    pub fn show_answer(&self, text: &str) {
        let _ = crossterm::execute!(std::io::stdout(), Print(text));
    }

    /// Show a pretty-formatted tool/function call with its JSON arguments.
    pub async fn show_tool_call(&self, name: &str, args: &serde_json::Value) {
        let args = serde_json::to_string(args).unwrap_or_else(|_| args.to_string());
        if self.stderr_is_tty {
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
    pub async fn show_tool_output(
        &self,
        name: &str,
        command_id: Option<u64>,
        stdout: &str,
        stderr: &str,
    ) {
        if stdout.is_empty() && stderr.is_empty() {
            return;
        }
        let label = command_id.map_or_else(|| name.to_string(), |id| format!("command {id}"));
        if self.stderr_is_tty {
            let _ = crossterm::execute!(
                std::io::stderr(),
                SetForegroundColor(Color::DarkCyan),
                Print(format!("{label} output:")),
                ResetColor,
                Print("\n"),
            );
        } else {
            eprintln!("{label} output:");
        }
        if !stdout.is_empty() {
            eprintln!("stdout:\n{stdout}");
        }
        if !stderr.is_empty() {
            eprintln!("stderr:\n{stderr}");
        }
        eprintln!();
    }

    /// Explain to the user how to get weights.
    pub async fn show_onboarding(&self) {
        if self.stderr_is_tty {
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

/// Create a streaming display. Prefer colorful UI on TTY stderr; fallback to plain printing.
pub fn make_display() -> Display {
    let stderr_is_tty = atty::is(atty::Stream::Stderr);
    let stdin_is_tty = atty::is(atty::Stream::Stdin);

    // CLI is the only consumer today; readout is enabled for foreground hub runs.
    let hub_runs_in_foreground =
        ["run", "start"].contains(&std::env::args().nth(1).unwrap_or_default().as_str());

    Display {
        stderr_is_tty,
        stdin_is_tty,
        should_show_readout: hub_runs_in_foreground
            || std::env::var("PLEASE_LOG_EVERYTHING").is_ok(),
    }
}
