mod approval;
mod pane;
mod spinner;
mod terminal;

pub use approval::Approval;
pub use spinner::Spinner;
pub use terminal::read_line as read_terminal_line;

use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use std::io::IsTerminal;

/// The probe's output: the answer on stdout, everything else on stderr, colored on a terminal.
pub struct Display {
    stderr_is_tty: bool,
    stdin_is_tty: bool,
    /// Whether hub logs are shown, as they are when the hub runs in the foreground.
    shows_hub_logs: bool,
}

/// A visible reasoning line that ends when the step moves on or is interrupted.
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
    /// Make the display; hub logs show when the hub runs in the foreground, or always with
    /// `PLEASE_LOG_EVERYTHING` set.
    pub fn new(hub_in_foreground: bool) -> Self {
        let stderr_is_tty = std::io::stderr().is_terminal();
        let stdin_is_tty = std::io::stdin().is_terminal();
        let log_everything = std::env::var("PLEASE_LOG_EVERYTHING").is_ok();
        Display {
            stderr_is_tty,
            stdin_is_tty,
            shows_hub_logs: hub_in_foreground || log_everything,
        }
    }

    /// Start a spinner on stderr, or an inert one when stderr is not a terminal.
    pub fn start_spinning(&self) -> Spinner {
        Spinner::start(self.stderr_is_tty)
    }

    /// Show a line of the hub's log, if those are shown.
    pub fn show_log(&self, line: &str) {
        if !self.shows_hub_logs {
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

    /// Show a tool call with its JSON arguments.
    pub fn show_tool_call(&self, name: &str, args: &serde_json::Value) {
        if self.stderr_is_tty {
            let _ = crossterm::execute!(
                std::io::stderr(),
                SetForegroundColor(Color::DarkCyan),
                Print(name),
                Print(args),
                ResetColor,
                Print("\n\n"),
            );
        } else {
            eprintln!("call: {name} {args}");
        }
    }

    /// Show what a command printed, labelled by its command ID when it has one.
    pub fn show_tool_output(
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
    pub fn show_onboarding(&self) {
        if self.stderr_is_tty {
            use crossterm::style::{Attribute, SetAttribute};
            let _ = crossterm::execute!(
                std::io::stderr(),
                Print("\rTo get started with please, load the model once by running:\n\n"),
                SetAttribute(Attribute::Bold),
                Print("$ please load"),
                SetAttribute(Attribute::Reset),
                Print("\n\n"),
                Print(
                    "Wait until it finishes; the weights will be stored in `~/.please/weights`.\n\n"
                ),
            );
        } else {
            eprintln!("To get started with please, run: please load");
            eprintln!("Wait until it finishes; weights go to ~/.please/weights.");
        }
    }
}
