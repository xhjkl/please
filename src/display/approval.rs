use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use std::io;

use super::{Display, read_terminal_line};

/// The user's answer to a call that needs approval.
#[derive(Debug, PartialEq, Eq)]
pub enum Approval {
    Approved,
    /// Declined, for the reason the user typed.
    Denied(String),
    /// No answer could be asked for or read, for the reason given.
    Unavailable(String),
}

impl Display {
    /// Ask for approval on the terminal and capture the user's reason for declining.
    pub async fn confirm(&self, name: &str, preview: &str) -> Approval {
        if !(self.stdin_is_tty && self.stderr_is_tty) {
            let reason = "approval requires a terminal";
            eprintln!("please: {name}: {reason}");
            return Approval::Unavailable(reason.into());
        }

        if !preview.is_empty() {
            let _ = crossterm::execute!(
                io::stderr(),
                SetForegroundColor(Color::DarkYellow),
                Print(preview),
                ResetColor,
                Print("\n"),
            );
        }
        let _ = crossterm::execute!(
            io::stderr(),
            Print("Proceed? [Enter to approve, or type a reason to decline] "),
        );
        let line = read_terminal_line().await;
        match line {
            Ok(line) => parse_approval(&line),
            Err(error) => Approval::Unavailable(error.to_string()),
        }
    }
}

fn parse_approval(line: &str) -> Approval {
    let reason = line.trim();
    if !reason.is_empty() {
        return Approval::Denied(reason.into());
    }
    // Requiring a newline distinguishes a blank submission from EOF.
    if !line.ends_with('\n') {
        return Approval::Unavailable("no approval line was submitted".into());
    }
    Approval::Approved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_submission_approves_and_text_is_the_denial_reason() {
        assert_eq!(parse_approval("\n"), Approval::Approved);
        assert_eq!(parse_approval("  \n"), Approval::Approved);
        assert_eq!(parse_approval("y\n"), Approval::Denied("y".into()));
        assert_eq!(
            parse_approval("Use the local fixture instead.\n"),
            Approval::Denied("Use the local fixture instead.".into()),
        );
        assert!(matches!(parse_approval(""), Approval::Unavailable(_)));
        assert!(matches!(parse_approval("  "), Approval::Unavailable(_)));
    }
}
