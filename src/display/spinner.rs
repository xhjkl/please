//! Pseudographical progress indicator.

use crossterm::cursor;
use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::terminal::{Clear, ClearType};

async fn display_spinner() {
    use std::time::Duration;
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let mut index: usize = 0;

    let _ = crossterm::execute!(std::io::stderr(), cursor::Hide);
    loop {
        let frame = frames[index];
        let _ = crossterm::execute!(
            std::io::stderr(),
            Print("\r"),
            SetForegroundColor(Color::DarkGrey),
            Print(frame),
            ResetColor
        );
        index += 1;
        index %= frames.len();

        tokio::time::sleep(Duration::from_millis(120)).await;
    }
}

/// Guard to keep spinner active while in scope.
pub struct Spinner {
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Spinner {
    /// For compatibility, make a spinner that does nothing.
    pub(super) fn start_empty() -> Self {
        Spinner { task: None }
    }

    /// Immediately start a task that will show a spinner until dropped.
    pub(super) fn start() -> Self {
        Spinner {
            task: Some(tokio::spawn(display_spinner())),
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.task.take().map(|task| {
            tokio::spawn(async move {
                task.abort();
                let _ = task.await;
                let _ = crossterm::execute!(
                    std::io::stderr(),
                    Clear(ClearType::CurrentLine),
                    Print("\r"),
                    ResetColor,
                    cursor::Show,
                );
            })
        });
    }
}
