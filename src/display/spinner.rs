//! Pseudographical progress indicator.

use crossterm::cursor;
use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::terminal::{Clear, ClearType};
use std::time::Duration;

const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

async fn spin() {
    let _ = crossterm::execute!(std::io::stderr(), cursor::Hide);
    for frame in FRAMES.iter().cycle() {
        let _ = crossterm::execute!(
            std::io::stderr(),
            Print("\r"),
            SetForegroundColor(Color::DarkGrey),
            Print(frame),
            ResetColor
        );
        tokio::time::sleep(Duration::from_millis(120)).await;
    }
}

/// A spinner drawing on stderr until stopped or dropped. Only `stop` waits for the last frame
/// to be drawn, so stop it before writing anything else to stderr.
#[must_use = "spinner keeps drawing until stopped"]
pub struct Spinner {
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Spinner {
    /// Start a spinner, one that draws nothing unless `visible`.
    pub fn start(visible: bool) -> Self {
        Spinner {
            task: visible.then(|| tokio::spawn(spin())),
        }
    }

    /// Stop drawing and clear the spinner's line.
    pub async fn stop(mut self) {
        let Some(task) = self.task.take() else {
            return;
        };
        task.abort();
        let _ = task.await;
        clear_spinner_line();
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
            clear_spinner_line();
        }
    }
}

fn clear_spinner_line() {
    let _ = crossterm::execute!(
        std::io::stderr(),
        Clear(ClearType::CurrentLine),
        Print("\r"),
        ResetColor,
        cursor::Show,
    );
}
