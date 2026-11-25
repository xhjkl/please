use std::future::Future;

use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use super::Display;

/// Live output for one tool invocation, driven by the same task as the turn.
pub struct ExecutionPane(UnboundedReceiver<String>);

impl ExecutionPane {
    /// Run `invocation` while showing what it streams, and return its result once all of that
    /// has been shown.
    pub async fn render<T>(mut self, invocation: impl Future<Output = T>) -> T {
        tokio::pin!(invocation);
        let result = tokio::select! {
            result = &mut invocation => result,
            () = self.drain() => invocation.await,
        };
        self.0.close();
        self.drain().await;
        result
    }

    async fn drain(&mut self) {
        while let Some(chunk) = self.0.recv().await {
            let _ = crossterm::execute!(
                std::io::stderr(),
                SetForegroundColor(Color::DarkYellow),
                Print(chunk),
                ResetColor,
            );
        }
    }
}

impl Drop for ExecutionPane {
    fn drop(&mut self) {
        let _ = crossterm::execute!(std::io::stderr(), Print("\n"));
    }
}

impl Display {
    /// Open a pane for one command's live output, with the sender its output goes to; `None`
    /// when stderr is not a terminal.
    pub fn execution_pane(&self) -> Option<(UnboundedSender<String>, ExecutionPane)> {
        if !self.stderr_is_tty {
            return None;
        }
        let (sender, receiver) = unbounded_channel();
        Some((sender, ExecutionPane(receiver)))
    }
}
