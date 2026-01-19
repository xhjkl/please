use std::sync::Arc;

use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task::JoinHandle;

use super::{Display, Phase};

pub struct ExecutionPane {
    display: Arc<Display>,
    sender: UnboundedSender<String>,
    task: Option<JoinHandle<()>>,
}

impl ExecutionPane {
    pub fn sender(&self) -> UnboundedSender<String> {
        self.sender.clone()
    }
}

impl Drop for ExecutionPane {
    fn drop(&mut self) {
        let _ = self.task.take().map(|task| {
            tokio::spawn(async move {
                task.abort();
                let _ = task.await;
            })
        });
        let _ = crossterm::execute!(std::io::stderr(), Print("\n"));
        *self.display.phase.write().unwrap() = Phase::Answering;
    }
}

impl Display {
    pub fn start_executing(self: &Arc<Self>) -> Option<ExecutionPane> {
        if !self.caps.colorful {
            return None;
        }
        *self.phase.write().unwrap() = Phase::Executing;

        let (sender, mut rx) = unbounded_channel::<String>();
        let display = Arc::clone(self);
        let display_for_task = Arc::clone(&display);
        let task = tokio::spawn(async move {
            while let Some(chunk) = rx.recv().await {
                display_for_task.append_execution_output(&chunk);
            }
        });

        Some(ExecutionPane {
            display,
            sender,
            task: Some(task),
        })
    }

    fn append_execution_output(&self, chunk: &str) {
        if chunk.is_empty() {
            return;
        }
        let _ = crossterm::execute!(
            std::io::stderr(),
            SetForegroundColor(Color::DarkYellow),
            Print(chunk),
            ResetColor
        );
    }
}
