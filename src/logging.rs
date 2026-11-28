use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};

use crate::display::Display;

/// Writer that forwards each formatted log line to the display's task.
struct DisplayWriter(UnboundedSender<String>);

impl std::io::Write for DisplayWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let line = String::from_utf8_lossy(buf).into_owned();
        let _ = self.0.send(line);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Route tracing logs into the display.
pub fn route_to_display(display: Arc<Display>) {
    let (sender, mut lines) = unbounded_channel::<String>();
    tokio::spawn(async move {
        while let Some(line) = lines.recv().await {
            display.show_log(&line);
        }
    });

    // Showing info by default; RUST_LOG can raise it, e.g. `please=debug` for session reuse.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .without_time()
        .with_file(false)
        .with_line_number(false)
        .with_level(false)
        .with_target(false)
        .with_writer(move || DisplayWriter(sender.clone()))
        .try_init();
}
