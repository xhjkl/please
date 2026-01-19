use std::sync::Arc;

/// Writer that forwards writes to a logging channel.
struct DisplayWriter {
    sender: tokio::sync::mpsc::UnboundedSender<String>,
}

impl std::io::Write for DisplayWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let line = String::from_utf8_lossy(buf).to_string();
        let _ = self.sender.send(line);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

async fn pump(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    display: Arc<crate::display::Display>,
) {
    while let Some(line) = rx.recv().await {
        display.show_log(&line).await;
    }
}

/// Route tracing logs into the display renderer.
pub fn setup_tracing_display_logger(display: Arc<crate::display::Display>) {
    gg::send_logs_to_tracing(gg::LogOptions::default().with_logs_enabled(true));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(pump(rx, display));

    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_file(false)
        .with_line_number(false)
        .with_level(false)
        .with_target(false)
        .with_writer(move || DisplayWriter { sender: tx.clone() })
        .try_init();
}
