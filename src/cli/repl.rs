use eyre::{Result, eyre};
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};

use crate::display::{Display, read_terminal_line};
use crate::protocol::{HubBusy, Message};

use super::turn::run_turn;

/// Read requests until an empty line or end of input, carrying the whole conversation.
pub async fn interact_forever(
    stream: &mut Option<UnixStream>,
    display: &Display,
    history: Vec<Message>,
) -> Result<()> {
    use rustyline::error::{
        ReadlineError::{Eof, Interrupted, Signal},
        Signal::Interrupt,
    };

    let term = std::env::var("TERM").unwrap_or_default();
    // Rustyline uses a blocking cooked read on these terminals.
    let mut editor = if ["dumb", "cons25", "emacs"]
        .iter()
        .any(|name| term.eq_ignore_ascii_case(name))
    {
        None
    } else {
        Some(rustyline::DefaultEditor::new()?)
    };
    let mut history = history;
    loop {
        let line = if let Some(editor) = editor.as_mut() {
            let input = editor.readline(">> ");
            match input {
                Ok(line) => line,
                Err(Eof) => break,
                Err(Interrupted | Signal(Interrupt)) => continue,
                Err(e) => return Err(eyre!(e)),
            }
        } else {
            let mut interrupt = signal(SignalKind::interrupt())?;
            crossterm::execute!(std::io::stderr(), crossterm::style::Print(">> "))?;
            tokio::select! {
                biased;
                _ = interrupt.recv() => {
                    eprintln!();
                    continue;
                }
                input = read_terminal_line() => input?,
            }
        };
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        if let Some(editor) = editor.as_mut() {
            let _ = editor.add_history_entry(line);
        }

        // A finished turn keeps everything it appended, so the next prompt extends it.
        let before = history.len();
        history.push(Message::User(line.to_string()));
        match run_turn(stream, display, &mut history).await {
            Ok(_) => {}
            // Another probe took the hub while this one reconnected; the caller steps aside.
            Err(error) if error.downcast_ref::<HubBusy>().is_some() => return Err(error),
            // A cancelled or failed turn leaves no trace, and the conversation goes on.
            Err(error) => {
                history.truncate(before);
                if super::turn::is_cancelled(&error) {
                    eprintln!("\nplease: turn cancelled");
                } else {
                    eprintln!("please: {error:#}");
                }
                eprintln!();
                continue;
            }
        }
        eprintln!();
    }
    Ok(())
}
