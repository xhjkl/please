use eyre::{Result, eyre};
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};

use crate::display::{Display, read_terminal_line};
use crate::protocol::{HubBusy, Message};

use super::commands::{self, Command};
use super::turn::{Scene, is_cancelled, restart, run_turn};

/// Read requests and commands until an empty line or end of input; the hub carries the
/// conversation from one to the next.
pub async fn interact_forever(
    stream: &mut Option<UnixStream>,
    display: &Display,
    scene: Scene<'_>,
) -> Result<()> {
    use rustyline::error::{
        ReadlineError::{Eof, Interrupted, Signal},
        Signal::Interrupt,
    };

    let term = std::env::var("TERM").unwrap_or_default();
    // Rustyline uses a blocking cooked read on these terminals.
    let cooked = ["dumb", "cons25", "emacs"]
        .iter()
        .any(|name| term.eq_ignore_ascii_case(name));
    let mut editor = if cooked {
        None
    } else {
        Some(rustyline::DefaultEditor::new()?)
    };
    loop {
        let line = if let Some(editor) = editor.as_mut() {
            let input = editor.readline(">> ");
            match input {
                Err(Eof) => break,
                Err(Interrupted | Signal(Interrupt)) => continue,
                Err(error) => return Err(eyre!(error)),
                Ok(line) => line,
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

        let outcome = match commands::parse(line) {
            Some(Err(error)) => Err(eyre!(error)),
            Some(Ok(command)) => restart(stream, display, scene, command == Command::Compact).await,
            None => {
                let request = vec![Message::User(line.to_string())];
                run_turn(stream, display, scene, request).await.map(drop)
            }
        };
        match outcome {
            // Another probe took the hub while this one reconnected; the caller steps aside.
            Err(error) if error.downcast_ref::<HubBusy>().is_some() => return Err(error),
            // A cancelled or failed turn ends here, and the conversation goes on.
            Err(error) if is_cancelled(&error) => eprintln!("\nplease: turn cancelled"),
            Err(error) => eprintln!("please: {error:#}"),
            Ok(()) => {}
        }
        eprintln!();
    }
    Ok(())
}
