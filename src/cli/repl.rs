use eyre::{Result, eyre};
use std::sync::Arc;
use tokio::net::UnixStream;

use crate::display::Display;
use crate::protocol::{HubBusy, Message};

use super::connect::obtain_control_stream;
use super::turn::run_turn;

/// Read requests until an empty line or end of input, carrying the whole conversation.
pub async fn interact_forever(
    stream: &mut UnixStream,
    display: Arc<Display>,
    history: Vec<Message>,
) -> Result<()> {
    use rustyline::error::ReadlineError::{Eof, Interrupted};

    let mut rl = rustyline::DefaultEditor::new().map_err(|e| eyre!(e))?;
    let mut history = history;
    loop {
        let line = match rl.readline(">> ") {
            Ok(line) => line,
            Err(Eof) | Err(Interrupted) => break,
            Err(e) => return Err(eyre!(e)),
        };
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        rl.add_history_entry(line).ok();

        // A finished turn keeps everything it appended, so the next prompt extends it.
        let before = history.len();
        history.push(Message::User(line.to_string()));
        match run_turn(stream, display.clone(), &mut history).await {
            Ok(_) => {}
            // Another probe took the hub while this one reconnected; the caller steps aside.
            Err(error) if error.downcast_ref::<HubBusy>().is_some() => return Err(error),
            // A cancelled or failed turn leaves no trace, and the conversation goes on.
            Err(error) => {
                history.truncate(before);
                if !super::turn::is_cancelled(&error) {
                    eprintln!("please: {error:#}");
                }
                eprintln!();
                *stream = obtain_control_stream().await?;
                continue;
            }
        }
        eprintln!();
    }
    Ok(())
}
