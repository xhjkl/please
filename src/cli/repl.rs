use eyre::{Result, eyre};
use std::sync::Arc;
use tokio::net::UnixStream;

use crate::display::Display;
use crate::protocol::Message;

use super::connect::obtain_control_stream;
use super::turn::run_turn;

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

        let mut turn_history = history.clone();
        turn_history.push(Message::User(line.to_string()));

        let answer = match run_turn(stream, display.clone(), turn_history.clone()).await {
            Ok(answer) => answer,
            Err(error) if super::turn::is_cancelled(&error) => {
                eprintln!();
                *stream = obtain_control_stream().await?;
                continue;
            }
            Err(error) => return Err(error),
        };
        eprintln!();

        history = turn_history;
        history.push(Message::Assistant(answer));
    }
    Ok(())
}
