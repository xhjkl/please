use eyre::{Result, eyre};
use std::sync::Arc;
use tokio::net::UnixStream;

use crate::display::Display;
use crate::protocol::Message;

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
        history.push(Message::User(line.to_string()));

        let answer = run_turn(stream, display.clone(), history.clone()).await?;
        eprintln!();

        history.push(Message::Assistant(answer));
    }
    Ok(())
}
