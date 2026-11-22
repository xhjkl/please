use eyre::Result;
use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;
use tokio::signal::unix::{SignalKind, signal};

use crate::cli::io;
use crate::cli::specials;
use crate::display;
use crate::history;
use crate::protocol::{HubBusy, Message};

use super::connect::obtain_control_stream;
use super::repl::interact_forever;
use super::turn::{TurnCancelled, is_cancelled, run_turn};

/// sysexits' temporary failure: the user is invited to retry later.
const EX_TEMPFAIL: u8 = 75;
const INTERRUPTED: u8 = 130;

enum Conversation {
    Interactive,
    OneShot(String),
}

/// Initialize the UI pipeline and spawn the renderer.
/// Returns channels the rest of the app can use to stream status and content.
fn start_display() -> Result<Arc<display::Display>> {
    let display = Arc::new(display::make_display());
    crate::logging::setup_tracing_display_logger(display.clone());
    Ok(display)
}

/// CLI entrypoint: decide between hub mode, REPL, or one-shot batch prompt.
/// Keeps top-level flow readable while deferring details to real implementations.
pub async fn run() -> Result<ExitCode> {
    // Start display; all user-visible output goes through it
    let display = start_display()?;

    // One-shot specials (help/version/load) should exit early before any UI/hub work.
    let did_handle_specials = specials::handle_specials_if_needed().await?;
    if did_handle_specials {
        return Ok(ExitCode::SUCCESS);
    }

    let stdout_is_tty = std::io::stdout().is_terminal();
    let stderr_is_tty = std::io::stderr().is_terminal();
    let stdin_is_tty = std::io::stdin().is_terminal();
    let stdout_redirection_path =
        (!stdout_is_tty).then(|| io::stdout_redirection_path().unwrap_or_default());
    let stdin_content = io::read_whole_stdin()?;
    let history = history::make_history(stdin_content, stdout_redirection_path);

    // Build prompt from positional CLI args; if none provided, leave empty to enable REPL.
    // Collect positional args into a single prompt. If none provided, drop into REPL.
    let prompt = {
        let mut args = std::env::args();
        let _ = args.next(); // binary name
        let collected: String = args.collect::<Vec<String>>().join(" ");
        collected
    };

    let conversation = if stdout_is_tty && stderr_is_tty && stdin_is_tty && prompt.is_empty() {
        Conversation::Interactive
    } else {
        Conversation::OneShot(prompt)
    };
    match converse(&display, history, conversation).await {
        Err(error) if error.downcast_ref::<HubBusy>().is_some() => {
            eprintln!("please: {error}");
            Ok(ExitCode::from(EX_TEMPFAIL))
        }
        Err(error) if is_cancelled(&error) => {
            eprintln!("\nplease: turn cancelled");
            Ok(ExitCode::from(INTERRUPTED))
        }
        Err(error) => Err(error),
        Ok(()) => Ok(ExitCode::SUCCESS),
    }
}

/// Connect to the hub, maybe starting a new hub process, and hold the conversation.
async fn converse(
    display: &display::Display,
    mut history: Vec<Message>,
    conversation: Conversation,
) -> Result<()> {
    let mut interrupt = signal(SignalKind::interrupt())?;
    let little_snake = display.start_spinning().await;
    let stream = tokio::select! {
        biased;
        _ = interrupt.recv() => Err(TurnCancelled.into()),
        stream = obtain_control_stream() => stream,
    };
    little_snake.stop().await;
    drop(interrupt);

    // If there are no weights, show the onboarding and exit.
    let stream = match stream {
        Err(error) if error.downcast_ref::<crate::hub::NoWeights>().is_some() => {
            display.show_onboarding().await;
            return Err(error);
        }
        Err(error) => return Err(error),
        Ok(stream) => stream,
    };
    let mut stream = Some(stream);

    match conversation {
        Conversation::Interactive => interact_forever(&mut stream, display, history).await?,
        Conversation::OneShot(prompt) => {
            history.push(Message::User(prompt));
            run_turn(&mut stream, display, &mut history).await?;
        }
    }

    Ok(())
}
