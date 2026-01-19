use eyre::Result;
use std::sync::Arc;

use crate::cli::io;
use crate::cli::specials;
use crate::display;
use crate::history;
use crate::protocol::Message;

use super::connect::obtain_control_stream;
use super::repl::interact_forever;
use super::turn::run_turn;

/// Initialize the UI pipeline and spawn the renderer.
/// Returns channels the rest of the app can use to stream status and content.
fn start_display() -> Result<Arc<display::Display>> {
    let display = Arc::new(display::make_display());
    crate::logging::setup_tracing_display_logger(display.clone());
    Ok(display)
}

/// CLI entrypoint: decide between hub mode, REPL, or one-shot batch prompt.
/// Keeps top-level flow readable while deferring details to real implementations.
pub async fn run() -> Result<()> {
    // Start display; all user-visible output goes through it
    let display = start_display()?;

    // One-shot specials (help/version/load) should exit early before any UI/hub work.
    let did_handle_specials = specials::handle_specials_if_needed().await?;
    if did_handle_specials {
        return Ok(());
    }

    let stdout_is_tty = atty::is(atty::Stream::Stdout);
    let stderr_is_tty = atty::is(atty::Stream::Stderr);
    let stdin_is_tty = atty::is(atty::Stream::Stdin);
    let stdout_redirection_path =
        (!stdout_is_tty).then(|| io::stdout_redirection_path().unwrap_or_default());
    let stdin_content = io::read_whole_stdin()?;
    let mut history = history::make_history(stdin_content, stdout_redirection_path);

    // Build prompt from positional CLI args; if none provided, leave empty to enable REPL.
    // Collect positional args into a single prompt. If none provided, drop into REPL.
    let prompt = {
        let mut args = std::env::args();
        let _ = args.next(); // binary name
        let collected: String = args.collect::<Vec<String>>().join(" ");
        collected
    };

    // Connect to the hub, maybe starting a new hub process if necessary.
    let little_snake = display.start_spinning().await;
    let stream = obtain_control_stream().await;
    drop(little_snake);

    // If there are no weights, show the onboarding and exit.
    let mut stream = match stream {
        Err(_) => {
            display.show_onboarding().await;
            return Ok(());
        }
        Ok(stream) => stream,
    };

    // Choose between interactive and batch mode.
    // Step into interactive mode only when both stdout and stderr are teletype devices and the user provided no prompt.
    if stdout_is_tty && stderr_is_tty && stdin_is_tty && prompt.is_empty() {
        interact_forever(&mut stream, display, history).await?
    } else {
        // One-shot: append the user turn to the initial history and infer once.
        history.push(Message::User(prompt.to_string()));
        run_turn(&mut stream, display, history).await?;
    }

    Ok(())
}
