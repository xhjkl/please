use eyre::Result;
use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;
use tokio::signal::unix::{SignalKind, signal};

use crate::cli::io;
use crate::cli::specials::{self, Special};
use crate::display::Display;
use crate::protocol::{HubBusy, Message};

use super::commands::{self, Command};
use super::connect::obtain_control_stream;
use super::repl::interact_forever;
use super::turn::{Scene, TurnCancelled, is_cancelled, restart, run_turn};

/// sysexits' temporary failure: the user is invited to retry later.
const EX_TEMPFAIL: u8 = 75;
/// The shell's status for a process ended by SIGINT.
const INTERRUPTED: u8 = 130;

enum Conversation {
    Interactive,
    OneShot(String),
}

/// Run a special command, the REPL, or a one-shot request.
pub async fn run() -> Result<ExitCode> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let special = specials::parse(&args);
    // All user-visible output, logs included, goes through the display.
    let display = Arc::new(Display::new(special == Some(Special::Hub)));
    crate::logging::setup_tracing_display_logger(display.clone());

    if let Some(special) = special {
        specials::run(special).await?;
        return Ok(ExitCode::SUCCESS);
    }

    let stdout_is_tty = std::io::stdout().is_terminal();
    let stderr_is_tty = std::io::stderr().is_terminal();
    let stdin_is_tty = std::io::stdin().is_terminal();
    // Redirected stdout is worth telling the model about even when its path is unknown.
    let redirection = io::stdout_redirection_path();
    let redirection = (!stdout_is_tty).then(|| redirection.unwrap_or_default());
    // Piped data opens the request it came with, as the person's material.
    let input = io::read_whole_stdin()?;
    let input = input.map(Message::Input);
    let setting = crate::setting::current(redirection);
    let place = crate::place::here()?;
    let prompt = args.join(" ");

    let conversation = if stdout_is_tty && stderr_is_tty && stdin_is_tty && prompt.is_empty() {
        Conversation::Interactive
    } else {
        Conversation::OneShot(prompt)
    };
    let scene = Scene {
        place: &place,
        setting: &setting,
    };
    let conversed = converse(&display, scene, input, conversation).await;
    match conversed {
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
    display: &Display,
    scene: Scene<'_>,
    input: Option<Message>,
    conversation: Conversation,
) -> Result<()> {
    let mut interrupt = signal(SignalKind::interrupt())?;
    let little_snake = display.start_spinning();
    let stream = tokio::select! {
        biased;
        _ = interrupt.recv() => Err(TurnCancelled.into()),
        stream = obtain_control_stream() => stream,
    };
    little_snake.stop().await;
    drop(interrupt);

    let stream = match stream {
        Err(error) if error.downcast_ref::<crate::hub::NoWeights>().is_some() => {
            display.show_onboarding();
            return Err(error);
        }
        Err(error) => return Err(error),
        Ok(stream) => stream,
    };
    let mut stream = Some(stream);

    match conversation {
        Conversation::Interactive => interact_forever(&mut stream, display, scene).await?,
        Conversation::OneShot(prompt) => match commands::parse(&prompt) {
            Some(Err(error)) => eyre::bail!(error),
            Some(Ok(command)) => {
                restart(&mut stream, display, scene, command == Command::Compact).await?
            }
            None => {
                let request = input.into_iter().chain([Message::User(prompt)]).collect();
                run_turn(&mut stream, display, scene, request).await?;
            }
        },
    }

    Ok(())
}
