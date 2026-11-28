use eyre::Result;
use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal::unix::{SignalKind, signal};

use crate::cli::io;
use crate::cli::specials::{self, Special};
use crate::display::Display;
use crate::protocol::{HubBusy, Message};

use super::commands::{self, Command};
use super::connect::connect_to_hub;
use super::repl::interact;
use super::turn::{Scene, TurnCancelled, restart, run_turn};

/// sysexits' temporary failure: the user is invited to retry later.
const EX_TEMPFAIL: u8 = 75;
/// The shell's status for a process ended by SIGINT.
const INTERRUPTED: u8 = 130;
/// How long a special waits for piped data that would make its name a request instead.
const SPECIAL_PATIENCE: Duration = Duration::from_millis(100);

/// What one run of `please` asks for.
enum Invocation<'a> {
    /// A command to `please` itself.
    Special(Special<'a>),
    /// Work with the model, as nearly every run is, in one of its modes.
    Ordinary(Mode),
}

/// How an ordinary run of `please` works with the model.
enum Mode {
    /// At the prompt: a request or a command per line, until the user leaves.
    Interactive,
    /// One command, given as the whole one-shot request.
    Command(Command),
    /// One request, opening with any data piped along with it.
    Request(Vec<Message>),
}

/// Read what the arguments, the special they spell, and piped data ask for; `terminal` says
/// all three standard streams are one. Piped data makes the words a request about it, so then
/// nothing is a special or a command.
fn invocation<'a>(
    args: &[String],
    special: Option<Special<'a>>,
    input: Option<String>,
    terminal: bool,
) -> Result<Invocation<'a>> {
    let prompt = args.join(" ");
    if let Some(input) = input {
        let request = vec![Message::Input(input), Message::User(prompt)];
        return Ok(Invocation::Ordinary(Mode::Request(request)));
    }
    if let Some(special) = special {
        return Ok(Invocation::Special(special));
    }
    let mode = if terminal && prompt.is_empty() {
        Mode::Interactive
    } else {
        match commands::parse_one_shot(&prompt) {
            Some(Err(error)) => eyre::bail!(error),
            Some(Ok(command)) => Mode::Command(command),
            None => Mode::Request(vec![Message::User(prompt)]),
        }
    };
    Ok(Invocation::Ordinary(mode))
}

/// Run a special, or work with the model: at the prompt, or one-shot.
pub async fn run() -> Result<ExitCode> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let special = specials::parse(&args);
    // Waiting only briefly before a special, so an idle pipe on stdin, as agents' and CI's
    // shells may leave, cannot hang it.
    let patience = special.as_ref().map(|_| SPECIAL_PATIENCE);
    let input = io::read_whole_stdin(patience)?;
    let stdout_is_tty = std::io::stdout().is_terminal();
    let stderr_is_tty = std::io::stderr().is_terminal();
    let stdin_is_tty = std::io::stdin().is_terminal();
    let terminal = stdout_is_tty && stderr_is_tty && stdin_is_tty;
    let invocation = invocation(&args, special, input, terminal)?;
    let hub_in_foreground = matches!(invocation, Invocation::Special(Special::Hub));
    // All user-visible output, logs included, goes through the display.
    let display = Arc::new(Display::new(hub_in_foreground));
    crate::logging::route_to_display(display.clone());

    let mode = match invocation {
        Invocation::Special(special) => {
            specials::run(special).await?;
            return Ok(ExitCode::SUCCESS);
        }
        Invocation::Ordinary(mode) => mode,
    };
    // Redirected stdout is worth telling the model about even when its path is unknown.
    let redirection = io::stdout_redirection_path();
    let redirection = (!stdout_is_tty).then(|| redirection.unwrap_or_default());
    let setting = crate::setting::current(redirection);
    let place = crate::place::here()?;
    let scene = Scene {
        place: &place,
        setting: &setting,
    };
    let outcome = run_ordinary(&display, scene, mode).await;
    match outcome {
        Err(error) if error.is::<HubBusy>() => {
            eprintln!("please: {error}");
            Ok(ExitCode::from(EX_TEMPFAIL))
        }
        Err(error) if error.is::<TurnCancelled>() => {
            eprintln!("\nplease: turn cancelled");
            Ok(ExitCode::from(INTERRUPTED))
        }
        Err(error) => Err(error),
        Ok(()) => Ok(ExitCode::SUCCESS),
    }
}

/// Run an ordinary invocation in its mode, connecting to the hub first and maybe starting a
/// new hub process.
async fn run_ordinary(display: &Display, scene: Scene<'_>, mode: Mode) -> Result<()> {
    let mut interrupt = signal(SignalKind::interrupt())?;
    let little_snake = display.start_spinning();
    let stream = tokio::select! {
        biased;
        _ = interrupt.recv() => Err(TurnCancelled.into()),
        stream = connect_to_hub() => stream,
    };
    little_snake.stop().await;
    drop(interrupt);

    let stream = match stream {
        Err(error) if error.is::<crate::hub::NoWeights>() => {
            display.show_onboarding();
            return Err(error);
        }
        Err(error) => return Err(error),
        Ok(stream) => stream,
    };
    let mut stream = Some(stream);

    match mode {
        Mode::Interactive => interact(&mut stream, display, scene).await?,
        Mode::Command(command) => restart(&mut stream, display, scene, command).await?,
        Mode::Request(request) => {
            run_turn(&mut stream, display, scene, request).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(line: &str) -> Vec<String> {
        line.split_whitespace().map(String::from).collect()
    }

    fn read<'a>(args: &'a [String], input: Option<&str>, terminal: bool) -> Result<Invocation<'a>> {
        invocation(
            args,
            specials::parse(args),
            input.map(String::from),
            terminal,
        )
    }

    #[test]
    fn piped_data_makes_any_invocation_a_request() -> Result<()> {
        let (help, compact) = (words("help"), words("compact"));
        let Invocation::Ordinary(Mode::Request(request)) = read(&help, Some("data"), false)? else {
            panic!("piped data with a special's name is a request");
        };
        assert_eq!(
            request,
            [Message::Input("data".into()), Message::User("help".into())]
        );
        assert!(matches!(
            read(&compact, Some("data"), false)?,
            Invocation::Ordinary(Mode::Request(_))
        ));

        assert!(matches!(
            read(&help, None, false)?,
            Invocation::Special(Special::Help)
        ));
        assert!(matches!(
            read(&compact, None, false)?,
            Invocation::Ordinary(Mode::Command(Command::Compact))
        ));
        assert!(matches!(
            read(&[], None, true)?,
            Invocation::Ordinary(Mode::Interactive)
        ));
        assert!(read(&words("/frobnicate"), None, false).is_err());
        Ok(())
    }
}
