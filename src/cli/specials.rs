use eyre::Result;
use std::path::{Path, PathBuf};

use super::connect::dial;
use crate::protocol::{Frame, read_frame, write_frame};

const HELP: &str = "\
please: a polite LLM for CLI

  $ git diff --cached | please summarize to a concise commit message
  $ please fix all clippy diagnostics

  $ please load       Download and verify Inkling 2 Mini
  $ please version    Show the version and available models
";

/// A command to `please` itself rather than a request to the model.
#[derive(Debug, PartialEq, Eq)]
pub enum Special<'a> {
    Help,
    Version,
    Load,
    /// Serve as the hub in the foreground.
    Hub,
    Bench {
        model: &'a str,
        cases: &'a str,
    },
    /// Run docker with these arguments and the hub's socket mounted.
    Docker(&'a [String]),
}

/// Find the special the arguments spell, if any. Most are a lone word, with or without a
/// leading slash, so a sentence that merely starts with one stays a request; `docker` and
/// `bench` take arguments.
pub fn parse(args: &[String]) -> Option<Special<'_>> {
    let (first, rest) = args.split_first()?;
    let first = first.trim();
    let name = first.strip_prefix('/').unwrap_or(first).to_lowercase();
    match (name.as_str(), rest) {
        ("docker", rest) => Some(Special::Docker(rest)),
        ("bench", [model, cases]) => Some(Special::Bench { model, cases }),
        ("help" | "--help" | "-h" | "-?", []) => Some(Special::Help),
        ("version" | "--version" | "-v", []) => Some(Special::Version),
        ("load" | "download", []) => Some(Special::Load),
        ("run" | "start", []) => Some(Special::Hub),
        _ => None,
    }
}

/// Run a special; the program is done afterwards.
pub async fn run(special: Special<'_>) -> Result<()> {
    match special {
        Special::Help => println!("{HELP}"),
        Special::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            print_checkpoints().await?;
        }
        Special::Load => crate::hub::load_weights().await?,
        Special::Hub => crate::hub::run().await?,
        Special::Bench { model, cases } => crate::hub::benchmark(model, cases).await?,
        Special::Docker(args) => {
            // Mounting the host socket where a container's root user looks for it.
            let socket = crate::hub::socket_path();
            if let Some(directory) = socket.parent() {
                let _ = std::fs::create_dir_all(directory);
            }
            let volume = format!("{}:/root/.please/socket", socket.display());
            let mut docker = std::process::Command::new("docker");
            docker.arg("-v").arg(volume).args(args);
            let status = docker.status()?;
            std::process::exit(status.code().unwrap_or(1));
        }
    }
    Ok(())
}

/// Ask a running hub, local or forwarded, which checkpoint it has loaded.
async fn running_checkpoint() -> Result<Option<PathBuf>> {
    let socket = crate::hub::socket_path();
    let connected = dial(&socket).await;
    let mut stream = match connected {
        Err(error) if error.is_absent() => return Ok(None),
        Err(error) => return Err(error.into()),
        Ok(stream) => stream,
    };
    write_frame(&mut stream, &Frame::CheckpointRequest).await?;
    let mut buffer = Vec::new();
    let response = read_frame::<Frame>(
        &mut stream,
        &mut buffer,
        Some(std::time::Duration::from_millis(250)),
        Some(std::time::Duration::from_secs(2)),
    )
    .await;
    let Ok(Frame::Checkpoint { path }) = response else {
        return Ok(None);
    };
    Ok(Some(path.into()))
}

/// Print the available checkpoints under a heading for each folder, relative to `~` if inside.
async fn print_checkpoints() -> Result<()> {
    let running = running_checkpoint().await?;
    let paths = crate::hub::checkpoint_paths(running)?;
    let home = std::env::var_os("HOME");
    let home = home
        .as_deref()
        .filter(|home| !home.is_empty())
        .map(Path::new);
    let mut directory = None;
    for path in &paths {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        if directory != Some(parent) {
            let relative = home.and_then(|home| parent.strip_prefix(home).ok());
            match relative {
                Some(path) if path.as_os_str().is_empty() => println!("\n~:"),
                Some(path) => println!("\n~/{}:", path.display()),
                None => println!("\n{}:", parent.display()),
            }
            directory = Some(parent);
        }
        let name = path.file_name().unwrap_or(path.as_os_str());
        println!("  {}", name.to_string_lossy());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(line: &str) -> Vec<String> {
        line.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn a_special_is_a_lone_word_with_or_without_a_slash() {
        assert_eq!(parse(&words("help")), Some(Special::Help));
        assert_eq!(parse(&words("/load")), Some(Special::Load));
        assert_eq!(parse(&words("-V")), Some(Special::Version));
        assert_eq!(parse(&words("/Start")), Some(Special::Hub));
        assert_eq!(parse(&words("help me rename these files")), None);
        assert_eq!(parse(&words("run the tests")), None);
        assert_eq!(parse(&words("load the csv into sqlite")), None);
        assert_eq!(parse(&["version bump".into()]), None);
        assert_eq!(parse(&words("/etc/hosts")), None);
        assert_eq!(parse(&words("")), None);
        let docker = words("docker run --rm bash");
        assert_eq!(parse(&docker), Some(Special::Docker(&docker[1..])));
        let bench = words("bench model.gguf cases.json");
        let (model, cases) = ("model.gguf", "cases.json");
        assert_eq!(parse(&bench), Some(Special::Bench { model, cases }));
    }
}
