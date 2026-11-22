use eyre::Result;
use std::path::{Path, PathBuf};

/// Handle special one-shot CLI commands like `--help`, `--version`, or `load`.
/// Returns true if a special action was handled and the program should exit.
pub async fn handle_specials_if_needed() -> Result<bool> {
    let mut args = std::env::args();
    let _ = args.next(); // binary name

    let arg = args.next().unwrap_or_default();

    if matches!(arg.as_str(), "help" | "--help" | "-H" | "-h" | "-?") {
        println!(
            "{}",
            concat!(
                "please: a polite LLM for CLI\n\n",
                "  $ git diff --cached | please summarize to a concise commit message\n",
                "  $ please fix all clippy diagnostics\n"
            )
        );
        println!(
            "  $ please load       Download and verify Inkling 2 Mini\n  $ please version    Show the version and available models\n"
        );
        return Ok(true);
    }

    if matches!(arg.as_str(), "version" | "--version" | "-V" | "-v") {
        println!("{}", env!("CARGO_PKG_VERSION"));
        print_models().await?;
        return Ok(true);
    }

    if matches!(arg.as_str(), "bench") {
        let (Some(model), Some(cases), None) = (args.next(), args.next(), args.next()) else {
            eyre::bail!("usage: please bench MODEL.gguf CASES.json");
        };
        crate::hub::benchmark(&model, &cases).await?;
        return Ok(true);
    }

    if matches!(arg.as_str(), "docker") {
        // Wrap docker to bind-mount the host socket into the container at /root/.please/socket
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        let host_socket = std::path::Path::new(&home).join(".please").join("socket");
        let _ = std::fs::create_dir_all(std::path::Path::new(&home).join(".please"));
        let volume = format!("{}:/root/.please/socket", host_socket.display());

        let mut docker_args: Vec<String> = Vec::new();
        docker_args.push("-v".to_string());
        docker_args.push(volume);
        docker_args.extend(args);

        let status = std::process::Command::new("docker")
            .args(&docker_args)
            .status()
            .map_err(|e| eyre::eyre!(e))?;
        std::process::exit(status.code().unwrap_or(1));
    }

    if matches!(arg.as_str(), "run" | "start") {
        // Launch the hub in the foreground and exit when it stops.
        crate::hub::run().await?;
        return Ok(true);
    }

    if matches!(arg.as_str(), "load" | "download") {
        let extra = args.next();
        if extra.is_some() {
            eyre::bail!("usage: please load");
        }
        crate::hub::load_weights().await?;
        return Ok(true);
    }

    // Otherwise, not a special
    Ok(false)
}

/// Query the checkpoint already in use by a local or forwarded hub.
async fn running_model() -> Result<Option<PathBuf>> {
    use crate::protocol::{Frame, read_frame_from_stream, write_frame_to_stream};
    let socket = crate::hub::socket_path();
    let connected = super::connect::try_connect_to_hub(&socket).await;
    let mut stream = match connected {
        Err(
            super::connect::ConnectError::Missing { .. }
            | super::connect::ConnectError::NoListener { .. },
        ) => return Ok(None),
        Err(error) => return Err(error.into()),
        Ok(stream) => stream,
    };
    write_frame_to_stream(&mut stream, &Frame::ModelInfoRequest).await?;
    let mut buffer = Vec::new();
    let response = read_frame_from_stream::<Frame>(
        &mut stream,
        &mut buffer,
        Some(std::time::Duration::from_millis(250)),
        Some(std::time::Duration::from_secs(2)),
    )
    .await;
    let Ok(Frame::ModelInfo { path, .. }) = response else {
        return Ok(None);
    };
    Ok(Some(path.into()))
}

async fn print_models() -> Result<()> {
    let running = running_model().await?;
    let paths = crate::hub::model_paths(running)?;
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
