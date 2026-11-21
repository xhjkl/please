use eyre::Result;

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
            "  $ please load       Download the largest dense Qwen that fits\n  $ please version    Show the engine and weights\n"
        );
        return Ok(true);
    }

    if matches!(arg.as_str(), "version" | "--version" | "-V" | "-v") {
        println!("{}", env!("CARGO_PKG_VERSION"));
        print_model().await?;
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
        let which = args.next();
        if args.next().is_some() {
            eyre::bail!("usage: please load [qwen]");
        }
        crate::hub::load_weights(which.as_deref()).await?;
        return Ok(true);
    }

    // Otherwise, not a special
    Ok(false)
}

/// Report a connected hub's actual checkpoint; without one, say where a new hub would look,
/// without opening Metal or reading any checkpoint.
async fn print_model() -> Result<()> {
    use crate::protocol::{Frame, read_frame_from_stream, write_frame_to_stream};
    let socket = crate::hub::socket_path();
    let connected = super::connect::try_connect_to_hub(&socket).await;
    match connected {
        Err(
            super::connect::ConnectError::Missing { .. }
            | super::connect::ConnectError::NoListener { .. },
        ) => {}
        Err(error) => return Err(error.into()),
        Ok(mut stream) => {
            write_frame_to_stream(&mut stream, &Frame::ModelInfoRequest).await?;
            let mut buffer = Vec::new();
            let response = read_frame_from_stream::<Frame>(
                &mut stream,
                &mut buffer,
                Some(std::time::Duration::from_millis(250)),
                Some(std::time::Duration::from_secs(2)),
            )
            .await;
            match response {
                Ok(Frame::ModelInfo { engine, path }) => {
                    println!("Engine: {engine}\nWeights (running hub): {path}")
                }
                _ => println!(
                    "Weights: connected hub does not report its model; restart it with this version of please"
                ),
            }
            return Ok(());
        }
    }
    print_local_selection(&socket);
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn print_local_selection(_socket: &std::path::Path) {
    println!("Engine: Oco");
    if let Some(path) = std::env::var_os("PLEASE_MODEL") {
        println!(
            "Weights (PLEASE_MODEL): {}",
            std::path::Path::new(&path).display()
        );
        return;
    }
    let directory = crate::hub::weights_dir();
    let mut names: Vec<_> = std::fs::read_dir(&directory)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|name| name.ends_with(".gguf"))
        .collect();
    names.sort();
    match names.is_empty() {
        true => println!(
            "Weights: none in {}; run `please load`",
            directory.display()
        ),
        false => println!(
            "Weights in {}: {}\nA new hub loads the largest dense Qwen that fits, else gpt-oss.",
            directory.display(),
            names.join(", ")
        ),
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn print_local_selection(socket: &std::path::Path) {
    println!(
        "Engine: none in this build; no hub is listening at {}",
        socket.display()
    );
}
