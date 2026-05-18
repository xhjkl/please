use eyre::Result;

mod load;

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
        return Ok(true);
    }

    if matches!(arg.as_str(), "version" | "--version" | "-V" | "-v") {
        println!("{}", env!("CARGO_PKG_VERSION"));
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
        load::run_load(which.as_deref()).await?;
        return Ok(true);
    }

    // Otherwise, not a special
    Ok(false)
}
