use eyre::Result;

mod load;

/// Handle special one-shot CLI commands like `--help`, `--version`, or `load`.
/// Returns true if a special action was handled and the program should exit.
pub async fn handle_specials_if_needed() -> Result<bool> {
    let mut args = std::env::args();
    let _ = args.next(); // binary name

    let arg = args.next().unwrap_or_default();

    if matches!(arg.as_str(), "--help" | "-H" | "-h" | "-?") {
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

    if matches!(arg.as_str(), "--version" | "-V" | "-v") {
        println!("{}", env!("CARGO_PKG_VERSION"));
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
