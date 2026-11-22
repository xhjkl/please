//! Command-line entrypoint. Both the CLI and hub start here.
use eyre::Result;
use std::process::ExitCode;

mod cli;
mod display;
mod history;
mod hub;
mod logging;
mod protocol;
#[cfg(test)]
mod test_support;
mod tools;

#[tokio::main]
async fn main() -> Result<ExitCode> {
    cli::run().await
}
