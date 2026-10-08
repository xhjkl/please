//! Command-line entrypoint. Both the CLI and hub start here.
use eyre::Result;

mod cli;
mod display;
mod harmony;
mod history;
mod hub;
mod inference;
mod intuition;
mod logging;
mod prompting;
mod protocol;
mod tools;

#[tokio::main]
async fn main() -> Result<()> {
    cli::run().await
}
