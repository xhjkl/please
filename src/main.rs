//! Command-line entrypoint. Both the Hub and the Probe start here.
use eyre::Result;

pub mod cli;
pub mod display;
pub mod harmony;
pub mod history;
pub mod hub;
pub mod inference;
pub mod logging;
pub mod prompting;
pub mod protocol;
pub mod tools;

#[tokio::main]
async fn main() -> Result<()> {
    cli::run().await
}
