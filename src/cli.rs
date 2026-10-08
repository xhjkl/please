mod connect;
mod discovery;
mod io;
mod repl;
mod run;
mod specials;
mod turn;

pub use discovery::{choose_best_gpt, choose_best_qwen};
pub use run::run;
