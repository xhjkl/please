use super::ToolContext;
use super::common::{Param, ParamKind, Spec, resolve_path_within_cwd};
use serde::Deserialize;
use serde_json::json;
use std::fs::File;
use std::io::{self, Read};

#[derive(Deserialize)]
pub struct Args {
    path: String,
    #[serde(default = "default_max_bytes")]
    max_bytes: usize,
}

fn default_max_bytes() -> usize {
    512 * 1024
}

pub async fn call(args: Args, _context: ToolContext<'_>) -> serde_json::Value {
    let text = read(&args.path, args.max_bytes);
    match text {
        Err(error) => json!({ "error": error.to_string() }),
        Ok(text) => json!(text),
    }
}

fn read(path: &str, max_bytes: usize) -> io::Result<String> {
    let path = resolve_path_within_cwd(path)?;
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(max_bytes as u64).read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

pub const SPEC: Spec = Spec {
    name: "read_file",
    description: "Read a file's content with a byte limit",
    params: &[
        Param {
            name: "path",
            description: "Absolute or relative path to file",
            kind: ParamKind::String,
            required: true,
        },
        Param {
            name: "max_bytes",
            description: "Maximum number of bytes to read; default 524288",
            kind: ParamKind::Number,
            required: false,
        },
    ],
};
