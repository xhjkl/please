use super::common::{Param, ParamType, resolve_path_within_cwd};
use serde::Deserialize;
use std::io::Read;

#[derive(Deserialize)]
pub struct Args {
    path: String,
    #[serde(default = "default_max_bytes")]
    max_bytes: usize,
}

fn default_max_bytes() -> usize {
    512 * 1024
}

pub async fn call(
    args: Args,
    _sink: Option<tokio::sync::mpsc::UnboundedSender<String>>,
) -> serde_json::Value {
    let res = (|| -> Result<String, String> {
        let rel = resolve_path_within_cwd(&args.path).map_err(|e| e.to_string())?;
        let file = std::fs::File::open(rel).map_err(|e| e.to_string())?;
        let mut buf: Vec<u8> = Vec::with_capacity(std::cmp::min(args.max_bytes, 1024 * 1024));
        let mut limited = std::io::Read::take(file, args.max_bytes as u64);
        limited.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    })();

    match res {
        Ok(s) => serde_json::json!(s),
        Err(e) => serde_json::json!({ "error": e }),
    }
}

pub fn spec() -> (&'static str, &'static str, Vec<Param>) {
    (
        "read_file",
        "Read a file's content with a byte limit",
        vec![
            Param {
                name: "path",
                desc: "Absolute or relative path to file",
                param_type: ParamType::String,
                required: true,
            },
            Param {
                name: "max_bytes",
                desc: "Maximum number of bytes to read; default 524288",
                param_type: ParamType::Number,
                required: false,
            },
        ],
    )
}
