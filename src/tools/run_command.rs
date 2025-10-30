use super::common::{Param, ParamType};
use serde::Deserialize;
use serde_json::json;
use std::process::Stdio;
use tokio::io::AsyncReadExt;

#[derive(Deserialize)]
pub struct Args {
    /// Argument vector: first element is the program, followed by args
    argv: Vec<String>,
}

pub async fn call(args: Args) -> serde_json::Value {
    if args.argv.is_empty() {
        return json!({ "error": "argv must be non-empty" });
    }

    let mut cmd = tokio::process::Command::new(&args.argv[0]);
    if args.argv.len() > 1 {
        cmd.args(&args.argv[1..]);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return json!({ "error": e.to_string() }),
    };

    // Extract pipes before awaiting to avoid partial-move issues
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    // Read to completion (no truncation, no timeout)
    let wait_fut = child.wait();
    let read_out = async {
        let mut out = Vec::new();
        if let Some(mut s) = stdout_pipe {
            let _ = s.read_to_end(&mut out).await;
        }
        out
    };
    let read_err = async {
        let mut err = Vec::new();
        if let Some(mut s) = stderr_pipe {
            let _ = s.read_to_end(&mut err).await;
        }
        err
    };
    let (status_res, stdout_bytes, stderr_bytes) = tokio::join!(wait_fut, read_out, read_err);
    let status = match status_res {
        Ok(s) => s,
        Err(e) => return json!({ "error": e.to_string() }),
    };
    json!({
        "ok": true,
        "status": {
            "code": status.code(),
            "success": status.success(),
        },
        "stdout": String::from_utf8_lossy(&stdout_bytes).to_string(),
        "stderr": String::from_utf8_lossy(&stderr_bytes).to_string(),
    })
}

pub fn spec() -> (&'static str, &'static str, Vec<Param>) {
    (
        "run_command",
        "Run a command by argv: first element is program, rest are args",
        vec![Param {
            name: "argv",
            desc: "Argument vector: [program, ...args]",
            param_type: ParamType::String,
            required: true,
        }],
    )
}
