use super::common::{Param, ParamType};
use serde::Deserialize;
use serde_json::json;
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Deserialize)]
pub struct Args {
    /// Argument vector: first element is the program, followed by args
    argv: Vec<String>,
}

async fn read_stream<R: tokio::io::AsyncRead + Unpin>(
    mut reader: R,
    sink: Option<&UnboundedSender<String>>,
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let n = reader.read(&mut buf).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
        if let Some(tx) = sink {
            let _ = tx.send(String::from_utf8_lossy(&buf[..n]).to_string());
        }
    }
    out
}

/// Run a command and optionally stream stdout/stderr chunks to `sink`.
/// The returned JSON still includes full stdout/stderr for history use.
pub async fn call(args: Args, sink: Option<UnboundedSender<String>>) -> serde_json::Value {
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
        match stdout_pipe {
            Some(s) => read_stream(s, sink.as_ref()).await,
            None => Vec::new(),
        }
    };
    let read_err = async {
        match stderr_pipe {
            Some(s) => read_stream(s, sink.as_ref()).await,
            None => Vec::new(),
        }
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
