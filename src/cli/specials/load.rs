use eyre::{Result, eyre};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

fn weights_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    std::path::Path::new(&home).join(".please").join("weights")
}

fn ensure_dir(path: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o700);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

fn pick_repo(which: Option<&str>) -> (&'static str, &'static str) {
    // Returns (repo, default_file)
    let key = which.map(|s| s.trim()).unwrap_or("20b");
    let key = key
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase();

    match key.as_str() {
        // Big model aliases
        "120b" | "120" | "big" => ("ggml-org/gpt-oss-120b-GGUF", "gpt-oss-120b-mxfp4.gguf"),
        // Default: small
        _ => ("ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-mxfp4.gguf"),
    }
}

fn build_url(which: Option<&str>) -> (String, String) {
    let (repo, file) = pick_repo(which);
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    (url, file.to_string())
}

fn build_client() -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        reqwest::header::HeaderValue::from_static("please/hf-gguf-fetch/0.1"),
    );
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .use_rustls_tls()
        .build()
        .map_err(|e| eyre!(e))?;
    Ok(client)
}

async fn download_with_resume(url: &str, target: &std::path::Path) -> Result<()> {
    let client = build_client()?;

    // Determine current size if a partially downloaded file already exists
    let mut start_from = 0u64;
    if let Ok(meta) = tokio::fs::metadata(&target).await {
        start_from = meta.len();
    }

    let mut req = client.get(url);
    if start_from > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={}-", start_from));
    }

    let resp = req.send().await.map_err(|e| eyre!(e))?;
    let status = resp.status();
    if !(status.is_success() || status == reqwest::StatusCode::PARTIAL_CONTENT) {
        return Err(eyre!("download failed: {}", status));
    }

    let total_remaining = resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let mut stream = resp.bytes_stream();

    // If server ignored Range and returned full content while we already had partial,
    // truncate before writing to avoid duplication; otherwise append.
    let is_partial = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if start_from > 0 && status.is_success() && !is_partial {
        let _ = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&target)
            .await
            .map_err(|e| eyre!(e))?;
        start_from = 0;
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&target)
        .await
        .map_err(|e| eyre!(e))?;

    let mut downloaded = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| eyre!(e))?;
        file.write_all(&chunk).await.map_err(|e| eyre!(e))?;
        downloaded += chunk.len() as u64;

        if let Some(rem) = total_remaining {
            let done = start_from + downloaded;
            let total = start_from + rem;
            let pct = (done as f64 / total as f64) * 100.0;
            let _ = crossterm::execute!(
                std::io::stderr(),
                crossterm::style::Print(format!("\rdownloading: {done}/{total} bytes ({pct:.1}%)"))
            );
        }
    }
    let _ = crossterm::execute!(std::io::stderr(), crossterm::style::Print("\n"));

    file.flush().await.map_err(|e| eyre!(e))?;
    drop(file);
    Ok(())
}

pub async fn run_load(which: Option<&str>) -> Result<()> {
    let (url, name) = build_url(which);
    let dir = weights_dir();
    ensure_dir(&dir)?;
    let target = dir.join(&name);

    if target.exists() {
        eprintln!(
            "please load: {name} already present at {}",
            target.display()
        );
        return Ok(());
    }

    eprintln!("please load: downloading {name} -> {}", target.display());
    download_with_resume(&url, &target).await?;
    eprintln!("please load: done");
    Ok(())
}
