use eyre::{Result, eyre};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

/// Return the local directory where model weight files are stored.
fn weights_directory() -> std::path::PathBuf {
    let home_directory = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    std::path::Path::new(&home_directory)
        .join(".please")
        .join("weights")
}

/// Ensure the given directory exists with secure permissions (0700 on Unix).
fn ensure_directory(path: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(path)?.permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

/// Pick the appropriate repository and default file name based on a user-friendly alias.
/// Returns `(repository, default_file_name)`.
fn pick_repository(which: Option<&str>) -> (&'static str, &'static str) {
    // Returns (repository, default_file)
    let key = which.map(|s| s.trim()).unwrap_or("20b");
    let key = key
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase();

    match key.as_str() {
        // If the larger option is explicitly requested, use it
        "120b" | "120" | "big" | "large" => {
            ("ggml-org/gpt-oss-120b-GGUF", "gpt-oss-120b-mxfp4.gguf")
        }
        // Otherwise, default to small
        _ => ("ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-mxfp4.gguf"),
    }
}

/// Build the direct download URL and the file name for the chosen model.
/// Returns `(url, file_name)`.
fn build_url(which: Option<&str>) -> (String, String) {
    let (repository, file_name) = pick_repository(which);
    let url = format!("https://huggingface.co/{repository}/resolve/main/{file_name}");
    (url, file_name.to_string())
}

/// Build a configured HTTP client with a descriptive User-Agent.
fn build_http_client() -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        reqwest::header::HeaderValue::from_static(concat!("please/", env!("CARGO_PKG_VERSION"))),
    );
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .use_rustls_tls()
        .build()
        .map_err(|e| eyre!(e))?;
    Ok(client)
}

/// Parse a Content-Range header string.
/// Returns (start, end, total) where any missing component is None.
fn parse_content_range(header: &str) -> (Option<u64>, Option<u64>, Option<u64>) {
    // Accepts "bytes start-end/total" or "bytes */total"
    let header = header.trim();
    let mut header_parts = header.split_whitespace();
    let unit_token = header_parts.next();
    let range_value = header_parts.next();
    if unit_token != Some("bytes") || range_value.is_none() {
        return (None, None, None);
    }
    let range_value = range_value.unwrap();
    let mut range_and_total = range_value.split('/');
    let range_part = range_and_total.next().unwrap_or("");
    let total_part = range_and_total.next().unwrap_or("");
    let total_size = if total_part == "*" {
        None
    } else {
        total_part.parse::<u64>().ok()
    };
    if range_part == "*" {
        return (None, None, total_size);
    }
    let mut start_end_split = range_part.split('-');
    let start = start_end_split.next().and_then(|s| s.parse::<u64>().ok());
    let end = start_end_split.next().and_then(|s| s.parse::<u64>().ok());
    (start, end, total_size)
}

/// Truncate the file at the given path to the specified length.
async fn truncate_to(path: &std::path::Path, len: u64) -> Result<()> {
    let file_handle = tokio::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .await
        .map_err(|e| eyre!(e))?;
    file_handle.set_len(len).await.map_err(|e| eyre!(e))?;
    Ok(())
}

/// Download a remote file to `target_path`, resuming from a local partial file when possible.
/// Robustly handles servers that ignore ranges or respond with 416, and verifies final size when known.
async fn download_with_resume(url: &str, target_path: &std::path::Path) -> Result<()> {
    let client = build_http_client()?;

    // Determine current size if a partially downloaded file already exists
    let mut start_offset = 0u64;
    if let Ok(meta) = tokio::fs::metadata(&target_path).await {
        start_offset = meta.len();
    }

    // Try a HEAD to quickly determine total size (optimization and equality check)
    let mut known_total_size: Option<u64> = None;
    if let Ok(head) = client.head(url).send().await
        && head.status().is_success()
    {
        known_total_size = head
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        if let Some(total) = known_total_size {
            if start_offset == total {
                eprintln!("please load: already present at {}", target_path.display());
                return Ok(());
            }
            if start_offset > total {
                // Local file longer than remote: suspicious; restart full download
                eprintln!("please load: local larger than remote; restarting full download");
                start_offset = 0;
            }
        }
    }

    // Build initial GET (attempt ranged if we have partial local)
    let mut request = client.get(url);
    if start_offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={}-", start_offset));
    }

    let mut response = request.send().await.map_err(|e| eyre!(e))?;
    let mut status = response.status();

    // Handle 416 (Range Not Satisfiable)
    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let content_range_header = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|h| h.to_str().ok());
        if let Some(content_range_header) = content_range_header {
            let (_s, _e, total) = parse_content_range(content_range_header);
            if let Some(total) = total {
                if start_offset == total {
                    eprintln!("please load: already present at {}", target_path.display());
                    return Ok(());
                }
                if start_offset > total {
                    // Local file longer than remote: suspicious; restart full download
                    eprintln!("please load: local larger than remote; restarting full download");
                }
            }
        }
        // Cannot satisfy range; restart full
        start_offset = 0;
        response = client.get(url).send().await.map_err(|e| eyre!(e))?;
        status = response.status();
    }

    if !(status.is_success() || status == reqwest::StatusCode::PARTIAL_CONTENT) {
        return Err(eyre!("download failed: {}", status));
    }

    // If server ignored Range and returned full content while we already had partial,
    // truncate before writing to avoid duplication; otherwise append.
    let is_partial_response = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if start_offset > 0 {
        if is_partial_response {
            // Validate Content-Range alignment with local size
            let content_range_header = response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|h| h.to_str().ok());
            if let Some(content_range_header) = content_range_header {
                let (start, _end, total) = parse_content_range(content_range_header);
                if let Some(total) = total {
                    known_total_size = Some(total);
                }
                if start != Some(start_offset) {
                    // Mismatched range; restart full
                    start_offset = 0;
                    response = client.get(url).send().await.map_err(|e| eyre!(e))?;
                    status = response.status();
                }
            } else {
                // Partial without Content-Range, restart full
                start_offset = 0;
                response = client.get(url).send().await.map_err(|e| eyre!(e))?;
                status = response.status();
            }
        } else {
            // Got 200 OK ignoring Range -> restart from scratch using this response
            truncate_to(target_path, 0).await?;
            start_offset = 0;
        }
    }

    // Determine total if known from headers (helpful for progress)
    if known_total_size.is_none() {
        // For 200 OK, CONTENT_LENGTH is full total; for 206, it's remaining
        let content_length = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        if status == reqwest::StatusCode::PARTIAL_CONTENT {
            if let Some(remaining) = content_length {
                known_total_size = Some(start_offset + remaining);
            }
        } else {
            known_total_size = content_length;
        }
    }

    // Open file in the appropriate mode
    let mut file_handle = if start_offset == 0 {
        tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&target_path)
            .await
            .map_err(|e| eyre!(e))?
    } else {
        tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target_path)
            .await
            .map_err(|e| eyre!(e))?
    };

    let mut stream = response.bytes_stream();
    let mut bytes_downloaded = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| eyre!(e))?;
        file_handle.write_all(&chunk).await.map_err(|e| eyre!(e))?;
        bytes_downloaded += chunk.len() as u64;

        if let Some(total) = known_total_size {
            let bytes_done = start_offset + bytes_downloaded;
            let pct = (bytes_done as f64 / total as f64) * 100.0;
            let _ = crossterm::execute!(
                std::io::stderr(),
                crossterm::style::Print(format!(
                    "\rdownloading: {bytes_done}/{total} bytes ({pct:.1}%)"
                ))
            );
        }
    }
    let _ = crossterm::execute!(std::io::stderr(), crossterm::style::Print("\n"));

    file_handle.flush().await.map_err(|e| eyre!(e))?;
    drop(file_handle);

    // Final verification if we know total size
    if let Some(total) = known_total_size
        && let Ok(meta) = tokio::fs::metadata(&target_path).await
    {
        let final_size = meta.len();
        if final_size != total {
            return Err(eyre!(
                "download size mismatch: expected {}, got {}",
                total,
                final_size
            ));
        }
    }

    Ok(())
}

/// Entry point: compute the URL and target path, ensure directory, then download with resume.
pub async fn run_load(which: Option<&str>) -> Result<()> {
    let (url, file_name) = build_url(which);
    let weights_directory_path = weights_directory();
    ensure_directory(&weights_directory_path)?;
    let target_path = weights_directory_path.join(&file_name);

    eprintln!(
        "please load: downloading {file_name} -> {}",
        target_path.display()
    );
    download_with_resume(&url, &target_path).await?;
    eprintln!("please load: done");
    Ok(())
}
