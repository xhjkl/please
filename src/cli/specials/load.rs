use eyre::{Result, eyre};
use futures_util::{StreamExt, future::try_join_all};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::AsyncWriteExt;

/// Return the local directory where model weight files are stored.
fn weights_dir() -> std::path::PathBuf {
    let home_directory = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    std::path::Path::new(&home_directory)
        .join(".please")
        .join("weights")
}

/// Ensure the given directory exists with secure permissions (0700 on Unix).
fn ensure_dir(path: &std::path::Path) -> Result<()> {
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

/// Pick the appropriate repository and shard list based on a user-friendly alias.
fn pick_repository(which: Option<&str>) -> (&'static str, &'static [&'static str]) {
    let key = which.map(|s| s.trim()).unwrap_or("20b");
    let key = key
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase();

    match key.as_str() {
        "120b" | "120" | "big" | "large" => (
            "ggml-org/gpt-oss-120b-GGUF",
            &[
                "gpt-oss-120b-mxfp4-00001-of-00003.gguf",
                "gpt-oss-120b-mxfp4-00002-of-00003.gguf",
                "gpt-oss-120b-mxfp4-00003-of-00003.gguf",
            ],
        ),
        _ => ("ggml-org/gpt-oss-20b-GGUF", &["gpt-oss-20b-mxfp4.gguf"]),
    }
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
        .build()?;
    Ok(client)
}

fn shard_url(repository: &str, shard: &str) -> String {
    format!("https://huggingface.co/{repository}/resolve/main/{shard}")
}

/// Parsed view of a Content-Range header.
/// Missing components remain None.
struct ContentRange {
    start: Option<u64>,
    total: Option<u64>,
}

impl ContentRange {
    /// Parse a Content-Range header string.
    /// Accepts "bytes start-end/total" or "bytes */total".
    fn parse(header: &str) -> Self {
        let header = header.trim();
        let mut header_parts = header.split_whitespace();
        let unit_token = header_parts.next();
        let range_value = header_parts.next();
        if unit_token != Some("bytes") || range_value.is_none() {
            return Self {
                start: None,
                total: None,
            };
        }
        let range_value = range_value.unwrap();
        let mut range_and_total = range_value.split('/');
        let range_part = range_and_total.next().unwrap_or("");
        let total_part = range_and_total.next().unwrap_or("");
        let total = if total_part == "*" {
            None
        } else {
            total_part.parse::<u64>().ok()
        };
        if range_part == "*" {
            return Self { start: None, total };
        }
        let mut start_end_split = range_part.split('-');
        let start = start_end_split.next().and_then(|s| s.parse::<u64>().ok());
        Self { start, total }
    }
}

struct Progress {
    total: Option<u64>,
    downloaded: AtomicU64,
}

impl Progress {
    fn new(total: Option<u64>) -> Self {
        Self {
            total,
            downloaded: AtomicU64::new(0),
        }
    }

    fn add(&self, delta: u64) {
        let downloaded = self.downloaded.fetch_add(delta, Ordering::Relaxed) + delta;
        if let Some(total) = self.total {
            let pct = (downloaded as f64 / total as f64) * 100.0;
            eprint!("\rplease load: {downloaded}/{total} bytes ({pct:.1}%)");
        } else {
            eprint!("\rplease load: {downloaded} bytes");
        }
        let _ = std::io::stderr().flush();
    }
}

/// Derive a multi-shard target file name by stripping "-<n>-of-<m>" if present.
fn derive_multishard_target_name(shard_name: &str) -> String {
    let of_pos = match shard_name.find("-of-") {
        Some(i) => i,
        None => return shard_name.to_string(),
    };

    let bytes = shard_name.as_bytes();

    // Walk backwards over digits before "-of-".
    let mut start = of_pos;
    while start > 0 && bytes[start - 1].is_ascii_digit() {
        start -= 1;
    }
    // Expect a '-' right before those digits.
    if start == 0 || bytes[start - 1] != b'-' {
        return shard_name.to_string();
    }
    start -= 1;

    // Walk forwards over digits after "-of-".
    let mut end = of_pos + "-of-".len();
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }

    // Ensure we actually had digits around "-of-".
    if end <= of_pos + "-of-".len() {
        return shard_name.to_string();
    }
    if start + 1 >= of_pos || !bytes[start + 1..of_pos].iter().all(|b| b.is_ascii_digit()) {
        return shard_name.to_string();
    }

    let mut result = String::with_capacity(shard_name.len());
    result.push_str(&shard_name[..start]);
    result.push_str(&shard_name[end..]);
    result
}

/// Truncate the file at the given path to the specified length.
async fn truncate_to(path: &std::path::Path, len: u64) -> Result<()> {
    let file_handle = tokio::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .await?;
    file_handle.set_len(len).await?;
    Ok(())
}

async fn open_for_resume(path: &std::path::Path, start_offset: u64) -> Result<tokio::fs::File> {
    let mut options = tokio::fs::OpenOptions::new();
    options.create(true).write(true);
    if start_offset == 0 {
        options.truncate(true);
    } else {
        options.append(true);
    }
    let file = options.open(path).await?;
    Ok(file)
}

async fn get(client: &reqwest::Client, url: &str) -> Result<reqwest::Response> {
    let response = client.get(url).send().await?;
    Ok(response)
}

/// Download a remote file to `target_path`, resuming from a local partial file when possible.
/// Robustly handles servers that ignore ranges or respond with 416, and verifies final size when known.
async fn download_with_resume(
    client: reqwest::Client,
    url: String,
    target_path: std::path::PathBuf,
    progress: Arc<Progress>,
) -> Result<()> {
    // Determine current size if a partially downloaded file already exists.
    let mut start_offset = 0u64;
    if let Ok(meta) = tokio::fs::metadata(&target_path).await {
        start_offset = meta.len();
    }

    // Try a HEAD to quickly determine total size (optimization and equality check).
    let mut total_bytes: Option<u64> = None;
    if let Ok(head) = client.head(&url).send().await
        && head.status().is_success()
    {
        total_bytes = head
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        if let Some(total) = total_bytes {
            if start_offset == total {
                eprintln!("please load: already present at {}", target_path.display());
                return Ok(());
            }
            if start_offset > total {
                // Local file longer than remote: suspicious; restart full download.
                eprintln!("please load: local larger than remote; restarting full download");
                start_offset = 0;
            }
        }
    }

    // Build initial GET (attempt ranged if we have partial local).
    let mut request = client.get(&url);
    if start_offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={}-", start_offset));
    }

    let mut response = request.send().await?;
    let mut status = response.status();

    // Handle 416 (Range Not Satisfiable).
    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let content_range_header = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|h| h.to_str().ok());
        if let Some(content_range_header) = content_range_header {
            let range = ContentRange::parse(content_range_header);
            if let Some(total) = range.total {
                if start_offset == total {
                    eprintln!("please load: already present at {}", target_path.display());
                    return Ok(());
                }
                if start_offset > total {
                    // Local file longer than remote: suspicious; restart full download.
                    eprintln!("please load: local larger than remote; restarting full download");
                }
            }
        }
        // Cannot satisfy range; restart full.
        start_offset = 0;
        response = get(&client, &url).await?;
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
            // Validate Content-Range alignment with local size.
            let content_range_header = response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|h| h.to_str().ok());
            if let Some(content_range_header) = content_range_header {
                let range = ContentRange::parse(content_range_header);
                if let Some(total) = range.total {
                    total_bytes = Some(total);
                }
                if range.start != Some(start_offset) {
                    // Mismatched range; restart full.
                    start_offset = 0;
                    response = get(&client, &url).await?;
                    status = response.status();
                }
            } else {
                // Partial without Content-Range, restart full.
                start_offset = 0;
                response = get(&client, &url).await?;
                status = response.status();
            }
        } else {
            // Got 200 OK ignoring Range -> restart from scratch using this response.
            truncate_to(&target_path, 0).await?;
            start_offset = 0;
        }
    }

    // Determine total if known from headers (helpful for progress).
    if total_bytes.is_none() {
        // For 200 OK, CONTENT_LENGTH is full total; for 206, it's remaining.
        let content_length = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        if status == reqwest::StatusCode::PARTIAL_CONTENT {
            if let Some(remaining) = content_length {
                total_bytes = Some(start_offset + remaining);
            }
        } else {
            total_bytes = content_length;
        }
    }

    // Open file in the appropriate mode.
    let mut file_handle = open_for_resume(&target_path, start_offset).await?;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let delta = chunk.len() as u64;
        file_handle.write_all(&chunk).await?;
        progress.add(delta);
    }

    file_handle.flush().await?;
    drop(file_handle);

    // Final verification if we know total size.
    if let Some(total) = total_bytes
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

async fn stitch_shards(
    target_path: &std::path::Path,
    shard_paths: &[std::path::PathBuf],
) -> Result<()> {
    let mut final_file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(target_path)
        .await?;

    for shard_path in shard_paths {
        let mut shard_file = tokio::fs::File::open(shard_path).await?;
        tokio::io::copy(&mut shard_file, &mut final_file).await?;
    }

    final_file.flush().await?;
    Ok(())
}

/// Entry point: resolve repository, download shards in parallel, and stitch them into the final file.
pub async fn run_load(which: Option<&str>) -> Result<()> {
    let (repository, shards) = pick_repository(which);
    let weights_directory_path = weights_dir();
    ensure_dir(&weights_directory_path)?;
    let client = build_http_client()?;

    let shard_count = shards.len();
    let first_shard = shards[0];
    let final_name = if shard_count == 1 {
        first_shard.to_string()
    } else {
        derive_multishard_target_name(first_shard)
    };
    let target_path = weights_directory_path.join(&final_name);
    let shard_jobs: Vec<(String, std::path::PathBuf)> = if shard_count == 1 {
        vec![(shard_url(repository, first_shard), target_path.clone())]
    } else {
        shards
            .iter()
            .map(|shard| {
                (
                    shard_url(repository, shard),
                    weights_directory_path.join(shard),
                )
            })
            .collect()
    };

    let final_name = target_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let final_dir = {
        let dir = target_path
            .parent()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| ".".to_string());
        if let Ok(home) = std::env::var("HOME") {
            if dir == home {
                "~".to_string()
            } else if dir.starts_with(&home) && dir.as_bytes().get(home.len()) == Some(&b'/') {
                format!("~{}", &dir[home.len()..])
            } else {
                dir
            }
        } else {
            dir
        }
    };
    eprintln!(
        "please load: downloading `{}` into `{}`",
        final_name, final_dir
    );

    let total_bytes = {
        let mut total = Some(0u64);
        for (url, _) in &shard_jobs {
            match client.head(url).send().await {
                Ok(head) if head.status().is_success() => {
                    if let Some(len) = head
                        .headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|h| h.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        if let Some(acc) = &mut total {
                            *acc += len;
                        }
                    } else {
                        total = None;
                        break;
                    }
                }
                _ => {
                    total = None;
                    break;
                }
            }
        }
        total
    };

    let progress = Arc::new(Progress::new(total_bytes));

    let download_tasks = shard_jobs.iter().map(|(url, path)| {
        let client = client.clone();
        let url = url.clone();
        let path = path.clone();
        let progress = Arc::clone(&progress);
        async move { download_with_resume(client, url, path, progress).await }
    });

    try_join_all(download_tasks).await?;

    if shard_count > 1 {
        let shard_paths: Vec<std::path::PathBuf> =
            shard_jobs.iter().map(|(_, path)| path.clone()).collect();
        stitch_shards(&target_path, &shard_paths).await?;
        for shard_path in &shard_paths {
            if let Err(e) = tokio::fs::remove_file(shard_path).await {
                eprintln!(
                    "please load: failed to remove {}: {e}",
                    shard_path.display()
                );
            }
        }
        eprintln!(
            "please load: stitched {} shards into {}",
            shard_count,
            target_path.display()
        );
    }

    eprintln!("please load: done");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_multishard_strips_index_pattern() {
        let name = "gpt-oss-120b-mxfp4-00001-of-00003.gguf";
        let derived = derive_multishard_target_name(name);
        assert_eq!(derived, "gpt-oss-120b-mxfp4.gguf");
    }

    #[test]
    fn derive_multishard_leaves_non_sharded_names() {
        let name = "gpt-oss-20b-mxfp4.gguf";
        let derived = derive_multishard_target_name(name);
        assert_eq!(derived, name);
    }

    #[test]
    fn derive_multishard_handles_simple_prefix_without_extension() {
        let name = "model-1-of-2";
        let derived = derive_multishard_target_name(name);
        assert_eq!(derived, "model");
    }

    #[test]
    fn derive_multishard_ignores_malformed_patterns() {
        let name = "model-of-two";
        let derived = derive_multishard_target_name(name);
        assert_eq!(derived, name);
    }

    #[test]
    fn content_range_parses_full_range_with_total() {
        let header = "bytes 0-9/100";
        let range = ContentRange::parse(header);
        assert_eq!(range.start, Some(0));
        assert_eq!(range.total, Some(100));
    }

    #[test]
    fn content_range_parses_unspecified_start_with_total() {
        let header = "bytes */100";
        let range = ContentRange::parse(header);
        assert_eq!(range.start, None);
        assert_eq!(range.total, Some(100));
    }

    #[test]
    fn content_range_parses_missing_total() {
        let header = "bytes 0-9/*";
        let range = ContentRange::parse(header);
        assert_eq!(range.start, Some(0));
        assert_eq!(range.total, None);
    }

    #[test]
    fn content_range_rejects_invalid_unit() {
        let header = "items 0-9/100";
        let range = ContentRange::parse(header);
        assert_eq!(range.start, None);
        assert_eq!(range.total, None);
    }
}
