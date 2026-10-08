use eyre::{Result, eyre};
use futures_util::{
    StreamExt,
    future::{join_all, try_join_all},
};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::AsyncWriteExt;

use crate::intuition;

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

fn choose_best(models: &[(u64, &[&str])]) -> Result<Vec<String>> {
    let memory_bytes = intuition::available_memory_bytes()?;
    let budget_bytes = memory_bytes.saturating_mul(4) / 5;

    for &(model_bytes, urls) in models {
        if model_bytes <= budget_bytes {
            return Ok(urls.iter().map(|url| (*url).to_string()).collect());
        }
    }

    Err(eyre!(
        "no supported model fits the {} byte memory budget",
        budget_bytes
    ))
}

fn choose_best_qwen_urls() -> Result<Vec<String>> {
    const QWEN_MODELS: &[(u64, &[&str])] = &[
        (
            244_093_630_912,
            &[
                "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf",
                "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00002-of-00006.gguf",
                "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00003-of-00006.gguf",
                "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00004-of-00006.gguf",
                "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00005-of-00006.gguf",
                "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00006-of-00006.gguf",
            ],
        ),
        (
            76_536_964_608,
            &[
                "https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
                "https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00002-of-00003.gguf",
                "https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00003-of-00003.gguf",
            ],
        ),
        (
            22_016_023_168,
            &[
                "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf",
            ],
        ),
        (
            16_740_812_704,
            &[
                "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf",
            ],
        ),
        (
            5_680_522_464,
            &["https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf"],
        ),
        (
            2_740_937_888,
            &["https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"],
        ),
        (
            1_280_835_840,
            &["https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf"],
        ),
        (
            532_517_120,
            &[
                "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
            ],
        ),
    ];

    choose_best(QWEN_MODELS)
}

fn choose_best_gpt_urls() -> Result<Vec<String>> {
    const GPT_MODELS: &[(u64, &[&str])] = &[
        (
            63_387_346_208,
            &[
                "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-MXFP4.gguf",
            ],
        ),
        (
            12_109_566_624,
            &[
                "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf",
            ],
        ),
    ];

    choose_best(GPT_MODELS)
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
        .tls_backend_rustls()
        .build()?;
    Ok(client)
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
async fn download_one(
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

async fn download_many(urls: Vec<String>) -> Result<Vec<std::path::PathBuf>> {
    let weights_directory_path = weights_dir();
    ensure_dir(&weights_directory_path)?;
    let client = build_http_client()?;

    let shard_jobs = urls
        .into_iter()
        .map(|url| {
            let file_name = url
                .rsplit('/')
                .next()
                .filter(|file_name| !file_name.is_empty())
                .ok_or_else(|| eyre!("model shard URL has no file name: {url}"))?;
            let path = weights_directory_path.join(file_name);
            Ok((url, path))
        })
        .collect::<Result<Vec<_>>>()?;

    let head_requests = shard_jobs.iter().map(|(url, _)| {
        let client = client.clone();
        async move {
            let response = client.head(url).send().await.ok()?;
            if !response.status().is_success() {
                return None;
            }
            response
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)?
                .to_str()
                .ok()?
                .parse::<u64>()
                .ok()
        }
    });
    let content_lengths = join_all(head_requests).await;
    let total_bytes = content_lengths.into_iter().sum::<Option<u64>>();

    let progress = Arc::new(Progress::new(total_bytes));
    let downloads = shard_jobs.iter().map(|(url, path)| {
        let client = client.clone();
        let url = url.clone();
        let path = path.clone();
        let progress = Arc::clone(&progress);
        async move { download_one(client, url, path, progress).await }
    });
    try_join_all(downloads).await?;

    Ok(shard_jobs.into_iter().map(|(_, path)| path).collect())
}

async fn stitch_shards(shard_paths: Vec<std::path::PathBuf>) -> Result<std::path::PathBuf> {
    let Some(first_shard_path) = shard_paths.first() else {
        return Err(eyre!("model has no shards"));
    };
    if shard_paths.len() == 1 {
        return Ok(first_shard_path.clone());
    }

    let first_shard_name = first_shard_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            eyre!(
                "model shard has no file name: {}",
                first_shard_path.display()
            )
        })?;
    let target_name = derive_multishard_target_name(first_shard_name);
    let target_path = first_shard_path.with_file_name(target_name);
    let mut final_file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&target_path)
        .await?;

    for shard_path in &shard_paths {
        let mut shard_file = tokio::fs::File::open(shard_path).await?;
        tokio::io::copy(&mut shard_file, &mut final_file).await?;
    }

    final_file.flush().await?;
    drop(final_file);

    for shard_path in &shard_paths {
        let removed = tokio::fs::remove_file(shard_path).await;
        if let Err(error) = removed {
            eprintln!(
                "please load: failed to remove {}: {error}",
                shard_path.display()
            );
        }
    }

    Ok(target_path)
}

/// Select and download the greatest supported model that fits this machine.
pub async fn run_download(which: Option<&str>) -> Result<()> {
    let which = which.map(str::to_ascii_lowercase);
    let urls = if which.is_some_and(|which| which.starts_with("gpt")) {
        choose_best_gpt_urls()?
    } else {
        choose_best_qwen_urls()?
    };
    let shards = download_many(urls).await?;
    let model_path = stitch_shards(shards).await?;
    eprintln!("please load: done: {}", model_path.display());
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
