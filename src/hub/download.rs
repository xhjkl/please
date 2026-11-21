//! Fetch the largest pinned dense Qwen that fits this machine.
use super::dialect::Dialect;
use super::discovery;
use eyre::{Result, eyre};
use futures_util::StreamExt;
use std::path::Path;
use tokio::io::AsyncWriteExt;

/// A pinned checkpoint and the exact size its download must reach.
struct Model {
    name: &'static str,
    bytes: u64,
    /// Lower bound for this parameter class, allowing rounded checkpoint size labels.
    minimum_parameters: u64,
    url: &'static str,
}

// Pinning single-file dense checkpoints; no MoE, split GGUFs, or repository snapshots.
const MODELS: &[Model] = &[
    Model {
        name: "Qwen3.8-27B",
        bytes: 16_056_478_688,
        minimum_parameters: 24_000_000_000,
        url: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q4_0.gguf",
    },
    Model {
        name: "Qwen3.5-9B",
        bytes: 5_680_522_464,
        minimum_parameters: 8_000_000_000,
        url: "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf",
    },
    Model {
        name: "Qwen3.5-4B",
        bytes: 2_740_937_888,
        minimum_parameters: 3_500_000_000,
        url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
    },
    Model {
        name: "Qwen3.5-2B",
        bytes: 1_280_835_840,
        minimum_parameters: 1_500_000_000,
        url: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf",
    },
    Model {
        name: "Qwen3.5-0.8B",
        bytes: 532_517_120,
        minimum_parameters: 600_000_000,
        url: "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf",
    },
];

/// Pick the largest pinned checkpoint within the memory budget; the list runs largest first.
fn choose_best(budget: u64) -> Result<&'static Model> {
    MODELS
        .iter()
        .find(|model| model.bytes <= budget)
        .ok_or_else(|| eyre!("no supported dense Qwen fits the {budget}-byte memory budget"))
}

/// Accept only the dense Qwen family, so gpt-oss stays a fallback for existing weights.
fn validate_request(which: Option<&str>) -> Result<()> {
    if which.is_some_and(|which| !which.eq_ignore_ascii_case("qwen")) {
        eyre::bail!("please load only downloads dense Qwen; usage: please load [qwen]");
    }
    Ok(())
}

/// Reuse a fitting installed Qwen of the chosen size, otherwise fetch just its pinned GGUF.
pub async fn run_download(which: Option<&str>) -> Result<()> {
    validate_request(which)?;
    let budget = discovery::model_budget()?;
    let model = choose_best(budget)?;
    let directory = discovery::weights_dir();
    std::fs::create_dir_all(&directory)?;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let lock = std::fs::File::options()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(directory.join(".download.lock"))?;
    // Holding the lock until publication keeps concurrent resumes from appending the same bytes.
    lock.try_lock()
        .map_err(|error| eyre!("cannot lock checkpoint downloads: {error}"))?;
    let installed = discovery::installed_models(&directory, budget)?;
    if let Some(installed) = installed.iter().find(|installed| {
        installed.dialect == Dialect::Qwenish && installed.parameters >= model.minimum_parameters
    }) {
        eprintln!(
            "please load: using existing {} at {}",
            installed.name,
            installed.path.display()
        );
        return Ok(());
    }
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))?;
    let filename = model
        .url
        .rsplit('/')
        .next()
        .ok_or_else(|| eyre!("missing checkpoint filename"))?;
    let target = directory.join(filename);
    let exists = target.try_exists()?;
    if exists {
        eyre::bail!(
            "{} exists but is not a usable checkpoint; move it aside before retrying",
            target.display()
        );
    }
    let partial = target.with_extension("gguf.part");
    let client = reqwest::Client::builder()
        .user_agent(concat!("please/", env!("CARGO_PKG_VERSION")))
        .tls_backend_rustls()
        .build()?;
    eprintln!(
        "please load: {} ({} bytes; memory budget {} bytes)",
        model.name, model.bytes, budget
    );
    download_one(&client, model.url, &partial, model.bytes).await?;
    let downloaded = discovery::inspect(partial.clone())?;
    if downloaded.dialect != Dialect::Qwenish {
        eyre::bail!("downloaded checkpoint is not dense Qwen");
    }
    tokio::fs::rename(&partial, &target).await?;
    eprintln!("\nplease load: ready: {}", target.display());
    Ok(())
}

/// Resume into a hidden partial file and verify the pinned size before it becomes discoverable.
async fn download_one(
    client: &reqwest::Client,
    url: &str,
    path: &Path,
    expected: u64,
) -> Result<()> {
    let metadata = tokio::fs::metadata(path).await;
    let mut offset = match metadata {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
        Ok(metadata) => metadata.len(),
    };
    if offset == expected {
        return Ok(());
    }
    if offset > expected {
        eyre::bail!("partial download exceeds expected size: {}", path.display());
    }
    let mut request = client.get(url);
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let response = request.send().await?;
    let status = response.status();
    match status {
        reqwest::StatusCode::OK => offset = 0,
        reqwest::StatusCode::PARTIAL_CONTENT => {
            let range = response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|value| value.to_str().ok());
            let expected_range = format!("bytes {offset}-{}/{expected}", expected - 1);
            if range != Some(expected_range.as_str()) {
                eyre::bail!("unexpected download Content-Range: {range:?}");
            }
        }
        _ => return Err(eyre!("download failed: {status}")),
    }
    if let Some(length) = response.content_length()
        && length != expected - offset
    {
        eyre::bail!(
            "download size changed: expected {} bytes, got {length}",
            expected - offset
        );
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(offset == 0)
        .append(offset > 0)
        .open(path)
        .await?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.len() as u64 > expected - offset {
            eyre::bail!("download exceeds expected size");
        }
        file.write_all(&chunk).await?;
        offset += chunk.len() as u64;
        eprint!(
            "\rplease load: {offset}/{expected} bytes ({:.1}%)",
            offset as f64 * 100.0 / expected as f64
        );
    }
    file.flush().await?;
    if offset != expected {
        eyre::bail!("download size mismatch: expected {expected}, got {offset}");
    }
    file.sync_all().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_dense_qwen_is_downloadable() -> Result<()> {
        validate_request(None)?;
        validate_request(Some("Qwen"))?;
        for request in ["gpt", "gpt-oss", "gpt-oss-120b", "moe", "garbage"] {
            assert!(validate_request(Some(request)).is_err());
        }
        for (index, model) in MODELS.iter().enumerate() {
            assert_eq!(choose_best(model.bytes)?.name, model.name);
            assert!(model.url.contains("Qwen"));
            assert!(!model.url.contains("-A"));
            if let Some(next) = MODELS.get(index + 1) {
                assert_eq!(choose_best(model.bytes - 1)?.name, next.name);
            }
        }
        assert!(choose_best(MODELS.last().unwrap().bytes - 1).is_err());
        assert_eq!(choose_best(u64::MAX)?.name, MODELS[0].name);
        Ok(())
    }
    async fn server(response: String) -> Result<(String, tokio::task::JoinHandle<String>)> {
        use tokio::io::AsyncReadExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            loop {
                let byte = socket.read_u8().await.unwrap();
                request.push(byte);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            socket.write_all(response.as_bytes()).await.unwrap();
            String::from_utf8(request).unwrap()
        });
        Ok((format!("http://{address}/model.gguf"), task))
    }

    #[tokio::test]
    async fn download_resumes_and_handles_ignored_ranges() -> Result<()> {
        let directory = crate::test_support::TempDir::new()?;
        let path = directory.0.join("model.gguf.part");
        let client = reqwest::Client::new();
        for response in [
            "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 3-5/6\r\nConnection: close\r\n\r\ndef",
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef",
        ] {
            std::fs::write(&path, b"abc")?;
            let (url, task) = server(response.to_string()).await?;
            download_one(&client, &url, &path, 6).await?;
            let request = task.await?;
            assert!(request.to_lowercase().contains("range: bytes=3-"));
            let bytes = std::fs::read(&path)?;
            assert_eq!(bytes, b"abcdef");
        }
        // A complete partial file needs no network request before structural validation.
        download_one(&client, "http://127.0.0.1:1/unreachable", &path, 6).await?;
        Ok(())
    }

    #[tokio::test]
    async fn failed_or_misaligned_download_keeps_existing_bytes() -> Result<()> {
        let directory = crate::test_support::TempDir::new()?;
        let path = directory.0.join("model.gguf.part");
        let client = reqwest::Client::new();
        for response in [
            "HTTP/1.1 403 Forbidden\r\nContent-Length: 3\r\nConnection: close\r\n\r\nerr",
            "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 0-2/6\r\nConnection: close\r\n\r\nabc",
            "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc",
        ] {
            std::fs::write(&path, b"abc")?;
            let (url, task) = server(response.to_string()).await?;
            let result = download_one(&client, &url, &path, 6).await;
            assert!(result.is_err());
            task.await?;
            let bytes = std::fs::read(&path)?;
            assert_eq!(bytes, b"abc");
        }
        Ok(())
    }
}
