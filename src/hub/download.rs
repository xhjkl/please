//! Fetch and verify the pinned Inkling checkpoint.
use super::discovery;
use eyre::{Result, WrapErr, bail, eyre};
use sha2::{Digest, Sha256};
use std::fmt::Write;
use std::io::Read;
use std::path::Path;

mod transfer;

const FILENAME: &str = "Inkling-2-Mini-IQ4_NL.gguf";
const BYTES: u64 = 17_627_940_768;
const SHA256: &str = "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4";
const URL: &str = concat!(
    "https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/",
    "e8a2cecf0e78914f07cc27b9be711a9b2c7561d9/",
    "Inkling-2-Mini-IQ4_NL.gguf",
);

/// Reuse a verified copy or publish the download after verifying its SHA-256.
pub async fn run() -> Result<()> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let directory = discovery::weights_dir();
    std::fs::create_dir_all(&directory)?;
    let lock = std::fs::File::options()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(directory.join(".download.lock"))?;
    // Holding the lock until publication serializes resumes of the same partial file.
    lock.try_lock()
        .map_err(|error| eyre!("cannot lock checkpoint downloads: {error}"))?;
    let target = directory.join(FILENAME);
    let exists = target.try_exists()?;
    if exists {
        verify_checkpoint(&target).await?;
        eprintln!("please load: using verified {}", target.display());
        return Ok(());
    }

    let budget = discovery::memory_budget()?;
    if BYTES > budget {
        bail!("{FILENAME} requires {BYTES} bytes and exceeds the {budget}-byte memory budget");
    }
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))?;
    let partial = target.with_extension("gguf.part");
    eprintln!("please load: downloading {FILENAME} ({BYTES} bytes)");
    download_one(URL, &partial, BYTES).await?;
    verify_checkpoint(&partial).await?;
    std::fs::rename(&partial, &target)?;
    eprintln!("please load: ready: {}", target.display());
    Ok(())
}

async fn verify_checkpoint(path: &Path) -> Result<()> {
    eprintln!("please load: verifying SHA-256 of {}", path.display());
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || verify(&path, BYTES, SHA256)).await?
}

/// Hash a completed file with bounded memory.
fn verify(path: &Path, bytes: u64, expected: &str) -> Result<()> {
    let mut file = std::fs::File::open(path)
        .wrap_err_with(|| format!("cannot open checkpoint {}", path.display()))?;
    let metadata = file.metadata()?;
    if metadata.len() != bytes {
        bail!(
            "{} is {} bytes, expected {bytes}; move it aside before retrying",
            path.display(),
            metadata.len()
        );
    }
    let mut buffer = vec![0; 128 * 1024];
    let mut hash = Sha256::new();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    let mut actual = String::with_capacity(64);
    for byte in hash.finalize() {
        write!(actual, "{byte:02x}")?;
    }
    if actual != expected {
        bail!(
            "{} has SHA-256 {actual}, expected {expected}; move it aside before retrying",
            path.display()
        );
    }
    Ok(())
}

async fn download_one(url: &str, path: &Path, expected: u64) -> Result<()> {
    let url = url.to_owned();
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || transfer::fetch(&url, &path, expected)).await?
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    #[test]
    fn verification_rejects_same_size_corruption() -> Result<()> {
        let directory = crate::test_support::TempDir::new()?;
        let path = directory.0.join("checkpoint.gguf.part");
        let hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        std::fs::write(&path, b"abc")?;
        verify(&path, 3, hash)?;
        std::fs::write(&path, b"abb")?;
        let result = verify(&path, 3, hash);
        assert!(result.unwrap_err().to_string().contains("has SHA-256"));
        Ok(())
    }
    async fn server(response: String) -> Result<(String, tokio::task::JoinHandle<String>)> {
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
        for response in [
            "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 3-5/6\r\nConnection: close\r\n\r\ndef",
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef",
        ] {
            std::fs::write(&path, b"abc")?;
            let (url, task) = server(response.to_string()).await?;
            download_one(&url, &path, 6).await?;
            let request = task.await?;
            assert!(request.to_lowercase().contains("range: bytes=3-"));
            let bytes = std::fs::read(&path)?;
            assert_eq!(bytes, b"abcdef");
        }
        // A complete partial file needs no network request before structural validation.
        download_one("http://127.0.0.1:1/unreachable", &path, 6).await?;
        Ok(())
    }

    #[tokio::test]
    async fn download_follows_redirects_and_discards_trailers() -> Result<()> {
        let directory = crate::test_support::TempDir::new()?;
        let path = directory.0.join("model.gguf.part");
        std::fs::write(&path, b"abc")?;
        let (target, final_request) = server(
            concat!(
                "HTTP/1.1 103 Early Hints\r\nLink: </model.gguf>\r\n\r\n",
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 3-5/6\r\n",
                "Transfer-Encoding: chunked\r\nTrailer: X-Trailer\r\nConnection: close\r\n\r\n",
                "3\r\ndef\r\n0\r\nX-Trailer: metadata\r\n\r\n",
            )
            .to_string(),
        )
        .await?;
        let (url, redirect_request) = server(format!(
            "HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 4\r\nConnection: close\r\n\r\nskip",
        )).await?;

        download_one(&url, &path, 6).await?;

        for request in [redirect_request.await?, final_request.await?] {
            assert!(request.to_lowercase().contains("range: bytes=3-"));
        }
        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes, b"abcdef");
        Ok(())
    }

    #[tokio::test]
    async fn failed_or_misaligned_download_keeps_existing_bytes() -> Result<()> {
        let directory = crate::test_support::TempDir::new()?;
        let path = directory.0.join("model.gguf.part");
        for response in [
            "HTTP/1.1 403 Forbidden\r\nContent-Length: 3\r\nConnection: close\r\n\r\nerr",
            "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 0-2/6\r\nConnection: close\r\n\r\nabc",
            "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc",
        ] {
            std::fs::write(&path, b"abc")?;
            let (url, task) = server(response.to_string()).await?;
            let result = download_one(&url, &path, 6).await;
            assert!(result.is_err());
            task.await?;
            let bytes = std::fs::read(&path)?;
            assert_eq!(bytes, b"abc");
        }
        Ok(())
    }
}
