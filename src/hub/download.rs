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
        verify(&target).await?;
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
    fetch(URL, &partial, BYTES).await?;
    verify(&partial).await?;
    std::fs::rename(&partial, &target)?;
    eprintln!("please load: ready: {}", target.display());
    Ok(())
}

/// Check that the file at `path` is the pinned checkpoint whole.
async fn verify(path: &Path) -> Result<()> {
    eprintln!("please load: verifying SHA-256 of {}", path.display());
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || verify_blocking(&path, BYTES, SHA256)).await?
}

/// Check a completed file's size and SHA-256, hashing it with bounded memory.
fn verify_blocking(path: &Path, bytes: u64, expected: &str) -> Result<()> {
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

/// Download what `path` still lacks of the `expected` bytes at `url`.
async fn fetch(url: &str, path: &Path, expected: u64) -> Result<()> {
    let url = url.to_owned();
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || transfer::fetch_blocking(&url, &path, expected)).await?
}

#[cfg(test)]
mod tests;
