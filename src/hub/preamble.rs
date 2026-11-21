//! Saved static prompt heads, so a fresh session skips prefilling instructions and tools.
//!
//! oco restores a saved context without checking where it came from, so a file's name pins
//! everything its bytes depend on: the checkpoint, this build (which compiles in oco's kernels
//! and session layout), and the head's IDs. Cache filenames encode the first 128 bits of
//! the resulting SHA-256 in z-base-32. Stale files are left for pruning.
use eyre::Result;
use sha2::{Digest, Sha256};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Heads kept across all checkpoints, least recently used out first: one per reasoning effort
/// of the default model, whose sentence opens its prompt, plus the fallback's.
const KEPT: usize = 4;

/// z-base-32 digit alphabet
const ALPHABET: &[u8; 32] = b"ybndrfg8ejkmcpqxot1uwisza345h769";

/// Where cached heads live, beside `~/.please/weights`.
fn directory() -> PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".please/caches")
}

/// Encode `n` as a z-base-32 integer without leading zero digits.
fn z_base_32(mut n: u128) -> String {
    let mut digits = Vec::new();
    loop {
        digits.push(ALPHABET[(n % 32) as usize]);
        n /= 32;
        if n == 0 {
            break;
        }
    }
    digits
        .iter()
        .rev()
        .map(|&digit| char::from(digit))
        .collect()
}

/// Whether `name` could be a head this module wrote; anything else in the directory, such as
/// a save still being written, is left alone.
fn is_head(name: &str) -> bool {
    (1..=26).contains(&name.len()) && name.bytes().all(|byte| ALPHABET.contains(&byte))
}

/// What a saved head's bytes depend on besides its IDs.
#[derive(Clone)]
pub struct Origin(Sha256);

impl Origin {
    /// Identify the canonical checkpoint at `model` and the running build, each by path,
    /// length, and modification time, so replacing either names every head anew.
    pub fn new(model: &Path) -> Result<Self> {
        let mut hash = Sha256::new();
        for path in [model, &std::env::current_exe()?] {
            let metadata = std::fs::metadata(path)?;
            let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?;
            hash.update(path.as_os_str().as_bytes());
            hash.update([0]);
            hash.update(metadata.len().to_le_bytes());
            hash.update(modified.as_nanos().to_le_bytes());
        }
        Ok(Self(hash))
    }

    /// Where the head `head` of this origin is saved.
    pub fn path(&self, head: &[u32]) -> PathBuf {
        let mut ids = Sha256::new();
        for id in head {
            ids.update(id.to_le_bytes());
        }
        let mut hash = self.0.clone();
        hash.update(ids.finalize());
        let mut name = [0; 16];
        name.copy_from_slice(&hash.finalize()[..16]);
        directory().join(z_base_32(u128::from_be_bytes(name)))
    }
}

/// Write a head through `write`, private to the user, then keep only the most recently used.
pub fn save(path: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let directory = directory();
    std::fs::create_dir_all(&directory)?;
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))?;
    write(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    prune(&directory)?;
    Ok(())
}

/// Mark a restored head as just used, so pruning evicts the heads nobody opens.
pub fn touch(path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::options().write(true).open(path)?;
    file.set_modified(SystemTime::now())
}

/// Delete heads beyond the [`KEPT`] most recently used; each is as large as its attention
/// cache. Another hub pruning at the same time may have removed a file already.
fn prune(directory: &Path) -> std::io::Result<()> {
    let mut heads = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if is_head(&entry.file_name().to_string_lossy()) {
            heads.push((entry.metadata()?.modified()?, entry.path()));
        }
    }
    heads.sort();
    for (_, path) in heads.iter().rev().skip(KEPT) {
        match std::fs::remove_file(path) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;
    use std::time::Duration;

    #[test]
    fn cache_names_use_z_base_32() {
        assert_eq!(z_base_32(0), "y");
        assert_eq!(z_base_32(1), "b");
        assert_eq!(z_base_32(32), "by");
        assert_eq!(z_base_32(u128::MAX), format!("8{}", "9".repeat(25)));
        assert!(is_head(&z_base_32(u128::MAX)) && is_head("y"));
        assert!(!is_head("ybnd.123.0.tmp") && !is_head("") && !is_head(&"y".repeat(27)));
    }

    #[test]
    fn heads_are_named_by_checkpoint_build_and_content() -> Result<()> {
        let directory = TempDir::new()?;
        let model = directory.0.join("model.gguf");
        std::fs::write(&model, b"weights")?;
        let origin = Origin::new(&model)?;
        let first = origin.path(&[1, 2, 3]);
        assert_eq!(first, Origin::new(&model)?.path(&[1, 2, 3]));
        assert_ne!(first, origin.path(&[1, 2, 4]));
        assert!(is_head(&first.file_name().unwrap().to_string_lossy()));
        // Replacing the checkpoint in place renames all of its heads.
        std::fs::write(&model, b"other weights")?;
        assert_ne!(first, Origin::new(&model)?.path(&[1, 2, 3]));
        Ok(())
    }

    #[test]
    fn pruning_keeps_the_most_recently_used_heads_only() -> Result<()> {
        let directory = TempDir::new()?;
        let head = |index: u128| directory.0.join(z_base_32(index));
        let pending = directory.0.join("b.123.0.tmp");
        let files = std::iter::once(pending.clone()).chain((0..6).map(head));
        for (index, path) in files.enumerate() {
            std::fs::write(&path, [index as u8])?;
            let time = SystemTime::UNIX_EPOCH + Duration::from_secs(index as u64);
            std::fs::File::options()
                .write(true)
                .open(&path)?
                .set_modified(time)?;
        }
        touch(&head(0))?;
        prune(&directory.0)?;
        assert!(pending.exists());
        assert_eq!((0..6).map(head).filter(|path| path.exists()).count(), KEPT);
        assert!(head(0).exists() && head(5).exists() && !head(1).exists());
        Ok(())
    }
}
