//! Prompts' cacheable prefixes, saved so a fresh session skips prefilling instructions and tools.
//!
//! oco restores a saved context without checking where it came from, so a file's name pins
//! everything its bytes depend on: the checkpoint, this build (which compiles in oco's kernels
//! and session layout), and the prefix's IDs. Cache filenames encode the first 128 bits of
//! the resulting SHA-256 in z-base-32. Stale files are left for pruning.
use eyre::Result;
use sha2::{Digest, Sha256};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Prefixes kept across all checkpoints, least recently used out first; a checkpoint has one,
/// or one per effort for gpt-oss, whose prefix holds it.
const KEPT: usize = 4;

/// Where cached prefixes live, beside `~/.please/weights`.
fn directory() -> PathBuf {
    super::directory().join("caches")
}

/// What a saved prefix's bytes depend on besides its IDs.
#[derive(Clone)]
pub struct Origin(Sha256);

impl Origin {
    /// Identify the canonical checkpoint at `model` and the running build, each by path,
    /// length, and modification time, so replacing either names every prefix anew.
    pub fn new(model: &Path) -> Result<Self> {
        let build = std::env::current_exe()?;
        let mut hash = Sha256::new();
        for path in [model, &build] {
            let metadata = std::fs::metadata(path)?;
            let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?;
            hash.update(path.as_os_str().as_bytes());
            hash.update([0]);
            hash.update(metadata.len().to_le_bytes());
            hash.update(modified.as_nanos().to_le_bytes());
        }
        Ok(Self(hash))
    }

    /// Where this origin's `prefix` is saved.
    pub fn path(&self, prefix: &[u32]) -> PathBuf {
        let mut ids = Sha256::new();
        for id in prefix {
            ids.update(id.to_le_bytes());
        }
        let mut hash = self.0.clone();
        hash.update(ids.finalize());
        directory().join(super::digest::name(hash))
    }
}

/// Write a prefix through `write`, private to the user, then keep only the most recently used.
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

/// Mark a restored prefix as just used, so pruning evicts the prefixes nobody opens.
pub fn touch(path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::options().write(true).open(path)?;
    file.set_modified(SystemTime::now())
}

/// Delete prefixes beyond the [`KEPT`] most recently used; each is as large as its attention
/// cache. Another hub pruning at the same time may have removed a file already. Whatever is
/// not named as a prefix, such as a save still being written, is left alone.
fn prune(directory: &Path) -> std::io::Result<()> {
    let mut prefixes = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if super::digest::is_name(&entry.file_name().to_string_lossy()) {
            let modified = entry.metadata()?.modified()?;
            prefixes.push((modified, entry.path()));
        }
    }
    prefixes.sort();
    for (_, path) in prefixes.iter().rev().skip(KEPT) {
        let removed = std::fs::remove_file(path);
        match removed {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::digest::z_base_32;
    use crate::test_support::TempDir;
    use std::time::Duration;

    #[test]
    fn prefixes_are_named_by_checkpoint_build_and_content() -> Result<()> {
        let directory = TempDir::new()?;
        let model = directory.0.join("model.gguf");
        std::fs::write(&model, b"weights")?;
        let origin = Origin::new(&model)?;
        let first = origin.path(&[1, 2, 3]);
        assert_eq!(first, Origin::new(&model)?.path(&[1, 2, 3]));
        assert_ne!(first, origin.path(&[1, 2, 4]));
        let name = first.file_name().unwrap().to_string_lossy();
        assert!(crate::hub::digest::is_name(&name));
        // Replacing the checkpoint in place renames all of its prefixes.
        std::fs::write(&model, b"other weights")?;
        assert_ne!(first, Origin::new(&model)?.path(&[1, 2, 3]));
        Ok(())
    }

    #[test]
    fn pruning_keeps_the_most_recently_used_prefixes_only() -> Result<()> {
        let directory = TempDir::new()?;
        let prefix = |index: u128| directory.0.join(z_base_32(index));
        let pending = directory.0.join("b.123.0.tmp");
        let files = std::iter::once(pending.clone()).chain((0..6).map(prefix));
        for (index, path) in files.enumerate() {
            std::fs::write(&path, [index as u8])?;
            let time = SystemTime::UNIX_EPOCH + Duration::from_secs(index as u64);
            std::fs::File::options()
                .write(true)
                .open(&path)?
                .set_modified(time)?;
        }
        touch(&prefix(0))?;
        prune(&directory.0)?;
        assert!(pending.exists());
        assert_eq!(
            (0..6).map(prefix).filter(|path| path.exists()).count(),
            KEPT
        );
        assert!(prefix(0).exists() && prefix(5).exists() && !prefix(1).exists());
        Ok(())
    }
}
