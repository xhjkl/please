//! Find the installed checkpoint a new hub should load.
use super::dialect::Dialect;
use eyre::{Result, bail};
use std::cmp::Reverse;
use std::path::{Path, PathBuf};

/// A complete checkpoint file on this machine, as its metadata describes it.
#[derive(Debug)]
pub struct Checkpoint {
    pub path: PathBuf,
    pub dialect: Dialect,
    parameters: u64,
    bytes: u64,
}

/// Locate where `please load` stores checkpoints and discovery looks for them.
pub fn weights_dir() -> PathBuf {
    super::directory().join("weights")
}

/// Bytes a checkpoint's weights may take: four fifths of what Metal recommends as the working
/// set and this process has not allocated yet. The rest is headroom for the context, scratch
/// storage, and the rest of the system.
pub fn memory_budget() -> Result<u64> {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    let device = MTLCreateSystemDefaultDevice();
    let device = device.ok_or_else(|| eyre::eyre!("no system Metal device found"))?;
    let recommended = device.recommendedMaxWorkingSetSize();
    let allocated = device.currentAllocatedSize();
    let available = recommended.saturating_sub(allocated as u64);
    Ok(available.saturating_mul(4) / 5)
}

/// Identify a checkpoint from its metadata without loading it onto Metal.
pub fn inspect(path: PathBuf) -> Result<Checkpoint> {
    let info = oco::inspect(&path).map_err(|error| eyre::eyre!("{}: {error:#}", path.display()))?;
    let dialect = Dialect::from_architecture(&info.architecture)?;
    Ok(Checkpoint {
        path,
        dialect,
        parameters: info.parameters,
        bytes: info.file_bytes,
    })
}

/// Inspect the weights directory's `.gguf` files alone: a download still in progress
/// (`.gguf.part`) or a checkpoint renamed to any other extension stays out of the choice.
fn installed(directory: &Path) -> Result<Vec<Checkpoint>> {
    let entries = std::fs::read_dir(directory);
    let entries = match entries {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(error) => return Err(error.into()),
        Ok(entries) => entries,
    };
    let mut checkpoints = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|extension| extension != "gguf") {
            continue;
        }
        let checkpoint = inspect(path);
        match checkpoint {
            Err(error) => tracing::debug!("ignoring checkpoint: {error:#}"),
            Ok(checkpoint) => checkpoints.push(checkpoint),
        }
    }
    let rank = |checkpoint: &Checkpoint| {
        let family = match checkpoint.dialect {
            Dialect::Inkling => 0,
            Dialect::Qwenish => 1,
            Dialect::Harmony => 2,
        };
        (
            family,
            Reverse(checkpoint.parameters),
            Reverse(checkpoint.bytes),
        )
    };
    checkpoints.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.path.cmp(&b.path)));
    Ok(checkpoints)
}

/// Choose fitting weights from the preferred family. Fallbacks require Inkling's absence.
fn selected_index(checkpoints: &[Checkpoint], budget: u64) -> Result<Option<usize>> {
    let smallest_inkling = checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.dialect == Dialect::Inkling)
        .min_by_key(|checkpoint| checkpoint.bytes);
    if let Some(checkpoint) = smallest_inkling
        && checkpoint.bytes > budget
    {
        bail!(
            "installed Inkling checkpoint {} requires {} bytes and exceeds the {budget}-byte memory budget",
            checkpoint.path.display(),
            checkpoint.bytes
        );
    }
    let family = smallest_inkling.map(|checkpoint| checkpoint.dialect);
    Ok(checkpoints.iter().position(|checkpoint| {
        checkpoint.bytes <= budget && family.is_none_or(|family| checkpoint.dialect == family)
    }))
}

fn select_installed(directory: &Path, budget: u64) -> Result<Option<Checkpoint>> {
    let mut checkpoints = installed(directory)?;
    let selection = selected_index(&checkpoints, budget)?;
    Ok(selection.map(|index| checkpoints.swap_remove(index)))
}

/// The checkpoint `PLEASE_MODEL` names, which overrides every choice among installed ones.
fn explicit() -> Option<PathBuf> {
    let path = std::env::var_os("PLEASE_MODEL");
    path.map(PathBuf::from)
}

/// Select the checkpoint a new hub loads, honoring an explicit override first.
pub fn select() -> Result<Option<Checkpoint>> {
    let explicit = explicit();
    if let Some(path) = explicit {
        let checkpoint = inspect(path)?;
        return Ok(Some(checkpoint));
    }
    let directory = weights_dir();
    let budget = memory_budget()?;
    select_installed(&directory, budget)
}

/// List supported checkpoints with the running or newly selected one first.
pub fn checkpoint_paths(running: Option<PathBuf>) -> Result<Vec<PathBuf>> {
    let explicit = explicit();
    let picked = running.or(explicit);
    let directory = weights_dir();
    let mut checkpoints = installed(&directory)?;
    if picked.is_none() && checkpoints.len() > 1 {
        let budget = memory_budget();
        let selection = budget.and_then(|budget| selected_index(&checkpoints, budget));
        if let Ok(Some(index)) = selection {
            checkpoints[..=index].rotate_right(1);
        }
    }
    let mut paths: Vec<_> = checkpoints
        .into_iter()
        .map(|checkpoint| checkpoint.path)
        .collect();
    if let Some(picked) = picked {
        promote_path(&mut paths, picked);
    }
    Ok(paths)
}

/// Prefer the installed name when the running hub reports a symlink's resolved path.
fn promote_path(paths: &mut Vec<PathBuf>, picked: PathBuf) {
    let canonical = picked.canonicalize();
    for (index, path) in paths.iter().enumerate() {
        let resolved = path.canonicalize();
        if *path == picked || matches!((&resolved, &canonical), (Ok(a), Ok(b)) if a == b) {
            paths[..=index].rotate_right(1);
            return;
        }
    }
    paths.insert(0, picked);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    fn checkpoint(path: &Path, architecture: &str, parameters: u64) -> Result<()> {
        fn text(bytes: &mut Vec<u8>, text: &str) {
            bytes.extend((text.len() as u64).to_le_bytes());
            bytes.extend(text.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        text(&mut bytes, "general.architecture");
        bytes.extend(8u32.to_le_bytes());
        text(&mut bytes, architecture);
        text(&mut bytes, "token_embd.weight");
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(parameters.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(32) * 32 + parameters as usize * 4, 0);
        std::fs::write(path, bytes)?;
        Ok(())
    }

    #[test]
    fn discovery_uses_fallbacks_only_in_inklings_absence() -> Result<()> {
        let directory = TempDir::new()?;
        let inkling = directory.0.join("inkling.gguf");
        let small = directory.0.join("unrelated-name.gguf");
        let large = directory.0.join("large.gguf");
        let fallback = directory.0.join("fallback.gguf");
        checkpoint(&inkling, "inkling2_loop", 192)?;
        checkpoint(&small, "qwen35", 64)?;
        checkpoint(&large, "qwen35", 128)?;
        checkpoint(&fallback, "gpt-oss", 256)?;
        checkpoint(&directory.0.join("moe.gguf"), "qwen35moe", 512)?;
        checkpoint(
            &directory.0.join("unfinished.gguf.part"),
            "inkling2_loop",
            1024,
        )?;
        std::fs::write(directory.0.join("broken.gguf"), b"GGUF")?;
        let checkpoints = installed(&directory.0)?;
        let paths: Vec<_> = checkpoints
            .iter()
            .map(|checkpoint| &checkpoint.path)
            .collect();
        assert_eq!(paths, [&inkling, &large, &small, &fallback]);
        let small_bytes = std::fs::metadata(&small)?.len();
        let selected = select_installed(&directory.0, u64::MAX)?;
        assert_eq!(selected.unwrap().path, inkling);
        let selected = select_installed(&directory.0, small_bytes);
        assert!(
            selected
                .unwrap_err()
                .to_string()
                .contains("installed Inkling")
        );
        std::fs::rename(&inkling, inkling.with_extension("gguf~"))?;
        let selected = select_installed(&directory.0, u64::MAX)?;
        assert_eq!(selected.unwrap().path, large);
        let selected = select_installed(&directory.0, small_bytes)?;
        assert_eq!(selected.unwrap().path, small);
        std::fs::rename(&small, small.with_extension("gguf~"))?;
        std::fs::rename(&large, large.with_extension("gguf~"))?;
        let selected = select_installed(&directory.0, u64::MAX)?;
        assert_eq!(selected.unwrap().path, fallback);
        Ok(())
    }
}
