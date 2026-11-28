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
mod tests;
