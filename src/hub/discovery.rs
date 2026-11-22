//! Find the installed checkpoint a new hub should load.
use super::dialect::Dialect;
use eyre::{Result, bail};
use std::cmp::Reverse;
use std::path::{Path, PathBuf};

/// A complete local checkpoint selected for the next hub.
#[derive(Debug)]
pub struct Model {
    pub path: PathBuf,
    pub dialect: Dialect,
    parameters: u64,
    bytes: u64,
}

/// Where `please load` stores checkpoints and discovery looks for them.
pub fn weights_dir() -> PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".please/weights")
}

/// Reserve working-set headroom for context, scratch storage, and the rest of the system.
pub fn model_budget() -> Result<u64> {
    let memory = super::intuition::available_memory_bytes()?;
    Ok(memory.saturating_mul(4) / 5)
}

/// Identify a checkpoint from its metadata without loading it onto Metal.
pub fn inspect(path: PathBuf) -> Result<Model> {
    let info = oco::inspect(&path).map_err(|error| eyre::eyre!("{}: {error:#}", path.display()))?;
    let dialect = Dialect::from_architecture(&info.architecture)?;
    Ok(Model {
        path,
        dialect,
        parameters: info.parameters,
        bytes: info.file_bytes,
    })
}

/// Inspect only completed files in the weights directory; tilde backups and partial downloads stay hidden.
fn installed_models(directory: &Path) -> Result<Vec<Model>> {
    let entries = std::fs::read_dir(directory);
    let entries = match entries {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(error) => return Err(error.into()),
        Ok(entries) => entries,
    };
    let mut models = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|extension| extension != "gguf") {
            continue;
        }
        let model = inspect(path);
        match model {
            Err(error) => tracing::debug!("ignoring checkpoint: {error:#}"),
            Ok(model) => models.push(model),
        }
    }
    let rank = |model: &Model| {
        let family = match model.dialect {
            Dialect::Inkling => 0,
            Dialect::Qwenish => 1,
            Dialect::Harmony => 2,
        };
        (family, Reverse(model.parameters), Reverse(model.bytes))
    };
    models.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.path.cmp(&b.path)));
    Ok(models)
}

/// Choose fitting weights from the preferred family. Fallbacks require Inkling's absence.
fn selected_index(models: &[Model], budget: u64) -> Result<Option<usize>> {
    let smallest_inkling = models
        .iter()
        .filter(|model| model.dialect == Dialect::Inkling)
        .min_by_key(|model| model.bytes);
    if let Some(model) = smallest_inkling
        && model.bytes > budget
    {
        bail!(
            "installed Inkling checkpoint {} requires {} bytes and exceeds the {budget}-byte memory budget",
            model.path.display(),
            model.bytes
        );
    }
    let family = smallest_inkling.map(|model| model.dialect);
    Ok(models.iter().position(|model| {
        model.bytes <= budget && family.is_none_or(|family| model.dialect == family)
    }))
}

fn select_installed(directory: &Path, budget: u64) -> Result<Option<Model>> {
    let mut models = installed_models(directory)?;
    let selection = selected_index(&models, budget)?;
    Ok(selection.map(|index| models.swap_remove(index)))
}

fn explicit_model() -> Option<PathBuf> {
    let path = std::env::var_os("PLEASE_MODEL");
    path.map(PathBuf::from)
}

/// Select a checkpoint for a new hub, honoring an explicit override first.
pub fn selected_model() -> Result<Option<Model>> {
    let explicit = explicit_model();
    if let Some(path) = explicit {
        let model = inspect(path)?;
        return Ok(Some(model));
    }
    let directory = weights_dir();
    let budget = model_budget()?;
    select_installed(&directory, budget)
}

/// List supported checkpoints with the running or newly selected model first.
pub fn model_paths(running: Option<PathBuf>) -> Result<Vec<PathBuf>> {
    let explicit = explicit_model();
    let picked = running.or(explicit);
    let directory = weights_dir();
    let mut models = installed_models(&directory)?;
    if picked.is_none() && models.len() > 1 {
        let budget = model_budget();
        let selection = budget.and_then(|budget| selected_index(&models, budget));
        if let Ok(Some(index)) = selection {
            models[..=index].rotate_right(1);
        }
    }
    let mut paths: Vec<_> = models.into_iter().map(|model| model.path).collect();
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
        let models = installed_models(&directory.0)?;
        let paths: Vec<_> = models.iter().map(|model| &model.path).collect();
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
