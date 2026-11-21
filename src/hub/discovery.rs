//! Find the installed checkpoint a new hub should load.
use super::dialect::Dialect;
use eyre::Result;
use std::path::PathBuf;

/// A complete local checkpoint selected for the next hub.
#[derive(Debug, Clone)]
pub struct Model {
    pub path: PathBuf,
    pub name: String,
    pub(super) dialect: Dialect,
    pub(super) parameters: u64,
    pub(super) bytes: u64,
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
        name: info.name,
        dialect,
        parameters: info.parameters,
        bytes: info.file_bytes,
    })
}

/// Inspect only completed files in the weights directory; tilde backups and partial downloads stay hidden.
pub fn installed_models(directory: &std::path::Path, budget: u64) -> Result<Vec<Model>> {
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
            Ok(model) if model.bytes <= budget => models.push(model),
            Ok(_) => {}
        }
    }
    models.sort_by(|a, b| {
        (b.dialect == Dialect::Qwenish, b.parameters, b.bytes)
            .cmp(&(a.dialect == Dialect::Qwenish, a.parameters, a.bytes))
            .then_with(|| a.path.cmp(&b.path))
    });
    Ok(models)
}

/// Prefer the largest fitting dense Qwen, then an installed gpt-oss checkpoint.
pub fn selected_model() -> Result<Option<Model>> {
    let explicit = std::env::var_os("PLEASE_MODEL");
    if let Some(path) = explicit {
        let model = inspect(path.into())?;
        return Ok(Some(model));
    }
    let directory = weights_dir();
    let budget = model_budget()?;
    let models = installed_models(&directory, budget)?;
    Ok(models.into_iter().next())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    fn checkpoint(path: &std::path::Path, architecture: &str, parameters: u64) -> Result<()> {
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
    fn discovery_prefers_dense_qwen_and_respects_hiding_and_budget() -> Result<()> {
        let directory = TempDir::new()?;
        let small = directory.0.join("unrelated-name.gguf");
        let large = directory.0.join("large.gguf");
        let fallback = directory.0.join("fallback.gguf");
        checkpoint(&small, "qwen35", 64)?;
        checkpoint(&large, "qwen35", 128)?;
        checkpoint(&fallback, "gpt-oss", 256)?;
        checkpoint(&directory.0.join("moe.gguf"), "qwen35moe", 512)?;
        checkpoint(&directory.0.join("unfinished.gguf.part"), "qwen35", 1024)?;
        std::fs::write(directory.0.join("broken.gguf"), b"GGUF")?;
        let models = installed_models(&directory.0, u64::MAX)?;
        assert_eq!(
            models.iter().map(|model| &model.path).collect::<Vec<_>>(),
            [&large, &small, &fallback]
        );
        let small_bytes = std::fs::metadata(&small)?.len();
        let models = installed_models(&directory.0, small_bytes)?;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].path, small);
        std::fs::rename(&small, small.with_extension("gguf~"))?;
        std::fs::rename(&large, large.with_extension("gguf~"))?;
        let models = installed_models(&directory.0, u64::MAX)?;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].path, fallback);
        Ok(())
    }
}
