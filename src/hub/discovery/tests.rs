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
