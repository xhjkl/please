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
