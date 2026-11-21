use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

/// A uniquely named directory removed on drop, so tests touching the file system never collide.
pub struct TempDir(pub PathBuf);
impl TempDir {
    pub fn new() -> std::io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir();
        let pid = std::process::id();
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!("please-test-{pid}-{id}"));
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
