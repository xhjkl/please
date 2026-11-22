//! Hardware-memory heuristics for model and context selection.

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

/// Report Metal's recommended working set minus what this process has already allocated.
pub fn available_memory_bytes() -> eyre::Result<u64> {
    let device = MTLCreateSystemDefaultDevice();
    let device = device.ok_or_else(|| eyre::eyre!("no system Metal device found"))?;
    let recommended = device.recommendedMaxWorkingSetSize();
    let allocated = device.currentAllocatedSize();
    Ok(recommended.saturating_sub(allocated as u64))
}
