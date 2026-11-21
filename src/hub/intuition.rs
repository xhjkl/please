//! Hardware-memory heuristics for model and context selection.

/// Report Metal's recommended working set minus what this process has already allocated.
pub fn available_memory_bytes() -> eyre::Result<u64> {
    let device = metal::Device::system_default();
    let device = device.ok_or_else(|| eyre::eyre!("no system Metal device found"))?;
    let recommended = device.recommended_max_working_set_size();
    let allocated = device.current_allocated_size();
    Ok(recommended.saturating_sub(allocated))
}
