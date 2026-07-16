//! Hephaestus-device-backed STFT backend descriptors.

use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use hephaestus_wgpu::{DeviceLimits, WgpuDevice};

/// Bluestein binds four working/kernel buffers and two operation I/O buffers.
const BLUESTEIN_STORAGE_BINDINGS: u32 = 6;

/// Pre-allocated execution buffers.
pub mod buffers;
/// Forward execution implementations.
pub mod forward;
/// Inverse execution implementations.
pub mod inverse;

/// WGPU backend for the STFT.
///
/// Owns a Hephaestus device for typed kernel dispatch.
#[derive(Debug, Clone)]
pub struct StftWgpuBackend {
    pub(crate) device: WgpuDevice,
}

impl StftWgpuBackend {
    /// Create a backend from an existing Hephaestus device.
    #[must_use]
    pub fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Return the Hephaestus limits required by Bluestein dispatch.
    ///
    /// The Bluestein chirp shader binds four working buffers and two operation
    /// buffers in one stage. The returned request preserves provider defaults
    /// and raises only that lower bound.
    #[must_use]
    pub fn required_device_limits() -> DeviceLimits {
        let mut limits = WgpuDevice::default_device_limits();
        limits.max_storage_buffers_per_shader_stage = Some(BLUESTEIN_STORAGE_BINDINGS);
        limits
    }

    /// Return truthful forward-and-inverse capability descriptor.
    #[must_use]
    pub fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_and_inverse(true)
    }
}

#[cfg(test)]
mod tests {
    use super::StftWgpuBackend;

    #[test]
    fn bluestein_limit_matches_shader_storage_bindings() {
        let limits = StftWgpuBackend::required_device_limits();
        assert_eq!(limits.max_storage_buffers_per_shader_stage, Some(6));
    }
}
