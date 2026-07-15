//! WGPU device acquisition and backend orchestration for the STFT.

use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::WgpuResult;
use hephaestus_wgpu::{DevicePreference, WgpuDevice};

/// Bluestein binds four working/kernel buffers and two operation I/O buffers.
const BLUESTEIN_STORAGE_BINDINGS: u32 = 6;

/// Pre-allocated execution buffers.
pub mod buffers;
/// Forward execution implementations.
pub mod forward;
/// Inverse execution implementations.
pub mod inverse;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    StftWgpuBackend::try_default().is_ok()
}

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

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        let mut limits = WgpuDevice::default_device_limits();
        limits.max_storage_buffers_per_shader_stage = Some(BLUESTEIN_STORAGE_BINDINGS);
        Ok(Self::new(
            WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
                "apollo-stft-wgpu",
                DevicePreference::HighPerformance,
                &[],
                limits,
            )?,
        ))
    }

    /// Return truthful forward-and-inverse capability descriptor.
    #[must_use]
    pub fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_and_inverse(true)
    }
}
