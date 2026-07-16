//! Hephaestus-device-backed NUFFT backend descriptors.

/// Fast Type-1 execution implementations.
pub mod fast_type1;
/// Fast Type-2 execution implementations.
pub mod fast_type2;
/// Helper functions for device execution.
pub mod helpers;
/// Direct Type-1 execution implementations.
pub mod type1;
/// Direct Type-2 execution implementations.
pub mod type2;

use crate::{UniformDomain1D, UniformGrid3D};

use crate::infrastructure::transport::gpu::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::infrastructure::transport::gpu::domain::capabilities::NufftWgpuCapabilities;
use hephaestus_wgpu::{DeviceLimits, WgpuDevice};

/// Fast NUFFT shaders bind storage buffers `0..=6` in one shader stage.
const FAST_NUFFT_STORAGE_BINDINGS: u32 = 7;

/// WGPU NUFFT backend descriptor.
#[derive(Debug, Clone)]
pub struct NufftWgpuBackend {
    pub(crate) device: WgpuDevice,
}

impl NufftWgpuBackend {
    /// Create a NUFFT backend from an existing Hephaestus device.
    #[must_use]
    pub fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Return the Hephaestus limits required by the fast NUFFT kernels.
    ///
    /// The fast one- and three-dimensional shader descriptors bind seven
    /// storage buffers in one stage. The returned request preserves the
    /// provider default limits and raises only that lower bound.
    #[must_use]
    pub fn required_device_limits() -> DeviceLimits {
        let mut limits = WgpuDevice::default_device_limits();
        limits.max_storage_buffers_per_shader_stage = Some(FAST_NUFFT_STORAGE_BINDINGS);
        limits
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> NufftWgpuCapabilities {
        NufftWgpuCapabilities::direct_all_fast_all(true)
    }

    /// Return the provider-owned accelerator device.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a 1D plan descriptor.
    #[must_use]
    pub const fn plan_1d(
        &self,
        domain: UniformDomain1D,
        oversampling: usize,
        kernel_width: usize,
    ) -> NufftWgpuPlan1D {
        NufftWgpuPlan1D::new(domain, oversampling, kernel_width)
    }

    /// Create a 3D plan descriptor.
    #[must_use]
    pub const fn plan_3d(
        &self,
        grid: UniformGrid3D,
        oversampling: usize,
        kernel_width: usize,
    ) -> NufftWgpuPlan3D {
        NufftWgpuPlan3D::new(grid, oversampling, kernel_width)
    }
}

#[cfg(test)]
mod tests {
    use super::NufftWgpuBackend;

    #[test]
    fn fast_nufft_limit_matches_shader_storage_bindings() {
        let limits = NufftWgpuBackend::required_device_limits();
        assert_eq!(limits.max_storage_buffers_per_shader_stage, Some(7));
    }
}
