//! WGPU device acquisition for NUFFT backends.

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
use crate::infrastructure::transport::gpu::domain::error::NufftWgpuResult;
use hephaestus_wgpu::{DevicePreference, WgpuDevice};

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

    /// Create a backend by requesting a default WGPU adapter and device.
    pub fn try_default() -> NufftWgpuResult<Self> {
        let mut limits = WgpuDevice::default_device_limits();
        limits.max_storage_buffers_per_shader_stage = Some(FAST_NUFFT_STORAGE_BINDINGS);
        let device =
            WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
                "apollo-nufft-wgpu",
                DevicePreference::HighPerformance,
                &[],
                limits,
            )?;
        Ok(Self::new(device))
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
