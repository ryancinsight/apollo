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

use std::sync::Arc;

use apollo_nufft::{UniformDomain1D, UniformGrid3D};

use crate::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::domain::capabilities::NufftWgpuCapabilities;
use crate::domain::error::{NufftWgpuError, NufftWgpuResult};
use crate::infrastructure::kernel::NufftGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn nufft_wgpu_available() -> bool {
    NufftWgpuBackend::try_default().is_ok()
}

/// WGPU NUFFT backend descriptor.
#[derive(Debug, Clone)]
pub struct NufftWgpuBackend {
    pub(crate) device: WgpuDevice,
    pub(crate) kernel: Arc<NufftGpuKernel>,
}

impl NufftWgpuBackend {
    /// Create a NUFFT WGPU backend from an existing device and queue.
    #[must_use]
    pub fn new(device: WgpuDevice) -> Self {
        Self {
            kernel: Arc::new(NufftGpuKernel::new(device.inner())),
            device,
        }
    }

    /// Create a backend by requesting a default WGPU adapter and device.
    pub fn try_default() -> NufftWgpuResult<Self> {
        let device = WgpuDevice::try_default_with_limits(
            "apollo-nufft-wgpu",
            wgpu::Limits {
                max_storage_buffers_per_shader_stage: 8,
                ..wgpu::Limits::downlevel_defaults()
            },
        )
        .map_err(NufftWgpuError::Device)?;
        Ok(Self::new(device))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> NufftWgpuCapabilities {
        NufftWgpuCapabilities::direct_all_fast_all(true)
    }

    /// Return the acquired WGPU device.
    #[must_use]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        self.device.device()
    }

    /// Return the acquired WGPU queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        self.device.queue()
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
