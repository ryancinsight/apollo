//! WGPU device acquisition and backend orchestration for the STFT.

use std::sync::Arc;
use apollo_wgpu_helpers::WgpuDevice;
use crate::infrastructure::kernel::StftGpuKernel;
use crate::domain::error::WgpuResult;
use crate::domain::capabilities::WgpuCapabilities;

/// Forward execution implementations.
pub mod forward;
/// Inverse execution implementations.
pub mod inverse;
/// Pre-allocated execution buffers.
pub mod buffers;
/// Layout translation helpers.
pub mod helpers;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    StftWgpuBackend::try_default().is_ok()
}

/// WGPU backend for the STFT.
///
/// Owns an acquired device/queue pair and a cached kernel pipeline.
#[derive(Debug, Clone)]
pub struct StftWgpuBackend {
    pub(crate) device: WgpuDevice,
    pub(crate) kernel: Arc<StftGpuKernel>,
}

impl StftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(StftGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-stft-wgpu")?)
    }

    /// Return truthful forward-and-inverse capability descriptor.
    #[must_use]
    pub fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_and_inverse(true)
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
}
