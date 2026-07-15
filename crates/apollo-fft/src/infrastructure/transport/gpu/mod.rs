#![warn(missing_docs)]
//! Hephaestus device-backed dense FFT surface for Apollo.
//!
//! The adapter acquires `hephaestus_wgpu::WgpuDevice` and exposes the same
//! numerical contract as the CPU dense FFT backend. NUFFT-specific GPU
//! execution is intentionally owned by `apollo-nufft-wgpu`.

pub mod application;
pub mod domain;
pub mod infrastructure;

use crate::domain::contracts::backend::BackendCapabilities;
use crate::{ApolloError, ApolloResult, BackendKind, FftBackend, Shape1D, Shape2D, Shape3D};
use hephaestus_wgpu::WgpuDevice;
pub use infrastructure::gpu_fft::{gpu_fft_available, GpuFft3d, GpuFft3dBuffers};

#[cfg(feature = "native-f16")]
pub use infrastructure::gpu_fft::GpuFft3dF16Native;

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct WgpuBackend {
    device: WgpuDevice,
}

impl WgpuBackend {
    /// Create a backend from an existing device and queue.
    #[must_use]
    pub fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> ApolloResult<Self> {
        Ok(Self::new(
            WgpuDevice::try_default("apollo-fft-wgpu").map_err(|e| ApolloError::Wgpu {
                message: e.to_string(),
            })?,
        ))
    }
}

impl FftBackend for WgpuBackend {
    type Plan1D = ();
    type Plan2D = ();
    type Plan3D = GpuFft3d;

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Wgpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::WGPU
    }

    fn plan_1d(&self, _shape: Shape1D) -> ApolloResult<Self::Plan1D> {
        Err(ApolloError::BackendUnavailable {
            backend: "wgpu 1D plans are not exposed in v1".to_string(),
        })
    }

    fn plan_2d(&self, _shape: Shape2D) -> ApolloResult<Self::Plan2D> {
        Err(ApolloError::BackendUnavailable {
            backend: "wgpu 2D plans are not exposed in v1".to_string(),
        })
    }

    fn plan_3d(&self, shape: Shape3D) -> ApolloResult<Self::Plan3D> {
        GpuFft3d::new(
            self.device.device().clone(),
            self.device.queue().clone(),
            shape.nx,
            shape.ny,
            shape.nz,
        )
        .map_err(|message| ApolloError::Wgpu { message })
    }
}
