#![warn(missing_docs)]
//! Hephaestus CUDA-backed dense FFT surface.
//!
//! Apollo supplies only the FFT kernel equation and validated plan values.
//! Hephaestus owns CUDA device acquisition, compilation, buffer lifetime,
//! binding validation, stream ordering, and synchronization.

mod kernel;
mod plan;

use crate::domain::contracts::backend::BackendCapabilities;
use crate::{ApolloError, ApolloResult, BackendKind, FftBackend, Shape1D, Shape2D, Shape3D};
use hephaestus_cuda::CudaDevice;
pub use plan::CudaFft1d;

/// CUDA backend descriptor over an existing typed Hephaestus device.
#[derive(Debug, Clone)]
pub struct CudaBackend {
    device: CudaDevice,
}

impl CudaBackend {
    /// Create a CUDA backend from an existing typed Hephaestus device.
    #[must_use]
    pub fn new(device: CudaDevice) -> Self {
        Self { device }
    }
}

impl FftBackend for CudaBackend {
    type Plan1D = CudaFft1d;
    type Plan2D = ();
    type Plan3D = ();

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::CUDA
    }

    fn plan_1d(&self, shape: Shape1D) -> ApolloResult<Self::Plan1D> {
        CudaFft1d::new(self.device.clone(), shape.n)
    }

    fn plan_2d(&self, _shape: Shape2D) -> ApolloResult<Self::Plan2D> {
        Err(ApolloError::BackendUnavailable {
            backend: "cuda 2D plans are not exposed in v1".to_string(),
        })
    }

    fn plan_3d(&self, _shape: Shape3D) -> ApolloResult<Self::Plan3D> {
        Err(ApolloError::BackendUnavailable {
            backend: "cuda 3D plans are not exposed in v1".to_string(),
        })
    }
}
