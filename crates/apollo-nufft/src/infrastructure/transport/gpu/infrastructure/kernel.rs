//! Provider-native NUFFT kernel hierarchy.
//!
//! Apollo owns the direct-sum and Kaiser--Bessel algorithms. Hephaestus owns
//! device allocation, binding validation, dispatch, submission, and transfer.

/// Reusable provider-owned fast-path buffers.
pub mod buffers;
mod descriptors;
mod direct;
mod fast;
mod fast_support;

pub use buffers::{NufftGpuBuffers1D, NufftGpuBuffers3D};
#[cfg(any(test, feature = "diagnostics"))]
pub use buffers::{NufftGridSnapshot, NufftType2GridDiagnostics};
pub(crate) use fast_support::{KaiserBesselOne, KaiserBesselThree};

/// Zero-sized NUFFT orchestration over the typed accelerator provider.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NufftGpuKernel;
