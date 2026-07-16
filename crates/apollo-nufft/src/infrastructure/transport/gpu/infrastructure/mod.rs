//! NUFFT WGPU infrastructure.

/// Provider-built device backend descriptor.
pub mod device;
/// NUFFT compute kernel orchestration.
pub mod kernel;

pub use kernel::{NufftGpuBuffers1D, NufftGpuBuffers3D};
#[cfg(any(test, feature = "diagnostics"))]
pub use kernel::{NufftGridSnapshot, NufftType2GridDiagnostics};
