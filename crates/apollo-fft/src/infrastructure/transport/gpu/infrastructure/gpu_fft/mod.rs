//! Shader-backed 3D FFT implementation for the Apollo WGPU backend.

pub mod dispatch;
pub mod pipeline;
pub mod strategy;
pub mod workspace;

#[cfg(feature = "native-f16")]
pub mod f16_plan;

pub use pipeline::{gpu_fft_available, GpuFft3d};
pub use workspace::GpuFft3dBuffers;

#[cfg(feature = "native-f16")]
pub use f16_plan::GpuFft3dF16Native;
