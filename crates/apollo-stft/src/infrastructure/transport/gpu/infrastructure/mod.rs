//! WGPU infrastructure for the STFT backend.

/// Pre-allocated GPU buffer sets for reusable dispatch.
pub mod buffers;
/// Provider-built device backend descriptor.
pub mod device;
/// GPU compute kernel for the forward STFT.
pub mod kernel;
