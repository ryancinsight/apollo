//! WGPU infrastructure.

/// Host conversion, grid construction, and Leto interop.
mod conversion;
/// WGPU device acquisition.
pub mod device;
/// SHT compute kernel orchestration.
pub mod kernel;
