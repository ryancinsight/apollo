//! Infrastructure for SHT WGPU execution.

/// Host-side conversion, validation, and quantization helpers.
pub(crate) mod conversion;
/// Hephaestus kernel definitions for the SHT passes.
pub mod kernel;
