//! WGPU error contracts.

use apollo_wgpu_helpers::WgpuDeviceError;
use thiserror::Error;

/// Result alias for WGPU operations.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Errors produced by WGPU backend operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WgpuError {
    /// WGPU device acquisition failed.
    #[error("wgpu device: {0}")]
    Device(#[from] WgpuDeviceError),
    /// Plan length is empty.
    #[error("invalid Hilbert WGPU plan length {len}: {message}")]
    InvalidLength {
        /// Requested logical length.
        len: usize,
        /// Failure explanation.
        message: &'static str,
    },
    /// Input length does not match the plan.
    #[error("input length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Plan length.
        expected: usize,
        /// Input length.
        actual: usize,
    },
    /// Host readback from the staging buffer failed.
    #[error("wgpu buffer map failed: {message}")]
    BufferMapFailed {
        /// Mapping failure context.
        message: String,
    },
    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed Hilbert WGPU storage")]
    InvalidPrecisionProfile,
    /// Numerical execution is unsupported for the requested operation.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },
}
