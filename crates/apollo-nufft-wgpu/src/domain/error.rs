//! NUFFT WGPU error contracts.

use apollo_wgpu_helpers::WgpuDeviceError;
use thiserror::Error;

/// Result alias for NUFFT WGPU operations.
pub type NufftWgpuResult<T> = Result<T, NufftWgpuError>;

/// Errors produced by NUFFT WGPU backend operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NufftWgpuError {
    /// WGPU device acquisition failed.
    #[error("wgpu device: {0}")]
    Device(#[from] WgpuDeviceError),
    /// Plan parameters are invalid for WGPU execution.
    #[error("invalid NUFFT WGPU plan: reason={message}")]
    InvalidPlan {
        /// Failure explanation.
        message: &'static str,
    },
    /// Positions and value arrays have incompatible lengths.
    #[error("input length mismatch: expected {expected}, got {actual}")]
    InputLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Host readback from the staging buffer failed.
    #[error("wgpu buffer map failed: {message}")]
    BufferMapFailed {
        /// Mapping failure context.
        message: String,
    },
    /// Numerical execution is unsupported for the requested operation.
    #[error("{operation} is unsupported by the current apollo-nufft-wgpu capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },
}
