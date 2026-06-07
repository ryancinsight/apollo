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
    /// Numerical execution is unsupported for the requested operation.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },
    /// Plan carries an invalid length.
    #[error("invalid plan length {len}: {message}")]
    InvalidPlan {
        /// Invalid length value.
        len: usize,
        /// Failure description.
        message: &'static str,
    },
    /// Transform order is NaN or infinite.
    #[error("fractional order must be finite")]
    NonFiniteOrder,
    /// Input slice length does not match the plan length.
    #[error("input length mismatch: expected {expected}, got {actual}")]
    InputLengthMismatch {
        /// Length required by the plan.
        expected: usize,
        /// Length supplied by the caller.
        actual: usize,
    },
    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed FrFT WGPU storage")]
    InvalidPrecisionProfile,
    /// GPU buffer map operation failed.
    #[error("buffer map failed: {message}")]
    BufferMapFailed {
        /// Failure context.
        message: String,
    },
}
