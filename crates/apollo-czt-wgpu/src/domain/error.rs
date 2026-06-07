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
    #[error("invalid CZT WGPU plan lengths input={input_len}, output={output_len}: {message}")]
    InvalidLength {
        /// Requested input length.
        input_len: usize,
        /// Requested output length.
        output_len: usize,
        /// Failure explanation.
        message: &'static str,
    },
    /// Input length does not match the plan.
    #[error("input length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Plan input length.
        expected: usize,
        /// Input length.
        actual: usize,
    },
    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed CZT WGPU storage")]
    InvalidPrecisionProfile,
    /// Spiral parameters are not valid for the transform.
    #[error("invalid CZT WGPU parameters: {message}")]
    InvalidParameters {
        /// Failure explanation.
        message: &'static str,
    },
    /// Host readback from the staging buffer failed.
    #[error("wgpu buffer map failed: {message}")]
    BufferMapFailed {
        /// Mapping failure context.
        message: String,
    },
    /// Numerical execution is unsupported for the requested operation.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },
}
