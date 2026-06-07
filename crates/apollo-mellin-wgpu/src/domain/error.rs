//! WGPU error contracts.

use apollo_wgpu_helpers::WgpuDeviceError;
use thiserror::Error;

/// Result alias for WGPU operations.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Errors produced by WGPU backend operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum WgpuError {
    /// WGPU device acquisition failed.
    #[error("wgpu device: {0}")]
    Device(#[from] WgpuDeviceError),
    /// Plan sample count is empty or scales are invalid.
    #[error("invalid Mellin WGPU plan: samples={samples}, min_scale={min_scale}, max_scale={max_scale}, reason={message}")]
    InvalidPlan {
        /// Sample count.
        samples: usize,
        /// Minimum scale.
        min_scale: f64,
        /// Maximum scale.
        max_scale: f64,
        /// Failure explanation.
        message: &'static str,
    },
    /// Signal length does not match the expected contract.
    #[error("input length mismatch: expected at least {expected}, got {actual}")]
    LengthMismatch {
        /// Expected signal length lower bound.
        expected: usize,
        /// Actual signal length.
        actual: usize,
    },
    /// Signal domain bounds are invalid.
    #[error(
        "invalid Mellin WGPU signal domain: min={signal_min}, max={signal_max}, reason={message}"
    )]
    InvalidSignalDomain {
        /// Minimum signal-domain scale.
        signal_min: f64,
        /// Maximum signal-domain scale.
        signal_max: f64,
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
    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed Mellin WGPU storage")]
    InvalidPrecisionProfile,
}
