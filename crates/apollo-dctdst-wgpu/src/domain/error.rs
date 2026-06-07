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
    #[error("invalid DCT/DST WGPU plan length {len}: {message}")]
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
    /// Requested transform kind is not implemented on this backend.
    #[error("transform kind {kind} is unsupported by the current WGPU capability set")]
    UnsupportedKind {
        /// Unsupported transform kind.
        kind: &'static str,
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
    #[error("precision profile does not match typed DCT/DST WGPU storage")]
    InvalidPrecisionProfile,
    /// Input shape does not match the plan dimensions.
    #[error(
        "input shape mismatch: expected {expected}x{expected} (n={expected}), got {rows}x{cols}"
    )]
    ShapeMismatch {
        /// Expected side length.
        expected: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },
    /// Input 3D shape does not match the plan dimensions.
    #[error("input 3D shape mismatch: expected {expected}^3, got {d0}x{d1}x{d2}")]
    ShapeMismatch3d {
        /// Expected side length.
        expected: usize,
        /// Actual dimension 0.
        d0: usize,
        /// Actual dimension 1.
        d1: usize,
        /// Actual dimension 2.
        d2: usize,
    },
}
