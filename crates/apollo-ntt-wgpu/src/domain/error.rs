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
    /// Plan metadata is invalid for the requested NTT.
    #[error("invalid NTT WGPU plan: len={len}, modulus={modulus}, primitive_root={primitive_root}, reason={message}")]
    InvalidPlan {
        /// Transform length.
        len: usize,
        /// Modulus.
        modulus: u64,
        /// Primitive root.
        primitive_root: u64,
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
    /// Reusable buffer length is invalid.
    #[error("invalid NTT WGPU reusable buffer length: len={len}")]
    InvalidBufferLength {
        /// Buffer transform length.
        len: usize,
    },
    /// Reusable buffer length does not match the plan.
    #[error("NTT WGPU reusable buffer length mismatch: expected {expected}, got {actual}")]
    BufferLengthMismatch {
        /// Plan length.
        expected: usize,
        /// Buffer length.
        actual: usize,
    },
    /// Output length does not match the plan.
    #[error("output length mismatch: expected {expected}, got {actual}")]
    OutputLengthMismatch {
        /// Plan length.
        expected: usize,
        /// Output length.
        actual: usize,
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
