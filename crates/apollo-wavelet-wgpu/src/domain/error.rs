//! WGPU error contracts for the Haar DWT backend.

use apollo_wgpu_helpers::WgpuDeviceError;
use thiserror::Error;

/// Result alias for Haar DWT WGPU operations.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Enumeration of all failure modes for the Haar DWT WGPU backend.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WgpuError {
    /// Plan parameters are invalid (non-power-of-two length, zero levels, or level overflow).
    #[error("invalid wavelet plan: len={len}, levels={levels}: {message}")]
    InvalidLength {
        /// Signal length supplied to the plan.
        len: usize,
        /// Decomposition level count supplied to the plan.
        levels: usize,
        /// Human-readable description of the constraint that was violated.
        message: &'static str,
    },
    /// Input buffer length does not match the plan length.
    #[error("length mismatch: expected {expected}, actual {actual}")]
    LengthMismatch {
        /// Expected buffer length derived from the plan.
        expected: usize,
        /// Actual buffer length supplied by the caller.
        actual: usize,
    },
    /// GPU buffer mapping failed.
    #[error("buffer map failed: {message}")]
    BufferMapFailed {
        /// Error detail from the WGPU runtime.
        message: String,
    },
    /// WGPU device acquisition failed.
    #[error("wgpu device: {0}")]
    Device(#[from] WgpuDeviceError),
    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed Haar DWT WGPU storage")]
    InvalidPrecisionProfile,
    /// The requested operation is not implemented by the current WGPU capability set.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Name of the unsupported operation.
        operation: &'static str,
    },
}
