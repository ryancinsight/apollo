//! Hephaestus accelerator error contracts.

use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Result alias for Mellin accelerator execution.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Failures produced by the concrete Hephaestus WGPU implementation.
#[derive(Debug, Error)]
pub enum WgpuError {
    /// The selected provider rejected acquisition, allocation, compilation,
    /// dispatch, synchronization, or transfer.
    #[error("accelerator provider: {0}")]
    Provider(#[from] HephaestusError),

    /// Requested precision profile does not match the admitted GPU storage.
    #[error("precision profile does not match typed GPU storage")]
    InvalidPrecisionProfile,

    /// Numerical execution is unsupported for the requested operation.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Operation requested by the caller.
        operation: &'static str,
    },

    /// Plan parameters or configuration are invalid.
    #[error("invalid plan: {message}")]
    InvalidPlan {
        /// Failure explanation including the offending value.
        message: String,
    },

    /// Input or output length does not match the plan expectation.
    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual supplied length.
        actual: usize,
    },

    /// Signal or output domain bounds are invalid.
    #[error("invalid signal domain: {message}")]
    InvalidSignalDomain {
        /// Failure explanation including the offending bounds.
        message: String,
    },
}
