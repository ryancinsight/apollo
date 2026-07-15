//! Hephaestus WGPU error contracts.

use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Result alias for NTT accelerator execution.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Failures produced by the concrete Hephaestus WGPU implementation.
#[derive(Debug, Error)]
pub enum WgpuError {
    /// The selected provider rejected acquisition, allocation, compilation,
    /// dispatch, synchronization, or transfer.
    #[error("accelerator provider: {0}")]
    Provider(#[from] HephaestusError),

    /// The requested operation is unavailable for the selected capability set.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Operation requested by the caller.
        operation: &'static str,
    },

    /// Plan parameters are invalid for the selected transform kernel.
    #[error("invalid plan: {message}")]
    InvalidPlan {
        /// Failure explanation including the offending value.
        message: String,
    },

    /// Input or output length does not match the plan.
    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Length required by the plan.
        expected: usize,
        /// Length supplied by the caller.
        actual: usize,
    },

    /// A Leto input view does not meet the transform boundary contract.
    #[error("shape mismatch: {message}")]
    ShapeMismatch {
        /// Failure explanation including the offending shape.
        message: String,
    },
}
