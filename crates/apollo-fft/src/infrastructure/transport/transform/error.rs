//! Shared WGPU transform error contract (ADR 0037).

use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Result alias for WGPU transform execution.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Failures produced by a Hephaestus-backed WGPU transform implementation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WgpuError {
    /// The selected Hephaestus device rejected acquisition, allocation,
    /// compilation, dispatch, synchronization, or transfer.
    #[error("accelerator provider: {0}")]
    Provider(#[from] HephaestusError),

    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed GPU storage")]
    InvalidPrecisionProfile,

    /// Plan parameters are invalid for the transform kernel.
    #[error("invalid plan: {message}")]
    InvalidPlan {
        /// Failure explanation including the offending plan value.
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
}
