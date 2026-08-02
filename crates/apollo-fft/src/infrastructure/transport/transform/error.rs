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

    /// The requested operation is unavailable for the selected capability set.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Operation requested by the caller.
        operation: &'static str,
    },

    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed GPU storage")]
    InvalidPrecisionProfile,

    /// Plan parameters are invalid for the transform kernel.
    #[error("invalid plan: {message}")]
    InvalidPlan {
        /// Failure explanation including the offending plan value.
        message: String,
    },

    /// A plan parameter that must be finite is NaN or infinite.
    #[error("parameter {parameter} must be finite")]
    NonFiniteParameter {
        /// Name of the offending parameter.
        parameter: &'static str,
    },

    /// A multi-dimensional operand does not match the plan's shape.
    #[error("shape mismatch: {message}")]
    ShapeMismatch {
        /// Failure explanation naming the expected and offending shapes.
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
