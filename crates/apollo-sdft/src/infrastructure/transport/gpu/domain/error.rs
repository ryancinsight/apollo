//! Hephaestus accelerator error contracts.

use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Result alias for SDFT accelerator execution.
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

    /// Plan or spectrum parameters violate the accelerator contract.
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
}
