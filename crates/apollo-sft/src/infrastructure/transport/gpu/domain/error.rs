//! Hephaestus accelerator error contracts.

use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Result alias for SFT accelerator execution.
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

    /// Requested precision profile does not match the admitted GPU storage.
    #[error("precision profile does not match typed GPU storage")]
    InvalidPrecisionProfile,

    /// Plan or spectrum parameters violate the accelerator contract.
    #[error("invalid plan: {message}")]
    InvalidPlan {
        /// Failure explanation including the offending value.
        message: String,
    },

    /// A sparse-spectrum component cannot enter concrete `f32` accelerator storage.
    #[error("sparse spectrum {component} component {value} cannot be represented exactly as f32")]
    PrecisionLoss {
        /// Complex component name.
        component: &'static str,
        /// Rejected high-accuracy value.
        value: f64,
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
