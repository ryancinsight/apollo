//! Hephaestus accelerator error contracts.

use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Result alias for SHT accelerator execution.
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

    /// A high-accuracy coefficient cannot enter concrete accelerator storage.
    #[error("coefficient {component} component {value} cannot be represented exactly as f32")]
    PrecisionLoss {
        /// Complex component name.
        component: &'static str,
        /// Rejected value.
        value: f64,
    },

    /// Plan parameters violate the accelerator contract.
    #[error("invalid plan: {message}")]
    InvalidPlan {
        /// Failure explanation including the offending value.
        message: String,
    },

    /// Input or output shape does not match the plan.
    #[error("shape mismatch: {message}")]
    ShapeMismatch {
        /// Required and supplied shape details.
        message: String,
    },
}
