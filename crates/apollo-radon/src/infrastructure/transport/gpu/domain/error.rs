//! Typed Radon accelerator errors.

use hephaestus_core::HephaestusError;

/// Radon accelerator failure preserving the violated boundary contract.
#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    /// Provider allocation, transfer, compilation, or dispatch failure.
    #[error(transparent)]
    Provider(#[from] HephaestusError),
    /// Geometry is not valid for this transform.
    #[error("invalid Radon plan: {message}")]
    InvalidPlan {
        /// Violated geometry or accelerator-range invariant.
        message: String,
    },
    /// Input shape differs from the plan geometry.
    #[error("Radon shape mismatch: {message}")]
    ShapeMismatch {
        /// Expected and actual shape detail.
        message: String,
    },
    /// Input length differs from the plan geometry.
    #[error("Radon length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Required element count.
        expected: usize,
        /// Provided element count.
        actual: usize,
    },
    /// Typed storage profile does not match the declared execution profile.
    #[error("Radon precision profile is incompatible with accelerator storage")]
    InvalidPrecisionProfile,
    /// The requested operation is not available on this backend.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },
}

/// Result type for Radon accelerator operations.
pub type WgpuResult<T> = Result<T, WgpuError>;
