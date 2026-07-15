//! Typed accelerator error contracts.

use hephaestus_core::HephaestusError;

/// Failures produced at the STFT accelerator boundary.
#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    /// Provider allocation, transfer, compilation, or dispatch failure.
    #[error(transparent)]
    Provider(#[from] HephaestusError),
    /// Plan dimensions violate the STFT domain contract.
    #[error("invalid STFT plan: {message}")]
    InvalidPlan {
        /// The violated dimension or range invariant.
        message: String,
    },
    /// The input does not contain one complete frame.
    #[error("STFT input is too short: need at least {min}, got {actual}")]
    InputTooShort {
        /// Minimum required input length.
        min: usize,
        /// Supplied input length.
        actual: usize,
    },
    /// The supplied slice length differs from the plan contract.
    #[error("STFT length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// The typed storage profile does not match the requested execution profile.
    #[error("STFT precision profile is incompatible with accelerator storage")]
    InvalidPrecisionProfile,
    /// The requested operation is unavailable for this backend capability set.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },
}

/// Result type for STFT accelerator operations.
pub type WgpuResult<T> = Result<T, WgpuError>;
