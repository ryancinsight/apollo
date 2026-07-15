//! NUFFT WGPU error contracts.

use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Result alias for NUFFT WGPU operations.
pub type NufftWgpuResult<T> = Result<T, NufftWgpuError>;

/// Errors produced by NUFFT WGPU backend operations.
#[derive(Debug, Error)]
pub enum NufftWgpuError {
    /// Provider acquisition, allocation, dispatch, or transfer failed.
    #[error(transparent)]
    Provider(#[from] HephaestusError),
    /// Composed dense-FFT stream construction or encoding failed.
    #[error(transparent)]
    Fft(#[from] apollo_fft::ApolloError),
    /// Plan parameters are invalid for WGPU execution.
    #[error("invalid NUFFT WGPU plan: reason={message}")]
    InvalidPlan {
        /// Failure explanation.
        message: &'static str,
    },
    /// Positions and value arrays have incompatible lengths.
    #[error("input length mismatch: expected {expected}, got {actual}")]
    InputLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// A host array could not preserve its validated logical layout.
    #[error("host array layout: {message}")]
    HostArrayLayout {
        /// Conversion failure detail.
        message: String,
    },
    /// Numerical execution is unsupported for the requested operation.
    #[error("{operation} is unsupported by the current apollo-nufft-wgpu capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },
}
