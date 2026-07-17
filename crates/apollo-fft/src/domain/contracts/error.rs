//! Error contracts for Apollo FFT.

use thiserror::Error;

/// Result type used throughout Apollo FFT.
pub type ApolloResult<T> = Result<T, ApolloError>;

/// Error contract for Apollo FFT operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ApolloError {
    /// The caller supplied an invalid field value.
    #[error("validation failed for `{field}`: `{value}` violates `{constraint}`")]
    Validation {
        /// The field name that failed validation.
        field: String,
        /// The invalid value rendered as text.
        value: String,
        /// The constraint that the field violated.
        constraint: String,
    },
    /// The caller supplied arrays whose shapes are incompatible with the plan.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        actual: String,
    },
    /// A required buffer was not standard-layout contiguous.
    #[error("non-contiguous buffer: {context}")]
    NonContiguous {
        /// Context explaining which buffer violated the layout contract.
        context: String,
    },
    /// The requested backend is not available in this build or on this host.
    #[error("backend unavailable: {backend}")]
    BackendUnavailable {
        /// Backend identifier.
        backend: String,
    },
    /// A WGPU-specific backend failure occurred.
    #[error("wgpu backend error: {message}")]
    Wgpu {
        /// Error message propagated from the backend.
        message: String,
    },
    /// A CUDA-specific provider failure occurred.
    #[error("cuda backend error: {message}")]
    Cuda {
        /// Error message propagated from the backend.
        message: String,
    },
}

impl ApolloError {
    /// Construct a validation error.
    #[must_use]
    pub fn validation(
        field: impl Into<String>,
        value: impl Into<String>,
        constraint: impl Into<String>,
    ) -> Self {
        Self::Validation {
            field: field.into(),
            value: value.into(),
            constraint: constraint.into(),
        }
    }
}
