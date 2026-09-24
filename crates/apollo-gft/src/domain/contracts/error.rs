//! Error contracts for graph Fourier transforms.

use thiserror::Error;

/// Result alias for GFT operations.
pub type GftResult<T> = Result<T, GftError>;

/// Errors produced by graph Fourier plan creation or execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GftError {
    /// Graph has no vertices.
    #[error("graph has no vertices")]
    EmptyGraph,
    /// Adjacency matrix is not square.
    #[error("adjacency matrix is not square")]
    NonSquareAdjacency,
    /// Adjacency matrix is not symmetric.
    #[error("adjacency matrix is not symmetric")]
    NonSymmetricAdjacency,
    /// Adjacency matrix contains a non-finite edge weight.
    #[error("adjacency matrix contains a non-finite edge weight")]
    NonFiniteWeight,
    /// Input length does not match the graph order.
    #[error("input length does not match graph order")]
    LengthMismatch,
    /// Precision profile does not match the requested storage type.
    #[error("precision profile does not match storage type")]
    PrecisionMismatch,
    /// The Laplacian eigendecomposition failed.
    #[error("Laplacian eigendecomposition failed: {0}")]
    SpectralDecomposition(SpectralFailure),
}

/// Why the Laplacian eigensolver rejected a graph that passed adjacency
/// validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SpectralFailure {
    /// The Laplacian was rejected as input: non-finite (degree sums past the
    /// scalar range) or not symmetric to the eigensolver's rounding bound.
    #[error("the eigensolver rejected the Laplacian as input")]
    RejectedInput,
    /// The eigensolver exhausted its iteration budget.
    #[error("the eigensolver did not converge within {max_iters} iterations")]
    NotConverged {
        /// The exhausted iteration budget.
        max_iters: usize,
    },
    /// An eigenvalue exceeds the scalar range.
    #[error("an eigenvalue exceeds the scalar range")]
    Overflow,
    /// A failure mode the eigensolver added after this mapping was written.
    #[error("the eigensolver failed")]
    Unclassified,
}
