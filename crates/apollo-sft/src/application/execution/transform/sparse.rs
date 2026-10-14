//! Sparse Fourier transform plan.
//!
//! # Mathematical contract
//!
//! For `x in C^N`, the plan computes the dense FFT coefficients in `O(N log N)`,
//! ranks them by squared magnitude with frequency-index tie-breaking via a
//! min-heap of size `K` in `O(N log K)`, and stores the largest `K` coefficients
//! in a sparse spectrum. Reconstruction expands the sparse spectrum to a dense
//! coefficient vector and evaluates the inverse FFT.
//!
//! ## Complexity
//! Spectrum density computation: `O(N log N)` via apollo-fft auto-selecting kernel.
//! Top-K selection: `O(N log K)` via min-heap of size `K`.
//!
//! The recovery kernel is dense and deterministic. Sparse plan parameters
//! remain explicit domain data so a later sublinear isolation kernel can
//! replace the infrastructure layer without changing this public API.
//!
//! ## Theorem: Top-K Coefficient Selection Optimality
//!
//! **Statement.** For a signal x ∈ ℂ^N with DFT coefficients X = FFT(x), the
//! K-sparse approximation X_K obtained by retaining the K largest-magnitude
//! coefficients minimizes the squared reconstruction error:
//!
//! ```text
//! ‖x - IFFT(X_K)‖² = Σ_{j=K+1}^{N} |X[π(j)]|² / N²
//! ```
//!
//! where π is the permutation sorting |X| in descending order.
//!
//! **Proof sketch.** Parseval's theorem states ‖u - v‖² = (1/N)‖FFT(u) - FFT(v)‖²
//! for all u, v ∈ ℂ^N. The reconstruction error after zeroing all but the K
//! retained DFT bins equals (1/N) Σ_{j: not in top-K} `|X[j]|²`. This sum is
//! minimised by retaining exactly the K coefficients with the largest `|X[j]|`,
//! since any other selection of K bins leaves a strictly larger or equal residual
//! energy. □
//!
//! **Reference.** Candès & Wakin (2008), "An Introduction to Compressive Sensing",
//! IEEE Signal Processing Magazine 25(2), pp. 21–30.
//!
//! ## Theorem: Exact Recovery for K-Sparse Signals
//!
//! **Statement.** If x ∈ ℂ^N is exactly K-sparse in the DFT domain (at most K
//! frequency components are nonzero), then `SparseFftPlan::new(N, K)?.forward(x)`
//! returns a `SparseSpectrum` containing exactly those K nonzero components, with
//! values matching `FFT(x)[k]` to within FFT numerical precision
//! (O(N log N · ε_machine)).
//!
//! **Proof sketch.** In exact arithmetic the N-K non-support DFT bins are zero.
//! The top-K heap selector retains all K nonzero bins because their squared
//! magnitudes are strictly positive while the remaining N-K bins have squared
//! magnitude zero; no nonzero bin can be displaced by a zero bin. The threshold
//! filter (default `threshold = 0.0`) passes every retained bin since each has
//! norm > 0. The only error is accumulated floating-point rounding in the
//! O(N log N) butterfly network, bounding per-coefficient error at
use apollo_fft::{ApolloError, ApolloResult};

#[cfg(test)]
use crate::domain::spectrum::sparse::SparseSpectrum;
#[cfg(test)]
use apollo_fft::{CpuStorage, F16};
#[cfg(test)]
use eunomia::{Complex32, Complex64};

#[inline]
pub(crate) fn validate_transform_len(expected: usize, actual: usize) -> ApolloResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(ApolloError::ShapeMismatch {
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }
}

#[inline]
pub(crate) fn validate_support_len(frequencies_len: usize, values_len: usize) -> ApolloResult<()> {
    if frequencies_len == values_len {
        Ok(())
    } else {
        Err(ApolloError::validation(
            "sparse_values",
            values_len.to_string(),
            "frequency and value counts must match",
        ))
    }
}
mod plan;
mod storage;

#[cfg(test)]
mod tests;

pub use plan::{SparseFftPlan, SparseLetoSpectrum};
pub use storage::SparseComplexStorage;
