//! 1D Type-1 and Type-2 NUFFT executions and plans.
//!
//! This module implements a Kaiser-Bessel gridding NUFFT backed by Apollo's
//! in-repo FFT plan rather than external FFT crates.
//!
//! The exact direct transforms serve as the mathematical reference surface,
//! while the fast paths use oversampled grid spreading followed by Apollo FFT
//! execution and deconvolution.
//!
//! # Mathematical contract
//!
//! Type-1 NUFFT maps non-uniform samples `(x_j, c_j)` to uniform Fourier bins
//!
//! `f_k = Σ_j c_j exp(-2πi k x_j / L)`
//!
//! Type-2 NUFFT maps uniform Fourier bins `f_k` to non-uniform positions
//!
//! `g_j = Σ_k f_k exp(2πi k x_j / L)`
//!
//! The fast paths approximate the direct sums by spreading to an oversampled
//! grid with a Kaiser-Bessel kernel and then applying an FFT on the grid.
//!
//! # Complexity
//!
//! Direct transforms are `O(MN)` where `M` is the number of target bins and
//! `N` is the number of samples or coefficients.
//! Fast transforms are `O(Nw + m log m)` where `w` is the kernel width and
//! `m = σn` is the oversampled grid length.
//!
//! # Failure modes
//!
//! - positions/value arrays must have equal length
//! - oversampling factor must satisfy `sigma >= 2`
//! - kernel width must satisfy `kernel_width >= 2`
#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet APOLLO-UNWRAP-1: pre-existing debt"
    )
)]

use leto::Array1;

#[cfg(test)]
use crate::domain::metadata::grid::UniformDomain1D;
#[cfg(test)]
use crate::infrastructure::kernel::kaiser_bessel::fft_signed_index;
#[cfg(test)]
use crate::DEFAULT_NUFFT_OVERSAMPLING;
#[cfg(test)]
use apollo_fft::{ApolloError, CpuStorage, PrecisionProfile, F16};
#[cfg(test)]
use eunomia::{Complex32, Complex64};
#[cfg(test)]
use std::f64::consts::PI;

/// Below this operation count, serial loops avoid parallel scheduling overhead.
pub(crate) const DIRECT_PAR_OP_THRESHOLD: usize = 16_384;

pub(crate) fn array1_from_vec<T>(values: Vec<T>) -> Array1<T> {
    Array1::from_shape_vec([values.len()], values)
        .expect("invariant: vector length matches 1D Leto shape")
}

pub(crate) fn array1_value<T: Copy>(array: &Array1<T>, index: usize) -> T {
    *array
        .get([index])
        .expect("invariant: validated rank-1 Leto index")
}

mod bucket;
mod execute;
mod plan;
mod storage;

#[cfg(test)]
mod proptest_suite;
#[cfg(test)]
mod tests;

pub(crate) use execute::write_typed_output;
pub use execute::{nufft_type1_1d, nufft_type1_1d_fast, nufft_type2_1d, nufft_type2_1d_fast};
pub use plan::NufftPlan1D;
pub use storage::NufftComplexStorage;
