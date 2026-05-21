//! Shared test utilities for FFT kernel testing.
//!
//! Centralizes common assertion and comparison functions used across
//! radix and Rader kernel tests to reduce DRY violations and
//! maintain consistent error tolerances.

use num_complex::{Complex32, Complex64};

/// Compute the maximum absolute error between two f64 FFT outputs.
///
/// Returns the largest element-wise norm distance: `max_k |output_k - expected_k|`.
/// Used for validating forward/inverse transforms against direct DFT reference.
#[cfg(test)]
#[inline]
pub(crate) fn max_abs_err_64(got: &[Complex64], expected: &[Complex64]) -> f64 {
    got.iter()
        .zip(expected.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

/// Compute the maximum absolute error between two f32 FFT outputs.
///
/// Returns the largest element-wise norm distance: `max_k |output_k - expected_k|`.
/// Used for validating forward/inverse transforms against direct DFT reference.
#[cfg(test)]
#[inline]
pub(crate) fn max_abs_err_32(got: &[Complex32], expected: &[Complex32]) -> f32 {
    got.iter()
        .zip(expected.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f32::max)
}
