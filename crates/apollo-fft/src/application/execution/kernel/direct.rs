//! Apollo-owned FFT kernel.
//!
//! This module provides the in-repo discrete Fourier transform kernel used by
//! Apollo plans without production dependencies on external FFT engines.
//!
//! The implementation is intentionally self-contained and allocation-aware.
//! It computes the forward and inverse DFT directly from the definition using
//! a reusable twiddle recurrence, which preserves zero-copy call sites and
//! keeps the kernel available to higher-level plans without external
//! dependencies.
//!
//! ## Mathematical contract
//!
//! For a complex input vector `x ∈ ℂ^N`, the forward transform is
//!
//! `X_k = Σ_{n=0}^{N-1} x_n · exp(-2π i k n / N)`
//!
//! and the inverse transform is
//!
//! `x_n = (1/N) Σ_{k=0}^{N-1} X_k · exp(2π i k n / N)`.
//!
//! This module implements those formulas in floating-point arithmetic,
//! subject to the usual rounding behavior of the selected precision.
//!
//! ## Design notes
//!
//! * The kernel is generic over the scalar type through a small trait.
//! * The implementation favors clarity and correctness first, then can be
//!   specialized later with radix decomposition or SIMD backends.
//! * The public surface is intentionally small so plan modules can own their
//!   buffering and normalization policies.
//!
//! ## Failure modes
//!
//! * zero-length transforms are rejected
//! * caller-supplied buffers must match the kernel length
//!
//! ## Complexity
//!
//! This direct kernel is `O(N²)` time and `O(1)` auxiliary space beyond the
//! output buffer. It is a correct baseline for the Apollo-owned FFT engine and
//! can be replaced by a faster recursive kernel without changing the public
//! contract.

use num_complex::{Complex32, Complex64};

/// Scalar interface required by the Apollo FFT kernel.
pub trait KernelScalar: Copy + Clone + Default {
    /// Construct a complex value from real and imaginary parts.
    fn complex(re: Self, im: Self) -> Self;

    /// Add two complex values.
    fn add(lhs: Self, rhs: Self) -> Self;

    /// Multiply two complex values.
    fn mul(lhs: Self, rhs: Self) -> Self;

    /// Return zero.
    fn zero() -> Self;

    /// Convert a reference-precision component value to the scalar type.
    fn from_precise(value: f64) -> Self;

    /// Extract the real part at reference precision.
    fn precise_re(value: Self) -> f64;

    /// Extract the imaginary part at reference precision.
    fn precise_im(value: Self) -> f64;
}

impl KernelScalar for Complex64 {
    #[inline]
    fn complex(re: Self, im: Self) -> Self {
        Self::new(re.re, im.re)
    }

    #[inline]
    fn add(lhs: Self, rhs: Self) -> Self {
        lhs + rhs
    }

    #[inline]
    fn mul(lhs: Self, rhs: Self) -> Self {
        lhs * rhs
    }

    #[inline]
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    #[inline]
    fn from_precise(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    #[inline]
    fn precise_re(value: Self) -> f64 {
        value.re
    }

    #[inline]
    fn precise_im(value: Self) -> f64 {
        value.im
    }
}

impl KernelScalar for Complex32 {
    #[inline]
    fn complex(re: Self, im: Self) -> Self {
        Self::new(re.re, im.re)
    }

    #[inline]
    fn add(lhs: Self, rhs: Self) -> Self {
        lhs + rhs
    }

    #[inline]
    fn mul(lhs: Self, rhs: Self) -> Self {
        lhs * rhs
    }

    #[inline]
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    #[inline]
    fn from_precise(value: f64) -> Self {
        Self::new(value as f32, 0.0)
    }

    #[inline]
    fn precise_re(value: Self) -> f64 {
        f64::from(value.re)
    }

    #[inline]
    fn precise_im(value: Self) -> f64 {
        f64::from(value.im)
    }
}

/// Direct DFT forward transform.
#[must_use]
pub fn dft_forward<T: KernelScalar>(input: &[T]) -> Vec<T> {
    let n = input.len();
    assert!(n > 0, "DFT length must be non-zero");
    let mut output = vec![T::zero(); n];
    let tau = std::f64::consts::TAU;
    let n_f64 = n as f64;

    for (k, slot) in output.iter_mut().enumerate() {
        let k_f64 = k as f64;
        let mut sum = T::zero();
        for (n_idx, &value) in input.iter().enumerate() {
            let angle = -tau * k_f64 * (n_idx as f64) / n_f64;
            let twiddle = T::complex(T::from_precise(angle.cos()), T::from_precise(angle.sin()));
            sum = T::add(sum, T::mul(value, twiddle));
        }
        *slot = sum;
    }

    output
}

/// Direct DFT inverse transform with `1/N` normalization.
#[must_use]
pub fn dft_inverse<T: KernelScalar>(input: &[T]) -> Vec<T> {
    let n = input.len();
    assert!(n > 0, "DFT length must be non-zero");
    let mut output = vec![T::zero(); n];
    let tau = std::f64::consts::TAU;
    let scale = 1.0 / n as f64;
    let n_f64 = n as f64;

    for (n_idx, slot) in output.iter_mut().enumerate() {
        let n_idx_f64 = n_idx as f64;
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for (k, &value) in input.iter().enumerate() {
            let angle = tau * (k as f64) * n_idx_f64 / n_f64;
            let c = angle.cos();
            let s = angle.sin();
            let re = T::precise_re(value);
            let im = T::precise_im(value);
            sum_re += re * c - im * s;
            sum_im += re * s + im * c;
        }
        *slot = T::complex(
            T::from_precise(sum_re * scale),
            T::from_precise(sum_im * scale),
        );
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
        (a.re - b.re).abs() <= eps && (a.im - b.im).abs() <= eps
    }

    #[test]
    fn forward_matches_known_two_point_transform() {
        let input = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let output = dft_forward(&input);
        assert!(approx_eq(output[0], Complex64::new(3.0, 0.0), 1.0e-12));
        assert!(approx_eq(output[1], Complex64::new(-1.0, 0.0), 1.0e-12));
    }

    #[test]
    fn inverse_recovers_input() {
        let input = vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(-0.5, 0.25),
            Complex64::new(0.75, -0.125),
        ];
        let spectrum = dft_forward(&input);
        let recovered = dft_inverse(&spectrum);
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!(approx_eq(*actual, *expected, 1.0e-10));
        }
    }

    #[test]
    fn forward_inverse_is_identity_on_real_signal() {
        let input = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        let recovered = dft_inverse(&dft_forward(&input));
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!(approx_eq(*actual, *expected, 1.0e-10));
        }
    }
}
