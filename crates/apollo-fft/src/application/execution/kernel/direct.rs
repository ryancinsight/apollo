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

use eunomia::{CastFrom, Complex, FloatElement};

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

/// One implementation for every complex element the crate carries.
///
/// The arithmetic runs in `T` itself; only the angle and the inverse's
/// accumulator are held in `f64`, which is the reference precision the trait
/// contract names. `from_precise` narrows through [`FloatElement::from_f64`]
/// and `precise_re`/`precise_im` widen exactly through [`CastFrom`].
impl<T> KernelScalar for Complex<T>
where
    T: FloatElement,
    f64: CastFrom<T>,
{
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
        Self::new(T::ZERO, T::ZERO)
    }

    #[inline]
    fn from_precise(value: f64) -> Self {
        Self::new(T::from_f64(value), T::ZERO)
    }

    #[inline]
    fn precise_re(value: Self) -> f64 {
        f64::cast_from(value.re)
    }

    #[inline]
    fn precise_im(value: Self) -> f64 {
        f64::cast_from(value.im)
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
    use eunomia::F16;

    /// Worst-component error bound for a length-`n` direct transform of `x`.
    ///
    /// Each forward output is `n` products `x_j·w` summed sequentially in `T`:
    /// the twiddle rounds once, the complex product adds at most `2u`, and
    /// sequential summation adds `(n − 1)u` (Higham, *Accuracy and Stability
    /// of Numerical Algorithms*, 2nd ed., §3.1 and §4.2), so every component
    /// lies within `(n + 2)·u·‖x‖₁`. The inverse accumulates in `f64` and
    /// rounds once into `T`, adding `u·‖X‖∞ ≤ u·‖x‖₁`. With `ε = 2u`,
    /// `(n + 4)·ε·‖x‖₁` covers a forward pass followed by an inverse.
    fn bound<T>(input: &[Complex<T>], epsilon: f64) -> f64
    where
        T: FloatElement,
        f64: CastFrom<T>,
    {
        let norm: f64 = input
            .iter()
            .map(|&z| {
                let re = Complex::<T>::precise_re(z);
                let im = Complex::<T>::precise_im(z);
                re.hypot(im)
            })
            .sum();
        (input.len() as f64 + 4.0) * epsilon * norm
    }

    fn assert_close<T>(actual: &[Complex<T>], expected: &[(f64, f64)], tolerance: f64)
    where
        T: FloatElement,
        f64: CastFrom<T>,
    {
        assert_eq!(actual.len(), expected.len());
        for (index, (&value, &(re, im))) in actual.iter().zip(expected).enumerate() {
            let got = (
                Complex::<T>::precise_re(value),
                Complex::<T>::precise_im(value),
            );
            assert!(
                (got.0 - re).abs() <= tolerance && (got.1 - im).abs() <= tolerance,
                "bin {index}: {got:?} differs from ({re}, {im}) by more than {tolerance}"
            );
        }
    }

    fn signal<T: FloatElement>(values: &[(f64, f64)]) -> Vec<Complex<T>> {
        values
            .iter()
            .map(|&(re, im)| Complex::new(T::from_f64(re), T::from_f64(im)))
            .collect()
    }

    fn forward_two_point<T>(epsilon: f64)
    where
        T: FloatElement,
        f64: CastFrom<T>,
    {
        let input = signal::<T>(&[(1.0, 0.0), (2.0, 0.0)]);
        let tolerance = bound(&input, epsilon);
        assert_close(&dft_forward(&input), &[(3.0, 0.0), (-1.0, 0.0)], tolerance);
    }

    fn round_trip<T>(values: &[(f64, f64)], epsilon: f64)
    where
        T: FloatElement,
        f64: CastFrom<T>,
    {
        let input = signal::<T>(values);
        let tolerance = bound(&input, epsilon);
        let expected: Vec<(f64, f64)> = input
            .iter()
            .map(|&z| (Complex::<T>::precise_re(z), Complex::<T>::precise_im(z)))
            .collect();
        assert_close(&dft_inverse(&dft_forward(&input)), &expected, tolerance);
    }

    const COMPLEX_SIGNAL: [(f64, f64); 4] = [(1.0, -1.0), (2.0, 0.5), (-0.5, 0.25), (0.75, -0.125)];
    const REAL_SIGNAL: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)];

    #[test]
    fn forward_matches_known_two_point_transform() {
        forward_two_point::<f64>(f64::EPSILON);
        forward_two_point::<f32>(f64::from(f32::EPSILON));
        forward_two_point::<F16>(f64::cast_from(F16::EPSILON));
    }

    #[test]
    fn inverse_recovers_input() {
        round_trip::<f64>(&COMPLEX_SIGNAL, f64::EPSILON);
        round_trip::<f32>(&COMPLEX_SIGNAL, f64::from(f32::EPSILON));
        round_trip::<F16>(&COMPLEX_SIGNAL, f64::cast_from(F16::EPSILON));
    }

    #[test]
    fn forward_inverse_is_identity_on_real_signal() {
        round_trip::<f64>(&REAL_SIGNAL, f64::EPSILON);
        round_trip::<f32>(&REAL_SIGNAL, f64::from(f32::EPSILON));
        round_trip::<F16>(&REAL_SIGNAL, f64::cast_from(F16::EPSILON));
    }
}
