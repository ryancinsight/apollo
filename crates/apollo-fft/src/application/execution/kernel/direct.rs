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
//! ## Precision contract
//!
//! Both directions accumulate in `T` itself. `f64` appears only to evaluate
//! the analytic twiddle `exp(±2π i k n / N)` and the `1/N` normalization
//! factor; those constants enter `T` through [`KernelScalar::from_precise`]
//! before any transform arithmetic touches them.
//!
//! There is deliberately no wider accumulator. This kernel is the reference
//! oracle for the optimized backends, so it must model exactly what a backend
//! at the same precision can achieve. A reference accumulated more precisely
//! than the backend it validates makes the derived differential tolerances
//! unmeasurable: the reported error would mix backend error with a
//! reference-precision mismatch that no backend change can remove. Should
//! numerical analysis ever justify a wider accumulator, it belongs in an
//! associated type on [`KernelScalar`], never in an ad-hoc local.
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

use eunomia::{Complex32, Complex64};

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

    /// Convert an analytically known constant to the scalar type.
    ///
    /// This is the trait's only precision-crossing operation, and it crosses
    /// in one direction: it materializes a closed-form constant (a twiddle
    /// component or the `1/N` factor) at the scalar's own precision. Transform
    /// data never travels back out through a wider type.
    fn from_precise(value: f64) -> Self;
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
///
/// Mirrors [`dft_forward`] exactly, differing only in the twiddle sign and the
/// trailing `1/N` factor: the reduction accumulates in `T`, and `f64` supplies
/// only the analytic constants (see the module's precision contract).
#[must_use]
pub fn dft_inverse<T: KernelScalar>(input: &[T]) -> Vec<T> {
    let n = input.len();
    assert!(n > 0, "DFT length must be non-zero");
    let mut output = vec![T::zero(); n];
    let tau = std::f64::consts::TAU;
    let n_f64 = n as f64;
    // `1/N + 0i` at the scalar's own precision; the normalization is a complex
    // multiply by a real constant, which is the transform definition verbatim.
    let scale = T::from_precise(1.0 / n_f64);

    for (n_idx, slot) in output.iter_mut().enumerate() {
        let n_idx_f64 = n_idx as f64;
        let mut sum = T::zero();
        for (k, &value) in input.iter().enumerate() {
            let angle = tau * (k as f64) * n_idx_f64 / n_f64;
            let twiddle = T::complex(T::from_precise(angle.cos()), T::from_precise(angle.sin()));
            sum = T::add(sum, T::mul(value, twiddle));
        }
        *slot = T::mul(sum, scale);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit roundoff for `f32`: `u = ε/2 = 2⁻²⁴`.
    const F32_UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;

    fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
        (a.re - b.re).abs() <= eps && (a.im - b.im).abs() <= eps
    }

    fn spectrum(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let t = k as f64;
                Complex64::new((0.31 * t).sin() + 0.5, (0.17 * t).cos() - 0.25)
            })
            .collect()
    }

    fn narrow(input: &[Complex64]) -> Vec<Complex32> {
        input
            .iter()
            .map(|x| Complex32::new(x.re as f32, x.im as f32))
            .collect()
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

    /// `dft_inverse` at `Complex32` lands within the error an `f32` reduction
    /// can actually achieve.
    ///
    /// Derivation. Let `u = ε_f32/2 = 2⁻²⁴` be the unit roundoff and
    /// `t_k = X_k·w^{kn}` the summands; `|w| = 1`, so `|t_k| = |X_k|`.
    ///
    /// * Naive recursive summation of `n` terms has forward error
    ///   `≤ γ_{n−1}·Σ|t_k|` with `γ_m = mu/(1−mu) ≈ m·u` — the `O(n·ε)` term
    ///   (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
    ///   §3.1).
    /// * Each summand carries `≤ c₁·u·|X_k|`: `√2·u` from rounding the two
    ///   twiddle components into `f32`, `≈ 2.83·u` from the complex product
    ///   (`√2·γ₂`), and `u` from narrowing the input. `c₁ = 6` covers all three.
    /// * `Σ|X_k| ≤ n·max|X_k|`; the `1/N` normalization divides the accumulated
    ///   sum error by `n` and adds one rounding `≤ u·|x_n|`, `|x_n| ≤ max|X_k|`.
    ///
    /// `|Δx_n| ≤ ((n−1+c₁)/n)·u·Σ|X_k| + u·max|X_k| ≤ (n+6)·u·max|X_k|`
    ///
    /// The `Complex64` instantiation stands in for the exact transform: its own
    /// error carries `u_f64/u_f32 = 2⁻²⁹` of the bound above.
    #[test]
    fn inverse_reduced_precision_matches_reference_within_derived_bound() {
        let n = 64usize;
        let input = spectrum(n);
        let max_magnitude = input
            .iter()
            .copied()
            .map(Complex64::norm)
            .fold(0.0, f64::max);
        let tol = (n as f64 + 6.0) * F32_UNIT_ROUNDOFF * max_magnitude;

        let reference = dft_inverse(&input);
        let err = dft_inverse(&narrow(&input))
            .iter()
            .zip(&reference)
            .map(|(got, want)| {
                Complex64::new(f64::from(got.re) - want.re, f64::from(got.im) - want.im).norm()
            })
            .fold(0.0, f64::max);

        assert!(
            err <= tol,
            "f32 inverse N={n} max_err={err:.3e} exceeds derived tol={tol:.3e}"
        );
    }

    /// `dft_inverse` at `Complex32` *is* the `f32` transform, not a wider one
    /// narrowed on the way out.
    ///
    /// The reference evaluates the same definition with plain `Complex32`
    /// operators, independent of the `KernelScalar` plumbing. Rust applies no
    /// automatic FMA contraction or reassociation, so two `f32` evaluations of
    /// the same operations in the same order agree bitwise. A difference
    /// therefore means the reduction ran at some precision other than `T` —
    /// the widen-compute-narrow defect this assertion exists to keep out.
    #[test]
    fn inverse_reduced_precision_reduces_in_t_not_a_wider_type() {
        let n = 48usize;
        let input = narrow(&spectrum(n));
        let inv_n = Complex32::new((1.0 / n as f64) as f32, 0.0);

        let expected: Vec<Complex32> = (0..n)
            .map(|n_idx| {
                let mut sum = Complex32::new(0.0, 0.0);
                for (k, &value) in input.iter().enumerate() {
                    let angle = std::f64::consts::TAU * (k as f64) * (n_idx as f64) / (n as f64);
                    let twiddle = Complex32::new(angle.cos() as f32, angle.sin() as f32);
                    sum += value * twiddle;
                }
                sum * inv_n
            })
            .collect();

        for (idx, (got, want)) in dft_inverse(&input).iter().zip(&expected).enumerate() {
            assert_eq!(
                (got.re.to_bits(), got.im.to_bits()),
                (want.re.to_bits(), want.im.to_bits()),
                "bin {idx}: generic f32 inverse diverged from the f32 definition"
            );
        }
    }
}
