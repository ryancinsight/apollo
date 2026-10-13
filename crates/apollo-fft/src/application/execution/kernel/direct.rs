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

use eunomia::{Complex, FloatElement};

/// Scalar interface required by the Apollo FFT kernel.
///
/// `zero` (`Default`), `add` (`+`), and `mul` (`*`) are covered by the
/// `Default + Add + Mul` supertraits so `KernelScalar` declares only the
/// two complex-structure operations that cannot be expressed generically.
pub trait KernelScalar:
    Copy + Clone + Default + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    /// Construct a complex value from real and imaginary parts.
    fn complex(re: Self, im: Self) -> Self;

    /// Convert an analytically known constant to the scalar type.
    ///
    /// This is the trait's only precision-crossing operation, and it crosses
    /// in one direction: it materializes a closed-form constant (a twiddle
    /// component or the `1/N` factor) at the scalar's own precision. Transform
    /// data never travels back out through a wider type.
    fn from_precise(value: f64) -> Self;
}

/// One implementation for every complex element the crate carries.
///
/// The arithmetic runs in `T` itself; `f64` evaluates only the analytic
/// constants, which `from_precise` narrows through [`FloatElement::from_f64`]
/// before any transform arithmetic touches them (see the module precision
/// contract above). There is deliberately no wider accumulator.
impl<T> KernelScalar for Complex<T>
where
    T: FloatElement,
{
    #[inline]
    fn complex(re: Self, im: Self) -> Self {
        Self::new(re.re, im.re)
    }

    #[inline]
    fn from_precise(value: f64) -> Self {
        Self::new(T::from_f64(value), T::ZERO)
    }
}

/// Direct DFT forward transform.
#[must_use]
pub fn dft_forward<T: KernelScalar>(input: &[T]) -> Vec<T> {
    let n = input.len();
    assert!(n > 0, "DFT length must be non-zero");
    let mut output = vec![T::default(); n];
    let tau = std::f64::consts::TAU;
    let n_f64 = n as f64;

    for (k, slot) in output.iter_mut().enumerate() {
        let k_f64 = k as f64;
        let mut sum = T::default();
        for (n_idx, &value) in input.iter().enumerate() {
            let angle = -tau * k_f64 * (n_idx as f64) / n_f64;
            let twiddle = T::complex(T::from_precise(angle.cos()), T::from_precise(angle.sin()));
            sum = sum + value * twiddle;
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
    let mut output = vec![T::default(); n];
    let tau = std::f64::consts::TAU;
    let n_f64 = n as f64;
    // `1/N + 0i` at the scalar's own precision; the normalization is a complex
    // multiply by a real constant, which is the transform definition verbatim.
    let scale = T::from_precise(1.0 / n_f64);

    for (n_idx, slot) in output.iter_mut().enumerate() {
        let n_idx_f64 = n_idx as f64;
        let mut sum = T::default();
        for (k, &value) in input.iter().enumerate() {
            let angle = tau * (k as f64) * n_idx_f64 / n_f64;
            let twiddle = T::complex(T::from_precise(angle.cos()), T::from_precise(angle.sin()));
            sum = sum + value * twiddle;
        }
        *slot = sum * scale;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::{Complex32, Complex64, NumericElement, F16};

    /// Unit roundoff for `f32`: `u = ε/2 = 2⁻²⁴`.
    const F32_UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;

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

    /// Worst-component error bound for a length-`n` direct transform of `x`.
    ///
    /// Each output is `n` products `x_j*w` summed sequentially in `T`: the
    /// twiddle rounds once, the complex product adds at most `2u`, and
    /// sequential summation adds `(n - 1)u` (Higham, *Accuracy and Stability
    /// of Numerical Algorithms*, 2nd ed., sections 3.1 and 4.2), so every
    /// component lies within `(n + 2)*u*norm` with `norm = sum |x|`. Both
    /// directions accumulate in `T` (see the module precision contract);
    /// the forward error propagates through the inverse unamplified and the
    /// inverse adds its own `(n + 2)*u*norm`, so with `eps = 2u` the
    /// `(n + 4)*eps*norm` bound covers a forward pass followed by an inverse.
    fn bound<T>(input: &[Complex<T>], epsilon: f64) -> f64
    where
        T: FloatElement,
    {
        let norm: f64 = input
            .iter()
            .map(|&z| {
                let re = z.re.to_f64();
                let im = z.im.to_f64();
                re.hypot(im)
            })
            .sum();
        (input.len() as f64 + 4.0) * epsilon * norm
    }

    fn assert_close<T>(actual: &[Complex<T>], expected: &[(f64, f64)], tolerance: f64)
    where
        T: FloatElement,
    {
        assert_eq!(actual.len(), expected.len());
        for (index, (&value, &(re, im))) in actual.iter().zip(expected).enumerate() {
            let got = (value.re.to_f64(), value.im.to_f64());
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
    {
        let input = signal::<T>(&[(1.0, 0.0), (2.0, 0.0)]);
        let tolerance = bound(&input, epsilon);
        assert_close(&dft_forward(&input), &[(3.0, 0.0), (-1.0, 0.0)], tolerance);
    }

    fn round_trip<T>(values: &[(f64, f64)], epsilon: f64)
    where
        T: FloatElement,
    {
        let input = signal::<T>(values);
        let tolerance = bound(&input, epsilon);
        let expected: Vec<(f64, f64)> = input
            .iter()
            .map(|&z| (z.re.to_f64(), z.im.to_f64()))
            .collect();
        assert_close(&dft_inverse(&dft_forward(&input)), &expected, tolerance);
    }

    const COMPLEX_SIGNAL: [(f64, f64); 4] = [(1.0, -1.0), (2.0, 0.5), (-0.5, 0.25), (0.75, -0.125)];
    const REAL_SIGNAL: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)];

    #[test]
    fn forward_matches_known_two_point_transform() {
        forward_two_point::<f64>(f64::EPSILON);
        forward_two_point::<f32>(f64::from(f32::EPSILON));
        forward_two_point::<F16>(F16::EPSILON.to_f64());
    }

    #[test]
    fn inverse_recovers_input() {
        round_trip::<f64>(&COMPLEX_SIGNAL, f64::EPSILON);
        round_trip::<f32>(&COMPLEX_SIGNAL, f64::from(f32::EPSILON));
        round_trip::<F16>(&COMPLEX_SIGNAL, F16::EPSILON.to_f64());
    }

    #[test]
    fn forward_inverse_is_identity_on_real_signal() {
        round_trip::<f64>(&REAL_SIGNAL, f64::EPSILON);
        round_trip::<f32>(&REAL_SIGNAL, f64::from(f32::EPSILON));
        round_trip::<F16>(&REAL_SIGNAL, F16::EPSILON.to_f64());
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
