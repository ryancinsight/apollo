//! Apollo-owned direct DFT kernel.
//!
//! This module provides the in-repo `O(N²)` discrete Fourier transform used as
//! the reference oracle for Apollo's fast kernels, without production
//! dependencies on external FFT engines.
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
//! ## Precision contract
//!
//! Both directions take `Complex<T>` for any [`FloatElement`] `T` and run
//! every product, sum, and the inverse's `1/N` normalization in
//! [`FloatElement::Accumulator`] — `T` itself for `f32` and `f64`, `f32` for
//! the reduced-precision formats. The input widens into the accumulator
//! exactly ([`FloatElement::to_accumulator`]); each output narrows into `T`
//! once ([`FloatElement::from_accumulator`]). Sums are sequential in the input
//! order, so each output component carries the recursive-summation error
//! `γ_{N−1}` in the accumulator's unit roundoff (Higham, *Accuracy and
//! Stability of Numerical Algorithms*, 2nd ed., §4.2).
//!
//! The twiddle `exp(±2π i j / N)` is a function of the two integers `j` and
//! `N` alone, so it is constant generation rather than computation on `T`
//! data: the phase index `j = k·n mod N` advances by integer addition, the
//! angle `2π j / N` (`|angle| < 2π`) and its sine and cosine are evaluated in
//! `f64` through eunomia's libm-backed [`FloatElement`] surface, and the
//! result rounds once into the accumulator — the correctly rounded twiddle up
//! to the `f64` evaluation error.
//!
//! ## Failure modes
//!
//! * zero-length transforms panic
//!
//! ## Complexity
//!
//! `O(N²)` time and `O(1)` auxiliary space beyond the output buffer.

use core::f64::consts::TAU;
use eunomia::{Complex, FloatElement, NumericElement};

/// Forward direct DFT, `X_k = Σ_n x_n · exp(-2π i k n / N)`.
///
/// Accumulates in `T::Accumulator` and rounds each output into `T` once
/// (module-level precision contract).
///
/// # Panics
///
/// Panics when `input` is empty.
///
/// # Examples
///
/// ```
/// use apollo_fft::application::execution::kernel::dft_forward;
/// use eunomia::Complex;
///
/// let spectrum = dft_forward(&[Complex::new(1.0_f64, 0.0), Complex::new(2.0, 0.0)]);
/// assert_eq!(spectrum[0], Complex::new(3.0, 0.0));
/// ```
#[must_use]
pub fn dft_forward<T: FloatElement>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    direct_dft::<T, false>(input)
}

/// Inverse direct DFT with `1/N` normalization,
/// `x_n = (1/N) Σ_k X_k · exp(2π i k n / N)`.
///
/// Accumulates and normalizes in `T::Accumulator` and rounds each output into
/// `T` once (module-level precision contract).
///
/// # Panics
///
/// Panics when `input` is empty.
///
/// # Examples
///
/// ```
/// use apollo_fft::application::execution::kernel::dft_inverse;
/// use eunomia::Complex;
///
/// let signal = dft_inverse(&[Complex::new(3.0_f64, 0.0), Complex::new(-1.0, 0.0)]);
/// assert_eq!(signal[0], Complex::new(1.0, 0.0));
/// ```
#[must_use]
pub fn dft_inverse<T: FloatElement>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    direct_dft::<T, true>(input)
}

/// The shared direct-summation body; `INVERSE` selects the twiddle sign and
/// the `1/N` normalization.
fn direct_dft<T: FloatElement, const INVERSE: bool>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    let n = input.len();
    assert!(n > 0, "DFT length must be non-zero");
    let length = T::Accumulator::from_count(n);

    (0..n)
        .map(|bin| {
            let zero = <T::Accumulator as NumericElement>::ZERO;
            let mut sum: Complex<T::Accumulator> = Complex::new(zero, zero);
            // `phase = bin·index mod n`; both terms stay below `n`, so the
            // addition cannot overflow for any slice length.
            let mut phase = 0;
            for &value in input {
                let term = Complex::new(value.re.to_accumulator(), value.im.to_accumulator());
                sum += term * twiddle::<T::Accumulator, INVERSE>(phase, n);
                phase += bin;
                if phase >= n {
                    phase -= n;
                }
            }
            let sum = if INVERSE { sum / length } else { sum };
            Complex::new(T::from_accumulator(sum.re), T::from_accumulator(sum.im))
        })
        .collect()
}

/// The root of unity `exp(∓2π i · phase / n)`, generated in `f64` from its two
/// integers and rounded once into `A`.
fn twiddle<A: FloatElement, const INVERSE: bool>(phase: usize, n: usize) -> Complex<A> {
    let turn = TAU * f64::from_count(phase) / f64::from_count(n);
    let angle = if INVERSE { turn } else { -turn };
    Complex::new(
        A::from_f64(FloatElement::cos(angle)),
        A::from_f64(FloatElement::sin(angle)),
    )
}

#[cfg(test)]
mod tests;
