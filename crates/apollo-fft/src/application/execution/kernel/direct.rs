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
mod tests {
    use super::*;
    use core::f64::consts::SQRT_2;
    use eunomia::F16;

    /// Unit roundoffs of one instantiation: `accumulator` for every product
    /// and sum, `storage` for the single rounding of each output into `T`.
    #[derive(Clone, Copy)]
    struct Roundoff {
        accumulator: f64,
        storage: f64,
    }

    const F64_ROUNDOFF: Roundoff = Roundoff {
        accumulator: f64::EPSILON / 2.0,
        storage: f64::EPSILON / 2.0,
    };

    fn f32_roundoff() -> Roundoff {
        let unit = f64::from(f32::EPSILON) / 2.0;
        Roundoff {
            accumulator: unit,
            storage: unit,
        }
    }

    /// `F16` accumulates in `f32` (`FloatElement::Accumulator`) and stores at
    /// its own 11-bit precision.
    fn f16_roundoff() -> Roundoff {
        Roundoff {
            accumulator: f64::from(f32::EPSILON) / 2.0,
            storage: F16::EPSILON.to_f64() / 2.0,
        }
    }

    /// `γ_k = k·u / (1 − k·u)` (Higham, 2nd ed., §3.1, Lemma 3.1).
    fn gamma(k: usize, unit: f64) -> f64 {
        let ku = f64::from_count(k) * unit;
        ku / (1.0 - ku)
    }

    /// Per-component bound `τ` on the twiddle error `|ŵ − w|`.
    ///
    /// The `f64` angle `fl(fl(τ₆₄·j)/N)` carries the rounding of the `2π`
    /// constant, the product, and the quotient (`j`, `N` convert exactly
    /// below `2^53`), so it is within `γ₃(u₆₄)·|angle| < 2π·γ₃(u₆₄)` of the
    /// exact angle, and sine and cosine are 1-Lipschitz. The libm sine and
    /// cosine are held to 1 ulp of the MPFR result by the libm test harness
    /// (rust-lang/compiler-builtins, `libm-test/src/precision.rs`,
    /// `default_ulp`), at most `2u₆₄` for a value in `[−1, 1]`. Rounding into
    /// the accumulator adds `u_A` relative to that value.
    fn twiddle_error(roundoff: Roundoff) -> f64 {
        let reference = f64::EPSILON / 2.0;
        (TAU * gamma(3, reference) + 2.0 * reference) * (1.0 + roundoff.accumulator)
            + roundoff.accumulator
    }

    /// Which pass a bound covers: the inverse adds one rounding for `/N`.
    #[derive(Clone, Copy)]
    enum Pass {
        Forward,
        Inverse,
    }

    /// Per-component accumulator error of one pass over input of modulus mass
    /// `mass` (`Σ |x_j|` forward, `Σ |X_k| / N` inverse).
    ///
    /// Each term is `x_j·ŵ_j` with `|ŵ_j − w_j| ≤ √2·τ` and `|ŵ_j| ≤ ω =
    /// 1 + √2·τ`; the complex product errs by `√2·γ₂·|x_j||ŵ_j|` (Higham,
    /// 2nd ed., §3.6, Lemma 3.5) and sequential summation of the `N` terms
    /// by `γ_{N−1}·Σ|terms|` (§4.2), which combine to `γ_{N+2}·ω·mass`
    /// through `√2·γ₂ ≤ γ₃` and `γ_j + γ_k + γ_j·γ_k ≤ γ_{j+k}` (§3.1,
    /// Lemma 3.3). The inverse's division by `N` adds one rounding, giving
    /// `γ_{N+3}` and a `(1 + u_A)` factor on the twiddle term.
    fn pass_error(n: usize, roundoff: Roundoff, mass: f64, pass: Pass) -> f64 {
        let tau = twiddle_error(roundoff);
        let twiddle_modulus = 1.0 + SQRT_2 * tau;
        let (terms, twiddle_term) = match pass {
            Pass::Forward => (n + 2, SQRT_2 * tau),
            Pass::Inverse => (n + 3, SQRT_2 * tau * (1.0 + roundoff.accumulator)),
        };
        (twiddle_term + gamma(terms, roundoff.accumulator) * twiddle_modulus) * mass
    }

    /// Forward bound: the accumulator error plus one storage rounding of a
    /// component no larger than `mass + error`.
    fn forward_bound(n: usize, roundoff: Roundoff, mass: f64) -> f64 {
        let error = pass_error(n, roundoff, mass, Pass::Forward);
        error + roundoff.storage * (mass + error)
    }

    /// Round-trip bound for `inverse(forward(x))` against `x`.
    ///
    /// The stored spectrum `X̂` differs from `X` by at most `b_f` per
    /// component, `√2·b_f` in modulus, so the exact inverse of `X̂` lies within
    /// `√2·b_f` of `x` and `Σ|X̂_k| / N ≤ ‖x‖₁ + √2·b_f` (`|X_k| ≤ ‖x‖₁`). The
    /// inverse pass adds its accumulator error over that mass, and the final
    /// storage rounding acts on a component no larger than `‖x‖₁` plus both.
    fn round_trip_bound(n: usize, roundoff: Roundoff, mass: f64) -> f64 {
        let spectrum = SQRT_2 * forward_bound(n, roundoff, mass);
        let error = spectrum + pass_error(n, roundoff, mass + spectrum, Pass::Inverse);
        error + roundoff.storage * (mass + error)
    }

    /// `Σ |x_j|` evaluated in `f64` from the exactly widened components.
    fn modulus_mass<T: FloatElement>(input: &[Complex<T>]) -> f64 {
        input
            .iter()
            .map(|z| z.re.to_f64().hypot(z.im.to_f64()))
            .sum()
    }

    fn assert_close<T>(actual: &[Complex<T>], expected: &[(f64, f64)], tolerance: f64)
    where
        T: FloatElement,
    {
        assert_eq!(actual.len(), expected.len());
        for (index, (value, &(re, im))) in actual.iter().zip(expected).enumerate() {
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

    fn forward_two_point<T: FloatElement>(roundoff: Roundoff) {
        let input = signal::<T>(&[(1.0, 0.0), (2.0, 0.0)]);
        let tolerance = forward_bound(input.len(), roundoff, modulus_mass(&input));
        assert_close(&dft_forward(&input), &[(3.0, 0.0), (-1.0, 0.0)], tolerance);
    }

    fn round_trip<T: FloatElement>(values: &[(f64, f64)], roundoff: Roundoff) {
        let input = signal::<T>(values);
        let tolerance = round_trip_bound(input.len(), roundoff, modulus_mass(&input));
        let expected: Vec<(f64, f64)> = input
            .iter()
            .map(|z| (z.re.to_f64(), z.im.to_f64()))
            .collect();
        assert_close(&dft_inverse(&dft_forward(&input)), &expected, tolerance);
    }

    /// Bin 0 of either direction on `[L, 1, 1, 1]` against exact constants.
    ///
    /// Output 0 has phase 0 at every term, so each twiddle is exactly
    /// `1 ± 0i` and the output is the plain sequential sum `((L + 1) + 1) + 1`
    /// in the accumulator (divided by the exact power of two `4` for the
    /// inverse), rounded once into `T`. With `L = 2^p` for the accumulator's
    /// significand precision `p`, each `+ 1` is a tie that rounds back to
    /// even, so the sum stagnates at `L` exactly when, and only when, the
    /// accumulator is `p` bits wide: a narrower accumulator stagnates at a
    /// smaller `L`, and a wider one carries `L + 3`.
    fn sums_in_the_accumulator<T: FloatElement>(leading: f64, forward: f64, inverse: f64) {
        let input = signal::<T>(&[(leading, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]);
        assert_eq!(
            dft_forward(&input)[0].re.to_f64().to_bits(),
            forward.to_bits()
        );
        assert_eq!(
            dft_inverse(&input)[0].re.to_f64().to_bits(),
            inverse.to_bits()
        );
    }

    const COMPLEX_SIGNAL: [(f64, f64); 4] = [(1.0, -1.0), (2.0, 0.5), (-0.5, 0.25), (0.75, -0.125)];
    const REAL_SIGNAL: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)];

    #[test]
    fn forward_matches_known_two_point_transform() {
        forward_two_point::<f64>(F64_ROUNDOFF);
        forward_two_point::<f32>(f32_roundoff());
        forward_two_point::<F16>(f16_roundoff());
    }

    #[test]
    fn inverse_recovers_input() {
        round_trip::<f64>(&COMPLEX_SIGNAL, F64_ROUNDOFF);
        round_trip::<f32>(&COMPLEX_SIGNAL, f32_roundoff());
        round_trip::<F16>(&COMPLEX_SIGNAL, f16_roundoff());
    }

    #[test]
    fn forward_inverse_is_identity_on_real_signal() {
        round_trip::<f64>(&REAL_SIGNAL, F64_ROUNDOFF);
        round_trip::<f32>(&REAL_SIGNAL, f32_roundoff());
        round_trip::<F16>(&REAL_SIGNAL, f16_roundoff());
    }

    #[test]
    fn sums_stagnate_at_the_accumulator_precision() {
        // f64 accumulates in f64 (p = 53): 2^53 + 1 ties back to 2^53.
        sums_in_the_accumulator::<f64>(
            9_007_199_254_740_992.0,
            9_007_199_254_740_992.0,
            2_251_799_813_685_248.0,
        );
        // f32 accumulates in f32 (p = 24): the sum stays 2^24 and the inverse
        // is 2^22. An f64 accumulator would carry 2^24 + 3, which rounds to
        // 2^24 + 4 in f32, and 2^22 + 0.75, which rounds to 2^22 + 1.
        sums_in_the_accumulator::<f32>(16_777_216.0, 16_777_216.0, 4_194_304.0);
        // F16 accumulates in f32: 2^11 + 3 = 2051 is exact there and rounds
        // to even at F16's spacing 2, giving 2052; 2051 / 4 = 512.75 rounds
        // to even at spacing 0.5, giving 513. An F16 accumulator would
        // stagnate at 2048 and 512.
        sums_in_the_accumulator::<F16>(2048.0, 2052.0, 513.0);
    }
}
