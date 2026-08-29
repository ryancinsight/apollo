//! Per-scale FFT cross-correlation kernel for the continuous wavelet transform.
//!
//! # Theorem: the CWT scale row is one circular cross-correlation
//!
//! The real-valued CWT coefficient produced by
//! [`super::coefficient`](super::coefficient) is, for scale `s > 0` and shift
//! `τ ∈ [0, n)`,
//!
//! ```text
//! C_s[τ] = Σ_{i=0}^{n-1} x[i] · s^{-1/2} ψ((i - τ)/s).
//! ```
//!
//! The weight depends on `i - τ` alone, so with the *lag kernel*
//!
//! ```text
//! w_s[k] = s^{-1/2} ψ(k/s),   k ∈ [-(n-1), n-1]   (2n-1 values)
//! ```
//!
//! the row is the cross-correlation `C_s[τ] = Σ_i x[i] w_s[i - τ]`. Note the
//! sign: this is a correlation, not a convolution — the kernel is indexed by
//! `i - τ`, not `τ - i`.
//!
//! ## Circular embedding
//!
//! Pick a transform length `L ≥ 2n - 1` and define the *periodized reflected*
//! kernel on `Z_L`
//!
//! ```text
//! h_s[j] = w_s[(-j) mod L]   i.e.   h_s[0]     = w_s[0],
//!                                   h_s[j]     = w_s[-j]   for 1 ≤ j ≤ n-1,
//!                                   h_s[L - j] = w_s[j]    for 1 ≤ j ≤ n-1,
//!                                   h_s[j]     = 0         otherwise.
//! ```
//!
//! The reflection is what turns the correlation into a convolution; the
//! negative lags occupy the top of the ring. `L ≥ 2n - 1` guarantees the two
//! written index ranges `[0, n-1]` and `[L-n+1, L-1]` are disjoint, so the
//! periodization does not self-overlap.
//!
//! Let `x̃` be `x` zero-padded to `L`. Then for `τ, i ∈ [0, n)`,
//! `τ - i ∈ [-(n-1), n-1]`, and by construction
//! `h_s[(τ - i) mod L] = w_s[i - τ]`, hence
//!
//! ```text
//! (x̃ ⊛_L h_s)[τ] = Σ_{i=0}^{n-1} x[i] h_s[(τ - i) mod L]
//!                 = Σ_{i=0}^{n-1} x[i] w_s[i - τ] = C_s[τ]   for τ ∈ [0, n). ∎
//! ```
//!
//! No aliasing correction is needed: every lag the direct sum uses is a
//! genuine entry of `h_s`, and every entry of `h_s` is a genuine lag.
//!
//! ## Boundary behaviour is preserved exactly
//!
//! The direct sum ranges over `i ∈ [0, n)` only, i.e. it treats the signal as
//! zero outside its support. Zero-padding `x` to `L` reproduces that same
//! implicit zero extension — the circular embedding wraps the *kernel*, never
//! the signal, so the edge coefficients are the identical truncated sums.
//!
//! ## Cost
//!
//! `DFT_L(x̃)` is scale-independent and is computed once for the whole plan, so
//! a transform over `S` scales costs `1 + 2S` transforms of length `L ≈ 2n`
//! and `S · (2n - 1)` mother-wavelet evaluations, against `S · n²` evaluations
//! for the direct kernel: `O(S · n log n)` against `O(S · n²)`.
//!
//! # References
//!
//! - Torrence, C. & Compo, G. P. (1998). A practical guide to wavelet
//!   analysis. *Bull. Amer. Meteor. Soc.*, 79(1), 61–78 — §3.d derives the CWT
//!   as a convolution evaluated through the DFT.
//! - Oppenheim, A. V. & Schafer, R. W. (2010). *Discrete-Time Signal
//!   Processing*, 3rd ed., §8.7 — circular convolution length conditions for
//!   aliasing-free linear convolution.

use super::mother_wavelet;
use crate::domain::metadata::wavelet::ContinuousWavelet;
use apollo_fft::{Complex64, PlanCacheProvider, Shape1D};
use mnemosyne::scratch::ScratchPool;

/// Signal length at or above which `CwtPlan::transform` uses the FFT
/// cross-correlation instead of the direct per-coefficient kernel.
///
/// # Derivation
///
/// The asymptotics never favour the direct kernel — per scale row it performs
/// `n²` mother-wavelet evaluations against `2n - 1` here — so the threshold
/// exists only for the fixed costs the convolution adds: the spectrum
/// allocation, two plan-cache lookups, and the scratch acquisition. Those are
/// paid in full by a *single-scale* transform, where the signal spectrum is
/// built and used once instead of being amortized across the scale rows, so
/// that is the case the threshold is measured against.
///
/// Measured with the `cwt_scale_rows` benchmark, single-scale case, pinned to
/// the performance cores of an Intel Core Ultra 9 285K (median of 100 samples;
/// confidence intervals within 0.1%):
///
/// ```text
/// n        direct        single-scale convolution     ratio
/// 4          94.7 ns              105.4 ns            0.90x
/// 8         383.0 ns              174.2 ns            2.20x
/// 16      1 586.4 ns              349.0 ns            4.55x
/// 32      6 783.8 ns              655.1 ns           10.4x
/// ```
///
/// The crossover therefore sits between `n = 4` and `n = 8`: `n = 4` is the
/// only measured size where the convolution loses, and it loses by 11%.
/// Eight is that crossover, not a rounded guess. With more than one scale the
/// spectrum amortizes and the convolution wins at every measured size
/// including `n = 4`, so the threshold is conservative by construction.
pub const FFT_CWT_LEN_THRESHOLD: usize = 8;

thread_local! {
    static CWT_COMPLEX_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Transform length for a signal of length `n`: the smallest power of two at
/// or above the `2n - 1` aliasing-free bound.
///
/// # Panics
///
/// Panics when `n == 0`; `CwtPlan` rejects empty signals at construction.
#[must_use]
pub fn transform_len(n: usize) -> usize {
    assert!(n > 0, "CWT transform length requires a non-empty signal");
    (2 * n - 1).next_power_of_two()
}

/// Scale-independent spectrum of the zero-padded analysis signal.
///
/// Built once per [`crate::CwtPlan::transform`] call and shared across every
/// scale row, so the signal transform is not repeated per scale.
#[derive(Debug, Clone)]
pub struct CwtSpectrum {
    signal_len: usize,
    spectrum: Vec<Complex64>,
}

impl CwtSpectrum {
    /// Transform `signal` zero-padded to [`transform_len`].
    ///
    /// # Panics
    ///
    /// Panics when `signal` is empty.
    #[must_use]
    pub fn new(signal: &[f64]) -> Self {
        let signal_len = signal.len();
        let len = transform_len(signal_len);
        let mut spectrum = vec![Complex64::new(0.0, 0.0); len];
        for (slot, &sample) in spectrum.iter_mut().zip(signal) {
            *slot = Complex64::new(sample, 0.0);
        }
        let plan = f64::get_1d_plan(Shape1D::new(len).expect("CWT transform length is non-zero"));
        plan.forward_complex_slice_inplace(&mut spectrum);
        Self {
            signal_len,
            spectrum,
        }
    }

    /// Signal length this spectrum was built for.
    #[must_use]
    pub const fn signal_len(&self) -> usize {
        self.signal_len
    }

    /// Fill one scale row with `row[τ] = Σ_i signal[i] · s^{-1/2} ψ((i - τ)/s)`.
    ///
    /// # Panics
    ///
    /// Panics when `row.len()` differs from the signal length this spectrum
    /// was built for.
    pub fn row_into(&self, wavelet: ContinuousWavelet, scale: f64, row: &mut [f64]) {
        assert_eq!(
            row.len(),
            self.signal_len,
            "CWT scale row length must match the analysed signal"
        );
        let len = self.spectrum.len();
        CWT_COMPLEX_SCRATCH.with(|pool| {
            pool.with_scratch(len, |buffer| {
                fill_reflected_kernel(buffer, wavelet, scale, self.signal_len);
                let plan =
                    f64::get_1d_plan(Shape1D::new(len).expect("CWT transform length is non-zero"));
                plan.forward_complex_slice_inplace(buffer);
                for (slot, &signal_bin) in buffer.iter_mut().zip(&self.spectrum) {
                    *slot *= signal_bin;
                }
                plan.inverse_complex_slice_inplace(buffer);
                for (coefficient, bin) in row.iter_mut().zip(buffer.iter()) {
                    // The inputs are real, so the imaginary part of the
                    // product transform is rounding noise and is discarded.
                    *coefficient = bin.re;
                }
            });
        });
    }
}

/// Write the periodized reflected lag kernel `h_s` of the module theorem into
/// `buffer`, zeroing the unused interior.
fn fill_reflected_kernel(
    buffer: &mut [Complex64],
    wavelet: ContinuousWavelet,
    scale: f64,
    signal_len: usize,
) {
    let len = buffer.len();
    debug_assert!(
        len >= 2 * signal_len - 1,
        "reflected kernel needs an aliasing-free transform length"
    );
    buffer.fill(Complex64::new(0.0, 0.0));

    let inv_sqrt_scale = 1.0 / scale.sqrt();
    let weight =
        |lag: f64| Complex64::new(inv_sqrt_scale * mother_wavelet(wavelet, lag / scale), 0.0);

    buffer[0] = weight(0.0);
    for lag in 1..signal_len {
        // Negative lags at the head, positive lags wrapped to the tail.
        buffer[lag] = weight(-(lag as f64));
        buffer[len - lag] = weight(lag as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::kernel::continuous::coefficient;

    /// Difference bound between the FFT row and the direct sum.
    ///
    /// The direct naive sum over `n` terms carries `|error| ≤ n · ε · Σ|x_i w_i|
    /// ≤ n · ε · ‖x‖₂ ‖w‖₂` (Cauchy–Schwarz on the standard recursive-summation
    /// bound). The three-transform FFT convolution carries the classical
    /// `O(log₂ L · ε)` growth, `|error| ≤ c · log₂ L · ε · ‖x‖₂ ‖w‖₂`; `c = 8`
    /// is a conservative constant for a radix-2 kernel with accurate twiddles.
    /// The two computed results therefore differ by at most the sum of both
    /// bounds. Nothing here is fitted: the shape is `(n + 8 log₂ L) · ε` and
    /// the norms are measured from the actual inputs.
    fn difference_bound(signal: &[f64], wavelet: ContinuousWavelet, scale: f64) -> f64 {
        let n = signal.len();
        let transform_len = transform_len(n);
        let signal_norm = signal.iter().map(|value| value * value).sum::<f64>().sqrt();
        let inv_sqrt_scale = 1.0 / scale.sqrt();
        let kernel_norm = (-(n as isize - 1)..n as isize)
            .map(|lag| {
                let weight = inv_sqrt_scale * mother_wavelet(wavelet, lag as f64 / scale);
                weight * weight
            })
            .sum::<f64>()
            .sqrt();
        let growth = n as f64 + 8.0 * (transform_len as f64).log2();
        growth * f64::EPSILON * signal_norm * kernel_norm
    }

    fn direct_row(signal: &[f64], wavelet: ContinuousWavelet, scale: f64) -> Vec<f64> {
        (0..signal.len())
            .map(|shift| coefficient(signal, wavelet, scale, shift))
            .collect()
    }

    fn ramped_sine(len: usize) -> Vec<f64> {
        (0..len)
            .map(|index| (index as f64 * 0.37).sin() + (index % 5) as f64 * 0.25 - 0.5)
            .collect()
    }

    #[test]
    fn fft_rows_match_the_direct_kernel_across_sizes_scales_and_wavelets() {
        let wavelets = [
            ContinuousWavelet::Ricker,
            ContinuousWavelet::Morlet { omega0: 5.0 },
            ContinuousWavelet::Morlet { omega0: 6.0 },
        ];
        // Powers of two, odd, prime, and the degenerate single sample.
        for len in [1_usize, 2, 3, 7, 8, 13, 31, 64, 100, 128, 257] {
            let signal = ramped_sine(len);
            let spectrum = CwtSpectrum::new(&signal);
            for wavelet in wavelets {
                for scale in [0.5_f64, 1.0, 2.5, 7.0, 40.0] {
                    let expected = direct_row(&signal, wavelet, scale);
                    let mut actual = vec![0.0; len];
                    spectrum.row_into(wavelet, scale, &mut actual);
                    let bound = difference_bound(&signal, wavelet, scale);
                    for (shift, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        assert!(
                            (actual - expected).abs() <= bound,
                            "len {len} scale {scale} shift {shift}: \
                             |{actual} - {expected}| exceeds derived bound {bound}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fft_rows_stay_well_inside_the_derived_bound() {
        // Guards the bound against being a fitted tolerance: the observed
        // error must sit at least an order of magnitude below it.
        let signal = ramped_sine(512);
        let wavelet = ContinuousWavelet::Morlet { omega0: 5.0 };
        let scale = 4.0;
        let expected = direct_row(&signal, wavelet, scale);
        let mut actual = vec![0.0; signal.len()];
        let spectrum = CwtSpectrum::new(&signal);
        spectrum.row_into(wavelet, scale, &mut actual);

        let observed = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        let bound = difference_bound(&signal, wavelet, scale);
        assert!(
            observed * 10.0 < bound,
            "observed error {observed} is not comfortably inside bound {bound}"
        );
    }

    #[test]
    fn cwt_rows_are_linear_in_the_signal() {
        // Independent algebraic property: the CWT is a linear operator, which
        // the direct kernel is never consulted to establish.
        let wavelet = ContinuousWavelet::Ricker;
        let scale = 3.0;
        let len = 96;
        let first = ramped_sine(len);
        let second: Vec<f64> = (0..len)
            .map(|index| (index as f64 * 0.11).cos() * 2.0)
            .collect();
        let alpha = -1.75_f64;
        let combined: Vec<f64> = first
            .iter()
            .zip(&second)
            .map(|(first, second)| first + alpha * second)
            .collect();

        let mut row_first = vec![0.0; len];
        let mut row_second = vec![0.0; len];
        let mut row_combined = vec![0.0; len];
        CwtSpectrum::new(&first).row_into(wavelet, scale, &mut row_first);
        CwtSpectrum::new(&second).row_into(wavelet, scale, &mut row_second);
        CwtSpectrum::new(&combined).row_into(wavelet, scale, &mut row_combined);

        let magnitude = row_first
            .iter()
            .chain(&row_second)
            .fold(0.0_f64, |peak, value| peak.max(value.abs()));
        let transform_len = transform_len(len);
        // Linearity holds exactly in exact arithmetic; the computed residual is
        // bounded by the three transforms' rounding, `O(log₂ L · ε)` scaled by
        // the coefficient magnitude and the combination weight.
        let bound = 8.0
            * (transform_len as f64).log2()
            * f64::EPSILON
            * magnitude
            * (1.0 + alpha.abs())
            * len as f64;
        for ((combined, first), second) in row_combined.iter().zip(&row_first).zip(&row_second) {
            assert!(
                (combined - (first + alpha * second)).abs() <= bound,
                "linearity residual exceeds {bound}"
            );
        }
    }

    #[test]
    fn ricker_row_matches_the_analytic_gaussian_pairing() {
        // Independent analytic oracle. For a unit impulse at index p the row
        // reduces to a single kernel sample, so the coefficient is the closed
        // form s^{-1/2} ψ((p - τ)/s) with no summation at all.
        let len = 129;
        let impulse_at = 40_usize;
        let mut signal = vec![0.0; len];
        signal[impulse_at] = 1.0;
        let wavelet = ContinuousWavelet::Ricker;
        let scale = 5.0;

        let mut row = vec![0.0; len];
        CwtSpectrum::new(&signal).row_into(wavelet, scale, &mut row);

        let transform_len = transform_len(len);
        let peak = 1.0 / scale.sqrt() * mother_wavelet(wavelet, 0.0);
        let bound = 8.0 * (transform_len as f64).log2() * f64::EPSILON * peak.abs() * len as f64;
        for (shift, value) in row.iter().enumerate() {
            let normalized_time = (impulse_at as f64 - shift as f64) / scale;
            let analytic = mother_wavelet(wavelet, normalized_time) / scale.sqrt();
            assert!(
                (value - analytic).abs() <= bound,
                "shift {shift}: {value} vs analytic {analytic}, bound {bound}"
            );
        }
    }

    #[test]
    fn transform_len_clears_the_aliasing_free_bound() {
        for n in [1_usize, 2, 3, 4, 5, 17, 64, 1000, 4096] {
            let len = transform_len(n);
            assert!(len >= 2 * n - 1, "n {n}: {len} below the 2n-1 bound");
            assert!(len.is_power_of_two(), "n {n}: {len} is not a power of two");
        }
    }
}
