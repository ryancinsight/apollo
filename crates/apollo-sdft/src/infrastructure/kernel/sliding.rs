//! Sliding DFT kernel primitives.
//!
//! For window length N, the tracked bin is
//! `X_k[n] = sum_{m=0}^{N-1} x[n-N+1+m] exp(-2pi i k m/N)`.
//! When x_old leaves and x_new enters at the end, the recurrence is
//! X_k <- (X_k + x_new - x_old) exp(2pi i k/N).

use crate::domain::contracts::error::{SdftError, SdftResult};
use eunomia::Complex64;
use mnemosyne::scratch::ScratchPool;
use moirai::ParallelSliceMut;

/// Below this O(bin_count * window_len) count, serial loops avoid scheduling overhead.
const DIRECT_PAR_OP_THRESHOLD: usize = 16_384;

/// Below this bin count, serial recurrence updates avoid scheduling overhead.
const UPDATE_PAR_BIN_THRESHOLD: usize = 16_384;

/// Below this window length, scalar accumulation avoids scratch setup overhead.
const HERMES_DIRECT_BIN_LEN_THRESHOLD: usize = 128;

thread_local! {
    static DIRECT_BIN_WEIGHT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Build update twiddle factors for SDFT bins.
#[must_use]
pub fn update_twiddles(window_len: usize, bin_count: usize) -> Vec<Complex64> {
    (0..bin_count)
        .map(|bin| {
            let angle = std::f64::consts::TAU * bin as f64 / window_len as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect()
}

/// Compute direct DFT bins for a real-valued window.
///
/// # Errors
/// Returns [`SdftError::EmptyWindow`] if `window` is empty.
/// Returns [`SdftError::BinCountExceedsWindow`] if `bin_count > window.len()`.
pub fn direct_bins(window: &[f64], bin_count: usize) -> SdftResult<Vec<Complex64>> {
    let mut bins = vec![Complex64::new(0.0, 0.0); bin_count];
    direct_bins_into(window, &mut bins)?;
    Ok(bins)
}

/// Compute direct DFT bins for a real-valued window into caller-owned storage.
pub fn direct_bins_into(window: &[f64], bins: &mut [Complex64]) -> SdftResult<()> {
    let n = window.len();
    if n == 0 {
        return Err(SdftError::EmptyWindow);
    }
    if bins.len() > n {
        return Err(SdftError::BinCountExceedsWindow);
    }
    let work_items = bins.len().saturating_mul(n);
    if work_items >= DIRECT_PAR_OP_THRESHOLD {
        bins.par_mut().enumerate(|bin, slot| {
            *slot = direct_bin(window, n, bin);
        });
    } else {
        bins.iter_mut().enumerate().for_each(|(bin, slot)| {
            *slot = direct_bin(window, n, bin);
        });
    }
    Ok(())
}

#[inline]
fn direct_bin(window: &[f64], n: usize, bin: usize) -> Complex64 {
    if window.len() >= HERMES_DIRECT_BIN_LEN_THRESHOLD {
        return direct_bin_hermes(window, n, bin);
    }
    direct_bin_scalar(window, n, bin)
}

#[inline]
fn direct_bin_scalar(window: &[f64], n: usize, bin: usize) -> Complex64 {
    window
        .iter()
        .enumerate()
        .fold(Complex64::new(0.0, 0.0), |acc, (index, &value)| {
            let angle = -std::f64::consts::TAU * bin as f64 * index as f64 / n as f64;
            acc + Complex64::new(value, 0.0) * Complex64::new(angle.cos(), angle.sin())
        })
}

fn direct_bin_hermes(window: &[f64], n: usize, bin: usize) -> Complex64 {
    DIRECT_BIN_WEIGHT_SCRATCH.with(|pool| {
        pool.with_scratch(window.len(), |weights| {
            fill_direct_bin_weights(weights, n, bin, DirectBinComponent::Real);
            let re = hermes_simd::dot::<f64>(window, weights)
                .expect("SDFT Hermes real dot uses equal-length window and weight slices");
            fill_direct_bin_weights(weights, n, bin, DirectBinComponent::Imaginary);
            let im = hermes_simd::dot::<f64>(window, weights)
                .expect("SDFT Hermes imaginary dot uses equal-length window and weight slices");
            Complex64::new(re, im)
        })
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectBinComponent {
    Real,
    Imaginary,
}

fn fill_direct_bin_weights(
    weights: &mut [f64],
    n: usize,
    bin: usize,
    component: DirectBinComponent,
) {
    for (index, weight) in weights.iter_mut().enumerate() {
        let angle = -std::f64::consts::TAU * bin as f64 * index as f64 / n as f64;
        *weight = match component {
            DirectBinComponent::Real => angle.cos(),
            DirectBinComponent::Imaginary => angle.sin(),
        };
    }
}

/// Apply one O(bin_count) sliding DFT update.
///
/// ## Invariant
///
/// After each call, `bins[k]` equals the DFT of the current sliding window:
/// `bins[k] = sum_{j=0}^{N-1} window[(head+j) % N] * exp(-2pi i k j / N)`
///
/// ## Recurrence derivation
/// When the window advances by one sample (removing x_old, inserting x_new):
/// `DFT_new[k] = DFT_old[k] - x_old + x_new`
/// followed by multiplication by `twiddle[k] = exp(2pi i k / N)` (phase advance).
/// Proof: shifting the window by one sample in time multiplies each DFT bin
/// by exp(2pi i k / N), as a one-sample time-delay corresponds to multiplication
/// by exp(-2pi i k / N) in frequency, and the recurrence advances the phase forward.
pub fn update_bins(bins: &mut [Complex64], twiddles: &[Complex64], outgoing: f64, incoming: f64) {
    let delta = Complex64::new(incoming - outgoing, 0.0);
    if bins.len() >= UPDATE_PAR_BIN_THRESHOLD && twiddles.len() >= bins.len() {
        bins.par_mut().enumerate(|index, bin| {
            *bin = update_bin(*bin, twiddles[index], delta);
        });
    } else {
        bins.iter_mut()
            .zip(twiddles.iter())
            .for_each(|(bin, twiddle)| {
                *bin = update_bin(*bin, *twiddle, delta);
            });
    }
}

#[inline]
fn update_bin(bin: Complex64, twiddle: Complex64, delta: Complex64) -> Complex64 {
    (bin + delta) * twiddle
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_abs_diff_eq;

    #[test]
    fn moirai_parallel_direct_bins_match_serial_formula_at_threshold() {
        let window_len = 128;
        let bin_count = DIRECT_PAR_OP_THRESHOLD / window_len;
        let window = (0..window_len)
            .map(|index| (index as f64 * 0.125).sin() - (index as f64 * 0.03125).cos())
            .collect::<Vec<_>>();
        let mut actual = vec![Complex64::new(0.0, 0.0); bin_count];

        direct_bins_into(&window, &mut actual).expect("parallel direct bins");

        for (bin, actual) in actual.iter().enumerate() {
            let expected = direct_bin_scalar(&window, window_len, bin);
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn hermes_direct_bin_matches_scalar_formula_at_threshold() {
        let window_len = HERMES_DIRECT_BIN_LEN_THRESHOLD;
        let window = (0..window_len)
            .map(|index| (index as f64 * 0.125).sin() - (index as f64 * 0.03125).cos())
            .collect::<Vec<_>>();

        for bin in [0usize, 1, 17, 64, 127] {
            let actual = direct_bin_hermes(&window, window_len, bin);
            let expected = direct_bin_scalar(&window, window_len, bin);
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn direct_bin_weights_match_component_formula() {
        let n = 16;
        let bin = 5;
        let mut weights = vec![0.0; n];

        fill_direct_bin_weights(&mut weights, n, bin, DirectBinComponent::Imaginary);

        for (index, actual) in weights.iter().copied().enumerate() {
            let angle = -std::f64::consts::TAU * bin as f64 * index as f64 / n as f64;
            assert_eq!(actual.to_bits(), angle.sin().to_bits(), "index={index}");
        }
    }

    #[test]
    fn moirai_parallel_update_bins_match_recurrence_formula_at_threshold() {
        let mut bins = (0..UPDATE_PAR_BIN_THRESHOLD)
            .map(|index| Complex64::new(index as f64 * 0.25, -(index as f64) * 0.125))
            .collect::<Vec<_>>();
        let before = bins.clone();
        let twiddles = update_twiddles(UPDATE_PAR_BIN_THRESHOLD, UPDATE_PAR_BIN_THRESHOLD);
        let delta = Complex64::new(1.25 - -0.5, 0.0);

        update_bins(&mut bins, &twiddles, -0.5, 1.25);

        for (index, actual) in bins.iter().enumerate() {
            let expected = update_bin(before[index], twiddles[index], delta);
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }
}
