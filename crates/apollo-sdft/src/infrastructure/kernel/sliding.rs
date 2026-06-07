//! Sliding DFT kernel primitives.
//!
//! For window length N, the tracked bin is
//! `X_k[n] = sum_{m=0}^{N-1} x[n-N+1+m] exp(-2pi i k m/N)`.
//! When x_old leaves and x_new enters at the end, the recurrence is
//! X_k <- (X_k + x_new - x_old) exp(2pi i k/N).

use crate::domain::contracts::error::{SdftError, SdftResult};
use num_complex::Complex64;

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
    for (bin, slot) in bins.iter_mut().enumerate() {
        *slot = window
            .iter()
            .enumerate()
            .fold(Complex64::new(0.0, 0.0), |acc, (index, &value)| {
                let angle = -std::f64::consts::TAU * bin as f64 * index as f64 / n as f64;
                acc + Complex64::new(value, 0.0) * Complex64::new(angle.cos(), angle.sin())
            });
    }
    Ok(())
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
    for (bin, twiddle) in bins.iter_mut().zip(twiddles.iter()) {
        *bin = (*bin + delta) * *twiddle;
    }
}
