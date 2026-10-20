//! Sliding DFT kernel primitives.
//!
//! For window length `N` and bin `k`, the tracked bin is
//! `X_k[n] = sum_{m=0}^{N-1} x[n-N+1+m] exp(-2πi k m/N)`: the DFT of the
//! window that ends at the newest sample `x[n]`.
//!
//! ## The modulated form
//!
//! The textbook recurrence `X_k <- (X_k + x_new - x_old) exp(2πi k/N)` has its
//! pole on the unit circle. The computed twiddle is not of unit modulus, so
//! the error already in `X_k` is scaled at every step while fresh rounding
//! joins it, and nothing ever removes what has accumulated (Jacobsen and
//! Lyons, "The sliding DFT", IEEE Signal Processing Magazine 20(2), 2003,
//! the section on stability). Measured here before this form landed, a
//! 48-sample window tracked over a million `f64` updates wandered two to
//! five times past the bound stated on `SdftPlan::drift_bound`.
//!
//! This kernel keeps the modulated form instead (Duda, "Accurate, guaranteed
//! stable, sliding discrete Fourier transform", IEEE Signal Processing
//! Magazine 27(6), 2010). Each bin accumulates
//! `S_k = sum_j x[j] · E[(k·j) mod N]` over the window, with `j` the absolute
//! sample index and `E[q] = exp(-2πi q/N)` read from one `N`-entry table. The
//! sample leaving and the sample entering have indices `N` apart, so they
//! take the same table entry and one advance is
//! `S_k += (x_new - x_old) · E[(k·n) mod N]`; the bin itself is
//! `X_k = S_k · conj(E[(k·(n+1)) mod N])`. No multiplication touches the
//! accumulated value, so its error is a sum of per-step roundings rather than
//! a product of them, and the refresh bounds that sum outright: on a fixed
//! cadence a bin is re-summed from the window through the same table
//! (`refresh_bin` below), which resets its error to that of one direct sum. The
//! resulting bound, independent of the update count, is derived on
//! `SdftPlan::drift_bound`.

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet APOLLO-UNWRAP-1: pre-existing debt"
    )
)]

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

/// Release scratch pools held by the sliding SDFT kernel on the calling thread.
#[inline]
pub fn release_scratch() {
    DIRECT_BIN_WEIGHT_SCRATCH.with(|pool| pool.release());
}

#[cfg(test)]
pub(crate) fn direct_bin_weight_scratch_capacity() -> usize {
    DIRECT_BIN_WEIGHT_SCRATCH.with(|pool| pool.capacity())
}

/// The modulation table `E[q] = exp(-2πi q/N)` for `q` in `0..N`.
///
/// Every twiddle a bin ever needs is one of these `N` entries, indexed by
/// `(k·j) mod N`, so a sample and the sample that replaces it `N` updates
/// later are weighted by the same value.
#[must_use]
pub fn modulation_table(window_len: usize) -> Vec<Complex64> {
    (0..window_len)
        .map(|index| {
            let angle = -std::f64::consts::TAU * index as f64 / window_len as f64;
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
///
/// # Errors
/// Returns [`SdftError::EmptyWindow`] if `window` is empty.
/// Returns [`SdftError::BinCountExceedsWindow`] if `bins.len() > window.len()`.
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

/// One tracked bin of the modulated sliding DFT.
///
/// `sum` is `sum_j x[j] · E[(k·j) mod N]` over the current window and
/// `phase` is `(k·n) mod N` for the newest sample `x[n]`; the bin index `k`
/// and the window length `N` are the slot the value sits in and the table it
/// reads, passed at every operation rather than stored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModulatedBin {
    sum: Complex64,
    phase: usize,
}

/// `value mod window_len` for a `value` below `2 · window_len`.
#[inline]
const fn wrap(value: usize, window_len: usize) -> usize {
    if value >= window_len {
        value - window_len
    } else {
        value
    }
}

impl ModulatedBin {
    /// An empty bin `bin` whose newest sample is the last of the first window,
    /// absolute index `window_len - 1`.
    #[must_use]
    pub const fn new(bin: usize, window_len: usize) -> Self {
        Self {
            sum: Complex64::new(0.0, 0.0),
            phase: (bin * (window_len - 1)) % window_len,
        }
    }

    /// The table index of the next sample, `(k·(n+1)) mod N`, which is also
    /// the index of the oldest sample in the window (`n-N+1 ≡ n+1`).
    #[inline]
    const fn next_phase(&self, bin: usize, window_len: usize) -> usize {
        wrap(self.phase + bin, window_len)
    }

    /// Advance by one sample: `sum += delta · E[(k·(n+1)) mod N]`.
    #[inline]
    fn advance(&mut self, bin: usize, table: &[Complex64], delta: f64) {
        let phase = self.next_phase(bin, table.len());
        self.sum += table[phase].scale(delta);
        self.phase = phase;
    }

    /// The tracked DFT bin, `sum · conj(E[(k·(n+1)) mod N])`.
    #[inline]
    fn value(&self, bin: usize, table: &[Complex64]) -> Complex64 {
        self.sum * table[self.next_phase(bin, table.len())].conj()
    }

    /// Re-sum from the window, oldest sample first: the oldest sample sits
    /// at the next phase and each later sample `k` entries on.
    fn refresh(&mut self, bin: usize, table: &[Complex64], window: (&[f64], &[f64])) {
        let window_len = table.len();
        let mut phase = self.next_phase(bin, window_len);
        let mut sum = Complex64::new(0.0, 0.0);
        for &sample in window.0.iter().chain(window.1) {
            sum += table[phase].scale(sample);
            phase = wrap(phase + bin, window_len);
        }
        self.sum = sum;
    }
}

/// Initialize every tracked bin from a full window, oldest sample first, and
/// write the bins; `tracked[k]` and `bins[k]` are bin `k`.
///
/// # Errors
/// Returns [`SdftError::EmptyWindow`] if `window` is empty,
/// [`SdftError::InitialWindowLengthMismatch`] if `table` is not
/// [`modulation_table`] of the window length, and
/// [`SdftError::BinCountExceedsWindow`] if more bins are tracked than the
/// window holds samples.
///
/// # Panics
/// Panics if `tracked` and `bins` differ in length.
pub fn initialize_bins(
    window: &[f64],
    table: &[Complex64],
    tracked: &mut [ModulatedBin],
    bins: &mut [Complex64],
) -> SdftResult<()> {
    let window_len = window.len();
    if window_len == 0 {
        return Err(SdftError::EmptyWindow);
    }
    if table.len() != window_len {
        return Err(SdftError::InitialWindowLengthMismatch);
    }
    if tracked.len() > window_len {
        return Err(SdftError::BinCountExceedsWindow);
    }
    assert_eq!(
        tracked.len(),
        bins.len(),
        "tracked bins and bin storage must hold the same bin count"
    );
    for (bin, (state, value)) in tracked.iter_mut().zip(bins.iter_mut()).enumerate() {
        *state = ModulatedBin::new(bin, window_len);
        state.refresh(bin, table, (window, &[]));
        *value = state.value(bin, table);
    }
    Ok(())
}

/// Advance every tracked bin by one sample and write the bins.
///
/// ## Invariant
///
/// After each call, `bins[k]` equals the DFT of the current sliding window
/// within the bound derived on `SdftPlan::drift_bound`:
/// `bins[k] = sum_{j=0}^{N-1} window[(head+j) % N] · exp(-2πi k j / N)`.
///
/// # Panics
/// Panics if `tracked` and `bins` differ in length.
pub fn update_bins(
    tracked: &mut [ModulatedBin],
    bins: &mut [Complex64],
    table: &[Complex64],
    outgoing: f64,
    incoming: f64,
) {
    assert_eq!(
        tracked.len(),
        bins.len(),
        "tracked bins and bin storage must hold the same bin count"
    );
    let delta = incoming - outgoing;
    if tracked.len() >= UPDATE_PAR_BIN_THRESHOLD {
        tracked.par_mut().enumerate(|bin, state| {
            state.advance(bin, table, delta);
        });
        let tracked: &[ModulatedBin] = tracked;
        bins.par_mut().enumerate(|bin, value| {
            *value = tracked[bin].value(bin, table);
        });
    } else {
        for (bin, (state, value)) in tracked.iter_mut().zip(bins.iter_mut()).enumerate() {
            state.advance(bin, table, delta);
            *value = state.value(bin, table);
        }
    }
}

/// Re-sum tracked bin `bin` from the current window, oldest sample first,
/// through the same table its advances read, and rewrite its bin.
///
/// The window arrives as the two slices of a ring buffer
/// (`VecDeque::as_slices`); the second may be empty.
pub fn refresh_bin(
    bin: usize,
    state: &mut ModulatedBin,
    value: &mut Complex64,
    table: &[Complex64],
    window: (&[f64], &[f64]),
) {
    debug_assert_eq!(
        window.0.len() + window.1.len(),
        table.len(),
        "invariant: the window holds one sample per table entry"
    );
    state.refresh(bin, table, window);
    *value = state.value(bin, table);
}

#[cfg(test)]
mod tests;
