//! Sub-bin peak parameters from three DFT bins, with estimate-and-subtract
//! across a set of peaks (ADR 0066, `docs/adr/0066-sub-bin-peak-estimation.md`).
//!
//! # Model
//!
//! A frame of `N` samples analysed through the rectangular window has the
//! unnormalized DFT `X_p = Σ_t s[t] e^{−2πi p t / N}`. A real tone
//! `A cos(2π f t / N + φ)`, `f` in bins, is `a e^{2πi f t / N} + ā e^{−2πi f t / N}`
//! with `a = (A/2) e^{iφ}`, so it contributes
//! `a R(f − p) + ā R(−f − p)` to bin `p`, where
//! `R(u) = Σ_{t<N} e^{2πi u t / N} = (e^{2πiu} − 1) / (e^{2πiu/N} − 1)`.
//!
//! # Offset
//!
//! For the peak bin `k` of a tone at `f = k + δ`, the offset is Candan's
//! corrected complex ratio (Candan, "A method for fine resolution frequency
//! estimation from three DFT samples", IEEE SPL 18(6), 2011):
//! `δ = Re[(X_{k−1} − X_{k+1}) / (2X_k − X_{k−1} − X_{k+1})] · tan(π/N)/(π/N)`.
//! For one complex exponential the ratio `r` is exactly
//! `tan(πδ/N) / tan(π/N)`, so the corrected ratio is `(N/π) tan(πδ/N)` and
//! errs by `(N/π) (tan(πδ/N) − πδ/N)`, `|δ|³ (π/N)² / 3` to leading order,
//! bins. A real tone adds its own negative-frequency image, which moves the
//! offset by at most `|δ| (1 − δ²) (π/N)² / sin²(π (2k + δ) / N)` bins
//! (ADR 0066). An offset outside `|δ| ≤ ½` means bin `k` holds no peak and the
//! estimate is rejected; so is a ratio whose denominator vanishes.
//!
//! # Amplitude and phase
//!
//! Bin `k` holds `a R(δ) + ā R(−(2k + δ))`. Solving it together with its
//! conjugate for `a` removes the image exactly:
//! `a = (X_k R̄(δ) − X̄_k R(−(2k + δ))) / (|R(δ)|² − |R(−(2k + δ))|²)`,
//! `A = 2|a|`, `φ = arg a`. At `k = 0` and `k = N/2` the tone is its own image,
//! the determinant vanishes and the estimate is rejected.
//!
//! # Several peaks
//!
//! Each peak's bins also hold the other tones' leakage. The peaks are estimated
//! strongest first, each on the spectrum with every other peak's current
//! estimate subtracted, and the whole set is re-estimated for the requested
//! number of rounds: a tone estimated to relative error `ε` leaves interference
//! of order `ε` for the next round. The DFT is linear, so the residual at a bin
//! is the input bin minus the estimates' `R` terms; nothing is re-transformed.
//! A round costs three kernel evaluations per pair of peaks.

use core::num::NonZeroUsize;

use eunomia::{Complex, RealField};

use crate::domain::contracts::error::PeakEstimationError;

/// A tone's parameters read from a spectrum below bin resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeakEstimate<T> {
    position: T,
    amplitude: T,
    phase: T,
}

impl<T: Copy> PeakEstimate<T> {
    /// The tone's frequency in bins, `k + δ`: multiply by `sample_rate / N`
    /// for hertz.
    #[must_use]
    pub fn position(&self) -> T {
        self.position
    }

    /// The tone's amplitude `A` in the units of the frame's samples.
    #[must_use]
    pub fn amplitude(&self) -> T {
        self.amplitude
    }

    /// The tone's phase `φ` in radians at the frame's first sample, in
    /// `(−π, π]`.
    #[must_use]
    pub fn phase(&self) -> T {
        self.phase
    }
}

/// Estimates the tone at each bin of `peaks` from `spectrum`, the unnormalized
/// DFT of an `N`-sample frame analysed through the rectangular window.
///
/// Returns one entry per element of `peaks`, in the same order: the estimate,
/// or `None` where the bin holds no resolvable peak (its offset leaves the
/// half-bin, or the tone at DC or Nyquist is its own image). `rounds` is the
/// number of estimate-and-subtract passes over the whole set; a single peak
/// needs one. Neighbouring bins wrap around the frame, as the DFT does.
///
/// # Errors
///
/// - [`PeakEstimationError::FrameTooShort`] when `spectrum` has fewer than
///   three bins;
/// - [`PeakEstimationError::FrameTooLong`] when `N` is not exactly
///   representable in `T`, so bin positions would round;
/// - [`PeakEstimationError::PeakOutOfRange`] when a peak bin is `N` or more;
/// - [`PeakEstimationError::DuplicatePeak`] when a bin is listed twice, which
///   would subtract its tone from itself.
///
/// # Examples
///
/// ```
/// use apollo_stft::estimate_peaks;
/// use core::num::NonZeroUsize;
///
/// let n = 256;
/// let signal: Vec<f64> = (0..n)
///     .map(|t| 0.8 * (std::f64::consts::TAU * 20.3 * t as f64 / n as f64 + 0.4).cos())
///     .collect();
/// let spectrum = apollo_fft::fft_1d_slice::<f64>(&signal);
/// let estimates = estimate_peaks(&spectrum, &[20], NonZeroUsize::MIN)?;
/// let tone = estimates[0].expect("bin 20 holds the tone");
/// assert!((tone.position() - 20.3).abs() < 1e-3);
/// assert!((tone.amplitude() - 0.8).abs() < 1e-3);
/// assert!((tone.phase() - 0.4).abs() < 1e-2);
/// # Ok::<(), apollo_stft::PeakEstimationError>(())
/// ```
pub fn estimate_peaks<T: RealField>(
    spectrum: &[Complex<T>],
    peaks: &[usize],
    rounds: NonZeroUsize,
) -> Result<Vec<Option<PeakEstimate<T>>>, PeakEstimationError> {
    let len = spectrum.len();
    if len < 3 {
        return Err(PeakEstimationError::FrameTooShort { len });
    }
    // An `N` exact in `T` makes every bin index below it exact too.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the round trip below rejects every length the conversion rounds"
    )]
    let len_f64 = len as f64;
    if T::from_f64(len_f64).to_f64() != len_f64 {
        return Err(PeakEstimationError::FrameTooLong { len });
    }
    for (index, &bin) in peaks.iter().enumerate() {
        if bin >= len {
            return Err(PeakEstimationError::PeakOutOfRange { bin, len });
        }
        if peaks[..index].contains(&bin) {
            return Err(PeakEstimationError::DuplicatePeak { bin });
        }
    }
    let frame = Frame::new(len);
    let mut order: Vec<usize> = (0..peaks.len()).collect();
    // Strongest first. Widening a magnitude to f64 is exact for both scalars
    // and gives the total order a sort requires.
    let strength = |i: usize| spectrum[peaks[i]].norm().to_f64();
    order.sort_by(|&a, &b| strength(b).total_cmp(&strength(a)));
    let mut estimates: Vec<Option<PeakEstimate<T>>> = vec![None; peaks.len()];
    for _ in 0..rounds.get() {
        for &i in &order {
            let read = |bin: usize| {
                let position = Frame::<T>::index(bin);
                estimates
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .filter_map(|(_, estimate)| estimate.as_ref())
                    .fold(spectrum[bin], |residual, tone| {
                        residual - frame.contribution(tone, position)
                    })
            };
            let estimate = frame.estimate(&read, peaks[i]);
            estimates[i] = estimate;
        }
    }
    Ok(estimates)
}

/// The frame's length in `T` and the constants of its kernel.
struct Frame<T> {
    len: usize,
    n: T,
    /// `tan(π/N) / (π/N)`, Candan's correction.
    correction: T,
}

impl<T: RealField> Frame<T> {
    fn new(len: usize) -> Self {
        let n = Self::index(len);
        let step = T::PI / n;
        Self {
            len,
            n,
            correction: step.tan() / step,
        }
    }

    /// A bin index or length in `T`, exact for every value up to the frame's
    /// length once `estimate_peaks` has checked that length.
    fn index(value: usize) -> T {
        #[expect(
            clippy::cast_precision_loss,
            reason = "estimate_peaks rejects frames whose length rounds in T"
        )]
        let value = value as f64;
        T::from_f64(value)
    }

    /// `R(u) = (e^{2πiu} − 1) / (e^{2πiu/N} − 1)`, with both phases reduced
    /// to their principal turn so a position thousands of bins out keeps the
    /// scalar's full precision; `N` where the denominator vanishes, the limit
    /// at the multiples of `N`.
    fn kernel(&self, u: T) -> Complex<T> {
        let one = Complex::new(T::ONE, T::ZERO);
        let turns = u / self.n;
        let denominator = Complex::cis(T::TAU * (turns - turns.round())) - one;
        if denominator.norm() < T::EPSILON {
            return Complex::new(self.n, T::ZERO);
        }
        (Complex::cis(T::TAU * (u - u.round())) - one) / denominator
    }

    /// A tone's contribution `a R(f − p) + ā R(−f − p)` to position `p`.
    fn contribution(&self, tone: &PeakEstimate<T>, p: T) -> Complex<T> {
        let half = T::ONE / (T::ONE + T::ONE);
        let a = Complex::from_polar(tone.amplitude * half, tone.phase);
        a * self.kernel(tone.position - p) + a.conj() * self.kernel(-tone.position - p)
    }

    /// The tone at bin `k` of the spectrum `read` returns, or `None` where its
    /// offset leaves the half-bin or the image solve is singular.
    fn estimate(&self, read: &impl Fn(usize) -> Complex<T>, k: usize) -> Option<PeakEstimate<T>> {
        let before = read((k + self.len - 1) % self.len);
        let peak = read(k);
        let after = read((k + 1) % self.len);
        let two = T::ONE + T::ONE;
        let half = T::ONE / two;
        let delta = ((before - after) / (peak * two - before - after)).re * self.correction;
        // False for the NaN of a vanishing denominator too.
        let within_half_bin = delta.abs() <= half;
        if !within_half_bin {
            return None;
        }
        let bin = Self::index(k);
        let direct = self.kernel(delta);
        let image = self.kernel(-(bin * two + delta));
        let determinant = direct.norm_sqr() - image.norm_sqr();
        let solvable = determinant > T::EPSILON * direct.norm_sqr();
        if !solvable {
            return None;
        }
        let a = (peak * direct.conj() - peak.conj() * image) / determinant;
        Some(PeakEstimate {
            position: bin + delta,
            amplitude: two * a.norm(),
            phase: a.arg(),
        })
    }
}

#[cfg(test)]
mod tests;
