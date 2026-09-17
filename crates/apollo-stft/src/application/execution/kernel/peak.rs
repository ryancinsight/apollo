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
//! complex ratio with its inverse-tangent closure (ADR 0066):
//! `δ = atan(Re[(X_{k−1} − X_{k+1}) / (2X_k − X_{k−1} − X_{k+1})] · tan(π/N))/(π/N)`.
//! For one complex exponential the ratio `r` is exactly
//! `tan(πδ/N) / tan(π/N)`. Inverting that relation removes the finite-length
//! bias that otherwise pushes a half-bin offset beyond the rejection boundary.
//! A real tone adds its own negative-frequency image. Its first-order offset
//! error scales as `|δ| (1 − δ²) (π/N)² / sin²(π (2k + δ) / N)` bins;
//! this is not an upper bound. An estimated offset outside `|δ| ≤ ½` is
//! rejected, as is a ratio whose denominator vanishes. Image bias or noise
//! can move an actual tone across this estimated-offset boundary.
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
//! Subtraction uses the same strength order, with bin index breaking ties, so
//! permuting the requested bins changes only the output order.
//! A round reads three bins per peak and subtracts every other estimate at
//! each, two kernel evaluations apiece: `6 P (P − 1)` for `P` peaks.
//!
//! A real tone appears at bin `b` and, as its image, at `N − b`; listing both
//! would subtract the tone from its own image, so the pair is refused.

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
    /// `[−π, π]`.
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
/// half-bin, or the tone at DC or Nyquist is its own image). A bin above
/// `N/2` reads the image of the tone at `N − b`: its position is `N − f` and
/// its phase `−φ`. `rounds` is the
/// number of estimate-and-subtract passes over the whole set; a single peak
/// needs one. Permuting `peaks` permutes the results without changing their
/// values. Neighbouring bins wrap around the frame, as the DFT does.
///
/// # Errors
///
/// - [`PeakEstimationError::FrameTooShort`] when `spectrum` has fewer than
///   three bins;
/// - [`PeakEstimationError::FrameTooLong`] when `N ε ≥ ½` for the scalar's
///   `ε`, where a position near `N` can no longer hold a sub-bin offset (below
///   that, positions resolve to `N ε` bins);
/// - [`PeakEstimationError::PeakOutOfRange`] when a peak bin is `N` or more;
/// - [`PeakEstimationError::DuplicatePeak`] when a bin is listed twice, and
///   [`PeakEstimationError::MirrorPeak`] when both `b` and `N − b` are, either
///   of which would subtract a tone from itself.
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
    // A position near N has spacing bounded by N ε. Require that bound below
    // half a bin; the exact DC/Nyquist rejection does not restrict frame size.
    // Below this limit N < ½ / ε ≤ 2^52, so index conversions are exact.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a length that rounds here is at least 2^53 and fails the check"
    )]
    let len_f64 = len as f64;
    if len_f64 * T::EPSILON.to_f64() >= 0.5 {
        return Err(PeakEstimationError::FrameTooLong { len });
    }
    for (index, &bin) in peaks.iter().enumerate() {
        if bin >= len {
            return Err(PeakEstimationError::PeakOutOfRange { bin, len });
        }
        if peaks[..index].contains(&bin) {
            return Err(PeakEstimationError::DuplicatePeak { bin });
        }
        let mirror = (len - bin) % len;
        if mirror != bin && peaks[..index].contains(&mirror) {
            return Err(PeakEstimationError::MirrorPeak { bin, mirror });
        }
    }
    let frame = Frame::new(len);
    let mut order: Vec<usize> = (0..peaks.len()).collect();
    // Strongest first. Widening a magnitude to f64 is exact for both scalars
    // and gives the total order a sort requires.
    let strength = |i: usize| spectrum[peaks[i]].norm().to_f64();
    order.sort_by(|&a, &b| {
        strength(b)
            .total_cmp(&strength(a))
            .then_with(|| peaks[a].cmp(&peaks[b]))
    });
    let mut estimates: Vec<Option<PeakEstimate<T>>> = vec![None; peaks.len()];
    for _ in 0..rounds.get() {
        for &i in &order {
            let read = |bin: usize| {
                let position = Frame::<T>::index(bin);
                order
                    .iter()
                    .copied()
                    .filter(|&j| j != i)
                    .filter_map(|j| estimates[j].as_ref())
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
    /// `π/N`, the inverse-tangent closure's angular scale.
    step: T,
    /// `tan(π/N)`, relating the three-bin ratio to `tan(πδ/N)`.
    tangent: T,
}

impl<T: RealField> Frame<T> {
    fn new(len: usize) -> Self {
        let n = Self::index(len);
        let step = T::PI / n;
        Self {
            len,
            n,
            step,
            tangent: step.tan(),
        }
    }

    /// A bin index or length in `T`, exact for every value up to the frame's
    /// length once `estimate_peaks` has checked that length.
    fn index(value: usize) -> T {
        #[expect(
            clippy::cast_precision_loss,
            reason = "estimate_peaks rejects lengths from ½ / ε, below 2^53"
        )]
        let value = value as f64;
        T::from_f64(value)
    }

    /// `R(u) = (e^{2πiu} − 1) / (e^{2πiu/N} − 1)` in the product form
    /// `e^{iπu(N−1)/N} sin(πu) / sin(πu/N)`, evaluated without cancellation.
    ///
    /// `R` has period `N`, so `u` is first reduced to `w = u − N·round(u/N)`,
    /// an exact subtraction, with `|w| ≤ N/2`. Writing `w = m + w'` for the
    /// nearest integer `m`, `e^{iπw} sin(πw) = e^{iπw'} sin(πw')`, so
    /// `R(w) = e^{iπw'} sin(πw') · e^{−iπw/N} / sin(πw/N)`: every factor is a
    /// sine or phasor of an angle within `±π/2`, accurate to a few `ε`
    /// relative even as `w'` or `w` approaches zero, where the quotient form
    /// subtracts nearly equal numbers. `R(0) = N`.
    fn kernel(&self, u: T) -> Complex<T> {
        let reduced = u - self.n * (u / self.n).round();
        if reduced == T::ZERO {
            return Complex::new(self.n, T::ZERO);
        }
        let fraction = reduced - reduced.round();
        let near = T::PI * fraction;
        let far = T::PI * reduced / self.n;
        Complex::cis(near - far) * (near.sin() / far.sin())
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
        // These bins equal their own mirrors exactly. Testing indices avoids
        // an O(k ε) determinant threshold that rejects valid large-frame bins.
        if k == 0 || (self.len % 2 == 0 && k == self.len / 2) {
            return None;
        }
        let before = read((k + self.len - 1) % self.len);
        let peak = read(k);
        let after = read((k + 1) % self.len);
        let two = T::ONE + T::ONE;
        let half = T::ONE / two;
        let ratio = ((before - after) / (peak * two - before - after)).re;
        let delta = (ratio * self.tangent).atan2(T::ONE) / self.step;
        // False for the NaN of a vanishing denominator too.
        let within_half_bin = delta.abs() <= half;
        if !within_half_bin {
            return None;
        }
        let bin = Self::index(k);
        let direct = self.kernel(delta);
        // `R` has period `N`, so the image's argument `−(2k + δ)` is formed
        // from the signed residue of `2k` modulo `N`, within `±N/2`: for bins
        // on either side of Nyquist that residue is small and keeps `δ`'s
        // precision where `2k + δ` would round it away in a long frame.
        let residue = (2 * k) % self.len;
        let signed = if 2 * residue > self.len {
            -Self::index(self.len - residue)
        } else {
            Self::index(residue)
        };
        let image = self.kernel(-(signed + delta));
        let determinant = direct.norm_sqr() - image.norm_sqr();
        // A non-positive or NaN determinant cannot yield a resolved tone.
        let solvable = determinant > T::ZERO;
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
