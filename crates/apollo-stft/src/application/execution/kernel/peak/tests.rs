//! The estimator against spectra summed directly from their samples, at every
//! shipped scalar.
//!
//! Bounds, for a tone of amplitude `A` at `k + δ` in an `N`-bin frame:
//!
//! - **Rounding.** A bin is a sum of `N` terms of magnitude at most `A`, so it
//!   carries at most `N ε A` of rounding (`ε` the scalar's epsilon, the
//!   summation's `(N − 1) ε Σ|x|` with the terms' own `≈ 3ε`), against a peak
//!   bin of at least `A N / π` on `|δ| ≤ ½`: a relative `η = 4 N ε`.
//! - **Offset.** With the bins `∝ 1/(δ − m)`, the ratio's denominator is
//!   `2 / (δ (1 − δ²))` and the three bins sum to at most that, so `η` moves the
//!   ratio by at most `1.5 η`, taken as `2η`. The estimator adds Candan's
//!   residual `(N/π) (tan(πδ/N) − πδ/N)` and the image floor
//!   `|δ| (1 − δ²) (π/N)² / sin²(π (2k + δ) / N)`.
//! - **Amplitude and phase.** An offset error `e` moves `ln|R(δ)|` by at most
//!   `2e` and `arg R(δ)` by at most `π e`; the image solve is exact, and the
//!   bins' rounding adds `2η`.
//! - **Another tone** `d` bins away with errors `Δf`, `ΔA`, `Δφ` leaves at
//!   most `(A_o/2) N (Δf + (ΔA/A_o + Δφ)/π) / d` in each bin read, relative
//!   `ρ = (A_o/A) (π/(2d)) (Δf + (ΔA/A_o + Δφ)/π)`, which moves the offset by
//!   at most `ρ` and the amplitude and phase by the offset's slopes plus `ρ`.

use core::num::NonZeroUsize;

use eunomia::{Complex, RealField};

use super::{estimate_peaks, Frame, PeakEstimate};
use crate::domain::contracts::error::PeakEstimationError;

const LEN: usize = 128;

#[derive(Clone, Copy, Debug)]
struct Tone {
    position: f64,
    amplitude: f64,
    phase: f64,
}

/// The unnormalized DFT of the tones' samples, summed term by term in `T`.
fn spectrum<T: RealField>(tones: &[Tone]) -> Vec<Complex<T>> {
    let n = T::from_f64(LEN as f64);
    let samples: Vec<T> = (0..LEN)
        .map(|t| {
            let t = T::from_f64(t as f64);
            tones.iter().fold(T::ZERO, |sum, tone| {
                sum + T::from_f64(tone.amplitude)
                    * (T::TAU * T::from_f64(tone.position) * t / n + T::from_f64(tone.phase)).cos()
            })
        })
        .collect();
    (0..LEN)
        .map(|p| {
            let step = -(T::TAU * T::from_f64(p as f64) / n);
            samples
                .iter()
                .enumerate()
                .fold(Complex::new(T::ZERO, T::ZERO), |sum, (t, &s)| {
                    sum + Complex::from_polar(s, step * T::from_f64(t as f64))
                })
        })
        .collect()
}

/// The bounds of the module doc for one tone alone.
#[derive(Clone, Copy, Debug)]
struct Bound {
    position: f64,
    amplitude: f64,
    phase: f64,
}

impl Bound {
    fn single<T: RealField>(tone: &Tone) -> Self {
        let n = LEN as f64;
        let eta = 4.0 * n * T::EPSILON.to_f64();
        let k = tone.position.round();
        let delta = tone.position - k;
        let step = std::f64::consts::PI / n;
        let image = delta.abs() * (1.0 - delta * delta) * step * step
            / (std::f64::consts::PI * (2.0 * k + delta) / n).sin().powi(2);
        let candan = ((delta.abs() * step).tan() - delta.abs() * step) / step;
        let offset = image + candan + 2.0 * eta;
        Self {
            position: offset,
            amplitude: tone.amplitude * (2.0 * offset + 2.0 * eta),
            phase: std::f64::consts::PI * offset + 2.0 * eta,
        }
    }

    /// This tone's bound with another tone's errors `other` added as `ρ`.
    fn with(self, tone: &Tone, other_tone: &Tone, other: &Self) -> Self {
        let d = (tone.position - other_tone.position).abs();
        let rho = other_tone.amplitude / tone.amplitude * std::f64::consts::PI / (2.0 * d)
            * (other.position
                + (other.amplitude / other_tone.amplitude + other.phase) / std::f64::consts::PI);
        let offset = self.position + rho;
        Self {
            position: offset,
            amplitude: self.amplitude + tone.amplitude * (2.0 * rho + rho),
            phase: self.phase + std::f64::consts::PI * rho + rho,
        }
    }

    fn check<T: RealField>(&self, estimate: &PeakEstimate<T>, tone: &Tone, label: &str) {
        let phase_error = {
            let raw = estimate.phase().to_f64() - tone.phase;
            (raw + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
        };
        let errors = (
            (estimate.position().to_f64() - tone.position).abs(),
            (estimate.amplitude().to_f64() - tone.amplitude).abs(),
            phase_error.abs(),
        );
        assert!(
            errors.0 <= self.position && errors.1 <= self.amplitude && errors.2 <= self.phase,
            "{label}: errors {errors:?} exceed {self:?} for {tone:?}"
        );
    }
}

fn single_tones_within_their_bounds<T: RealField>() {
    for tone in [
        Tone {
            position: 10.3,
            amplitude: 1.0,
            phase: 0.7,
        },
        Tone {
            position: 31.0,
            amplitude: 0.25,
            phase: -2.9,
        },
        Tone {
            position: 47.49,
            amplitude: 3.0,
            phase: 3.1,
        },
        Tone {
            position: 60.8,
            amplitude: 1.5,
            phase: -0.2,
        },
    ] {
        let spectrum = spectrum::<T>(&[tone]);
        let k = tone.position.round() as usize;
        let estimates = estimate_peaks(&spectrum, &[k], NonZeroUsize::MIN)
            .expect("invariant: a 128-bin frame and an in-range bin are valid");
        let estimate = estimates[0].expect("a lone tone's own bin holds its peak");
        Bound::single::<T>(&tone).check(&estimate, &tone, "single tone");
    }
}

#[test]
fn single_tones_within_their_bounds_for_every_scalar() {
    single_tones_within_their_bounds::<f64>();
    single_tones_within_their_bounds::<f32>();
}

/// One complex exponential has no image, so its offset error is Candan's
/// residual plus rounding alone: the ratio is `tan(πδ/N) / tan(π/N)` exactly,
/// the correction turns it into `(N/π) tan(πδ/N)`, and the error is
/// `(N/π) (tan(πδ/N) − πδ/N)`. Without the correction the ratio errs by about
/// `|δ| (1 − δ²) (π/N)² / 3`, ten times more at `δ = 0.3` and outside this
/// bound wherever the rounding is below it (in f64).
fn complex_exponential_offset_is_candans<T: RealField>() {
    let n = LEN as f64;
    let (position, delta) = (37.3, 0.3);
    let step = std::f64::consts::PI / n;
    let bound = ((delta * step).tan() - delta * step) / step + 2.0 * 4.0 * n * T::EPSILON.to_f64();
    let frame = Frame::<T>::new(LEN);
    let spectrum: Vec<Complex<T>> = (0..LEN)
        .map(|p| frame.kernel(T::from_f64(position - p as f64)))
        .collect();
    let estimates = estimate_peaks(&spectrum, &[37], NonZeroUsize::MIN)
        .expect("invariant: a 128-bin frame and an in-range bin are valid");
    let estimate = estimates[0].expect("the exponential's bin holds its peak");
    let error = (estimate.position().to_f64() - position).abs();
    assert!(
        error <= bound,
        "offset error {error:e} exceeds Candan's residual {bound:e}"
    );
}

#[test]
fn complex_exponential_offset_is_candans_for_every_scalar() {
    complex_exponential_offset_is_candans::<f64>();
    complex_exponential_offset_is_candans::<f32>();
}

/// Two tones eight bins apart, the weaker at half the stronger: one pass reads
/// each through the other's full leakage, and three rounds of
/// estimate-and-subtract bring both inside their single-tone bounds widened by
/// the other's residual. The residual's own feedback is geometric with ratio
/// `(π/(2d))² (1 + 1/π)² < ½` at `d = 8`, so the other tone's bound is taken
/// at twice its single-tone value.
fn separated_tones_resolve_after_peeling<T: RealField>() {
    let tones = [
        Tone {
            position: 20.35,
            amplitude: 1.0,
            phase: 1.2,
        },
        Tone {
            position: 28.1,
            amplitude: 0.5,
            phase: -0.6,
        },
    ];
    let d = tones[1].position - tones[0].position;
    let ratio = (std::f64::consts::PI / (2.0 * d) * (1.0 + 1.0 / std::f64::consts::PI)).powi(2);
    assert!(
        ratio < 0.5,
        "the feedback ratio {ratio} must be below one half"
    );
    let spectrum = spectrum::<T>(&tones);
    let bins = [20, 28];
    let rounds = NonZeroUsize::new(3).expect("invariant: three is non-zero");
    let estimates = estimate_peaks(&spectrum, &bins, rounds)
        .expect("invariant: a 128-bin frame and in-range bins are valid");
    let singles = tones.map(|tone| Bound::single::<T>(&tone));
    for (i, j) in [(0, 1), (1, 0)] {
        let doubled = Bound {
            position: 2.0 * singles[j].position,
            amplitude: 2.0 * singles[j].amplitude,
            phase: 2.0 * singles[j].phase,
        };
        let bound = singles[i].with(&tones[i], &tones[j], &doubled);
        let estimate = estimates[i].expect("a separated tone resolves after peeling");
        bound.check(&estimate, &tones[i], "peeled tone");
    }
}

#[test]
fn separated_tones_resolve_after_peeling_for_every_scalar() {
    separated_tones_resolve_after_peeling::<f64>();
    separated_tones_resolve_after_peeling::<f32>();
}

/// Bins that hold no peak of their own are rejected, not misread: a bin two
/// away from a lone tone reads an offset near two bins, and at DC and Nyquist
/// the tone is its own image.
fn bins_without_a_peak_are_rejected<T: RealField>() {
    let tone = Tone {
        position: 10.3,
        amplitude: 1.0,
        phase: 0.7,
    };
    let spectrum = spectrum::<T>(&[tone]);
    let estimates = estimate_peaks(&spectrum, &[12, 8], NonZeroUsize::MIN)
        .expect("invariant: in-range bins are valid");
    assert_eq!(estimates, vec![None, None], "bins two away from the tone");

    let on_dc = spectrum_on_bin::<T>(0.0);
    let on_nyquist = spectrum_on_bin::<T>(LEN as f64 / 2.0);
    for (label, spectrum, bin) in [("DC", on_dc, 0), ("Nyquist", on_nyquist, LEN / 2)] {
        let estimates = estimate_peaks(&spectrum, &[bin], NonZeroUsize::MIN)
            .expect("invariant: in-range bins are valid");
        assert_eq!(estimates, vec![None], "a tone at {label} is its own image");
    }
}

fn spectrum_on_bin<T: RealField>(position: f64) -> Vec<Complex<T>> {
    spectrum::<T>(&[Tone {
        position,
        amplitude: 1.0,
        phase: 0.3,
    }])
}

#[test]
fn bins_without_a_peak_are_rejected_for_every_scalar() {
    bins_without_a_peak_are_rejected::<f64>();
    bins_without_a_peak_are_rejected::<f32>();
}

#[test]
fn unreadable_inputs_are_typed_errors() {
    let short = vec![Complex::new(1.0_f64, 0.0); 2];
    assert_eq!(
        estimate_peaks(&short, &[0], NonZeroUsize::MIN),
        Err(PeakEstimationError::FrameTooShort { len: 2 })
    );
    let spectrum = spectrum::<f64>(&[Tone {
        position: 10.3,
        amplitude: 1.0,
        phase: 0.0,
    }]);
    assert_eq!(
        estimate_peaks(&spectrum, &[10, LEN], NonZeroUsize::MIN),
        Err(PeakEstimationError::PeakOutOfRange { bin: LEN, len: LEN })
    );
    assert_eq!(
        estimate_peaks(&spectrum, &[10, 30, 10], NonZeroUsize::MIN),
        Err(PeakEstimationError::DuplicatePeak { bin: 10 })
    );
    // 2^24 + 1 is the first length f32 rounds.
    let long = vec![Complex::new(0.0_f32, 0.0); (1 << 24) + 1];
    assert_eq!(
        estimate_peaks(&long, &[1], NonZeroUsize::MIN),
        Err(PeakEstimationError::FrameTooLong { len: (1 << 24) + 1 })
    );
}

/// The closed-form kernel against `Σ_{t<N} e^{2πi u t / N}` summed directly,
/// out to positions thousands of bins away. The sum of `N` unit terms carries
/// at most `N · 3ε` of rounding; the closed form, with its phases reduced to a
/// turn, a few `ε` relative to magnitudes up to `N`.
fn closed_form_kernel_matches_the_sum<T: RealField>() {
    let frame = Frame::<T>::new(LEN);
    let n = T::from_f64(LEN as f64);
    let bound = 8.0 * LEN as f64 * T::EPSILON.to_f64();
    for u in [0.0, 0.37, -0.63, 3.5, 49.23, -98.63, 1_279.5, -17_802.37] {
        let u_t = T::from_f64(u);
        let closed = frame.kernel(u_t);
        let summed = (0..LEN).fold(Complex::new(T::ZERO, T::ZERO), |sum, t| {
            let phase = T::TAU * (u_t - u_t.round()) * T::from_f64(t as f64) / n;
            sum + Complex::cis(
                phase + T::TAU * T::from_f64((u.round() * t as f64) % LEN as f64) / n,
            )
        });
        let error = (closed - summed).norm().to_f64();
        assert!(
            error <= bound,
            "kernel at {u}: closed form {closed:?} against the sum {summed:?}, {error:e} > {bound:e}"
        );
    }
}

#[test]
fn closed_form_kernel_matches_the_sum_for_every_scalar() {
    closed_form_kernel_matches_the_sum::<f64>();
    closed_form_kernel_matches_the_sum::<f32>();
}
