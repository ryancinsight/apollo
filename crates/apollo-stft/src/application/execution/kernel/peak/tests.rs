//! The estimator against spectra summed directly from their samples, at every
//! shipped scalar, with every bound computed from the model's exact terms.
//!
//! **Spectra.** Samples and phasors are generated in f64 and narrowed to `T`,
//! so each term `s_t w_t` of a bin carries at most `3ε |s_t|` (two narrowings
//! and a product); recursive summation adds at most `(N − 1) ε Σ|s_t|`
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, §4.2). A bin is
//! therefore within `(N + 3) ε Σ_t |s_t|` of its exact value.
//!
//! **Offset.** The estimator reads bins `k − 1, k, k + 1`, each the tone's
//! term `t_m` plus a perturbation of magnitude at most `e_m` (its own image,
//! other tones, rounding). With `D = 2t₀ − t₋₁ − t₊₁` and `r₀ = (t₋₁ − t₊₁)/D`,
//! the perturbed ratio differs from `r₀` by at most
//! `(e₋₁ + e₊₁ + |r₀| (2e₀ + e₋₁ + e₊₁)) / (|D| (1 − q))`,
//! `q = (2e₀ + e₋₁ + e₊₁) / |D| < 1`, by the triangle inequality. For one
//! complex exponential `r₀` is exact, so the offset errs by at most the
//! correction factor times that, plus `|c · Re r₀ − δ|` (Candan's residual,
//! computed from `r₀`), plus `8ε` for the ratio's own arithmetic and `N ε` for
//! the position's.
//!
//! **Amplitude and phase.** The solve `g(s)` returns `a` exactly at `s = δ`
//! from an exact bin. Along `s` its logarithm moves by the kernel's: `|d ln|R||`
//! at most `2` on the half-bin and `|d arg R|` at most `π`, with curvature at
//! most `π²` per bin beyond it and the image term's share `2ρ` of the slope
//! (`ρ = |c₀| / |t₀|`), so `|g(δ̂) − a| ≤ |a| (2 + π)(1 + 2ρ + π² e) e` for an
//! offset error `e`. A bin error `E` beyond the image adds at most
//! `E / (|R(δ)| (1 − ρ))`. Then `|Â − A| ≤ 2|â − a|` and
//! `|φ̂ − φ| ≤ (π/2) |â − a| / |a|`.
//!
//! **Another tone's residual.** An estimate off by `(Δf, ΔA, Δφ)` leaves at
//! most `|Δa| (|R(f − p)| + |R(−f − p)|) + |a| Δf (|R'(f − p)| + |R'(−f − p)|)`
//! at position `p`, each kernel bounded over the `Δf`-wide interval (see
//! [`kernel_size`] and [`kernel_slope`]), with `|Δa| ≤ ΔA/2 + (A/2) Δφ`.

use core::num::NonZeroUsize;
use std::f64::consts::{PI, TAU};

use eunomia::{Complex, Complex64, RealField};

use super::{estimate_peaks, Frame, PeakEstimate};
use crate::domain::contracts::error::PeakEstimationError;

const LEN: usize = 128;

#[derive(Clone, Copy, Debug)]
struct Tone {
    position: f64,
    amplitude: f64,
    phase: f64,
}

impl Tone {
    fn a(&self) -> Complex64 {
        Complex64::from_polar(self.amplitude / 2.0, self.phase)
    }

    /// The tone's term and its image's at position `p`.
    fn terms(&self, p: f64) -> (Complex64, Complex64) {
        let kernel = Frame::<f64>::new(LEN);
        let a = self.a();
        (
            a * kernel.kernel(self.position - p),
            a.conj() * kernel.kernel(-self.position - p),
        )
    }

    /// The bound on this tone's contribution error at `p` for estimate errors
    /// `errors`.
    fn residual(&self, errors: &Errors, p: f64) -> f64 {
        let delta_a = errors.amplitude / 2.0 + self.amplitude / 2.0 * errors.phase;
        let width = errors.position;
        let (direct, image) = (self.position - p, -self.position - p);
        delta_a * (kernel_size(direct, width) + kernel_size(image, width))
            + self.amplitude / 2.0
                * width
                * (kernel_slope(direct, width) + kernel_slope(image, width))
    }

    /// The same tone read at its mirror bin `N − k`: position `N − f`, phase
    /// `−φ`.
    fn mirrored(&self) -> Self {
        Self {
            position: LEN as f64 - self.position,
            amplitude: self.amplitude,
            phase: -self.phase,
        }
    }
}

/// The distance from `[u − w, u + w]` to the nearest multiple of `N`, where
/// the kernel peaks.
fn clearance(u: f64, w: f64) -> f64 {
    let n = LEN as f64;
    (u - n * (u / n).round()).abs() - w
}

/// `|R|` on `[u − w, u + w]`: `|sin πu| / |sin(πu/N)| ≤ N / (2d)` at clearance
/// `d ≥ 1` (Jordan, `sin x ≥ 2x/π` on `[0, π/2]`), `N` otherwise.
fn kernel_size(u: f64, w: f64) -> f64 {
    let n = LEN as f64;
    let d = clearance(u, w);
    if d >= 1.0 {
        n / (2.0 * d)
    } else {
        n
    }
}

/// `|R'|` on `[u − w, u + w]`: differentiating
/// `e^{iπu(N−1)/N} sin(πu) / sin(πu/N)` gives at most
/// `π|R| + π/|sin(πu/N)| + (π/N)/sin²(πu/N) ≤ πN/d + πN/(4d²)` at clearance
/// `d ≥ 1`, and `πN` from the defining sum otherwise.
fn kernel_slope(u: f64, w: f64) -> f64 {
    let n = LEN as f64;
    let d = clearance(u, w);
    if d >= 1.0 {
        PI * n / d + PI * n / (4.0 * d * d)
    } else {
        PI * n
    }
}

/// Estimate errors, or bounds on them.
#[derive(Clone, Copy, Debug)]
struct Errors {
    position: f64,
    amplitude: f64,
    phase: f64,
}

/// A bin's rounding for tones of total amplitude `total`.
fn rounding<T: RealField>(total: f64) -> f64 {
    let n = LEN as f64;
    (n + 3.0) * T::EPSILON.to_f64() * n * total
}

/// The bounds of the module doc for `tone` read at bin `k`, with `extra[m]`
/// bounding everything in bin `k + m − 1` beyond the tone and its own image.
fn bound<T: RealField>(tone: &Tone, k: usize, extra: [f64; 3]) -> Errors {
    let len = LEN as f64;
    let eps = T::EPSILON.to_f64();
    let terms: Vec<(Complex64, Complex64)> = (0..3)
        .map(|m| tone.terms(k as f64 + m as f64 - 1.0))
        .collect();
    let term = |m: usize| terms[m].0;
    let spill: Vec<f64> = (0..3).map(|m| terms[m].1.norm() + extra[m]).collect();
    let denominator = term(1) * 2.0 - term(0) - term(2);
    let exact_ratio = (term(0) - term(2)) / denominator;
    let spill_in_denominator = 2.0 * spill[1] + spill[0] + spill[2];
    let swamp = spill_in_denominator / denominator.norm();
    assert!(
        swamp < 1.0,
        "the perturbation {swamp} swamps the ratio of {tone:?} at {k}"
    );
    let ratio = (spill[0] + spill[2] + exact_ratio.norm() * spill_in_denominator)
        / (denominator.norm() * (1.0 - swamp));
    let correction = (PI / len).tan() / (PI / len);
    let delta = tone.position - k as f64;
    let candan = (correction * exact_ratio.re - delta).abs();
    let offset = correction * ratio + candan + 8.0 * eps + len * eps;

    let magnitude = tone.a().norm();
    let direct = term(1).norm() / magnitude;
    let rho = terms[1].1.norm() / term(1).norm();
    let solve = magnitude * (2.0 + PI) * (1.0 + 2.0 * rho + PI * PI * offset) * offset
        + extra[1] / (direct * (1.0 - rho));
    Errors {
        position: offset,
        amplitude: 2.0 * solve,
        phase: PI / 2.0 * solve / magnitude,
    }
}

fn errors_of<T: RealField>(estimate: &PeakEstimate<T>, tone: &Tone) -> Errors {
    let phase = (estimate.phase().to_f64() - tone.phase + PI).rem_euclid(TAU) - PI;
    Errors {
        position: (estimate.position().to_f64() - tone.position).abs(),
        amplitude: (estimate.amplitude().to_f64() - tone.amplitude).abs(),
        phase: phase.abs(),
    }
}

fn check<T: RealField>(estimate: &PeakEstimate<T>, tone: &Tone, bound: &Errors, label: &str) {
    let errors = errors_of(estimate, tone);
    assert!(
        errors.position <= bound.position
            && errors.amplitude <= bound.amplitude
            && errors.phase <= bound.phase,
        "{label}: errors {errors:?} exceed {bound:?} for {tone:?}"
    );
}

/// The unnormalized DFT of `samples`, summed term by term in `T` from samples
/// and phasors generated in f64.
fn summed<T: RealField>(samples: &[Complex64]) -> Vec<Complex<T>> {
    let n = LEN as f64;
    (0..LEN)
        .map(|p| {
            samples
                .iter()
                .enumerate()
                .fold(Complex::new(T::ZERO, T::ZERO), |sum, (t, s)| {
                    let angle = -TAU * ((p * t) % LEN) as f64 / n;
                    let phasor = Complex::new(T::from_f64(angle.cos()), T::from_f64(angle.sin()));
                    sum + Complex::new(T::from_f64(s.re), T::from_f64(s.im)) * phasor
                })
        })
        .collect()
}

fn real_spectrum<T: RealField>(tones: &[Tone]) -> Vec<Complex<T>> {
    let n = LEN as f64;
    let samples: Vec<Complex64> = (0..LEN)
        .map(|t| {
            let value = tones.iter().fold(0.0, |sum, tone| {
                sum + tone.amplitude * (TAU * tone.position * t as f64 / n + tone.phase).cos()
            });
            Complex64::new(value, 0.0)
        })
        .collect();
    summed(&samples)
}

const PHASES: [f64; 8] = [-3.1, -2.618, -2.5307, -1.4, -0.2, 0.7, 1.9, 3.1];

/// Lone real tones across the phase circle, including positions near DC and
/// Nyquist where the image is strongest, and each read again at its mirror bin.
fn single_tones_within_their_bounds<T: RealField>() {
    for (position, amplitude) in [
        (1.7, 0.6),
        (10.3, 1.0),
        (31.0, 0.25),
        (47.49, 3.0),
        (60.8, 1.5),
    ] {
        for phase in PHASES {
            let tone = Tone {
                position,
                amplitude,
                phase,
            };
            let spectrum = real_spectrum::<T>(&[tone]);
            let round = rounding::<T>(amplitude);
            for tone in [tone, tone.mirrored()] {
                let k = tone.position.round() as usize;
                let estimates = estimate_peaks(&spectrum, &[k], NonZeroUsize::MIN)
                    .expect("invariant: a 128-bin frame and an in-range bin are valid");
                let estimate = estimates[0].expect("a lone tone's own bin holds its peak");
                let bound = bound::<T>(&tone, k, [round; 3]);
                check(&estimate, &tone, &bound, "single tone");
            }
        }
    }
}

#[test]
fn single_tones_within_their_bounds_for_every_scalar() {
    single_tones_within_their_bounds::<f64>();
    single_tones_within_their_bounds::<f32>();
}

/// One complex exponential has no image, so its offset error is Candan's
/// residual and rounding alone. Placing it just below `N` makes the estimator
/// read bin `0` as the upper neighbour of bin `N − 1`. Without the correction
/// the offset errs by about `|δ| (1 − δ²) (π/N)² / 3`, ten times Candan's
/// residual at `δ = 0.4`, and outside this bound in f64.
fn complex_exponential_offset_is_candans<T: RealField>() {
    let n = LEN as f64;
    let tone = Tone {
        position: 127.4,
        amplitude: 2.0,
        phase: 0.9,
    };
    let samples: Vec<Complex64> = (0..LEN)
        .map(|t| Complex64::from_polar(1.0, TAU * tone.position * t as f64 / n + tone.phase))
        .collect();
    let spectrum = summed::<T>(&samples);
    let estimates = estimate_peaks(&spectrum, &[127], NonZeroUsize::MIN)
        .expect("invariant: a 128-bin frame and an in-range bin are valid");
    let estimate = estimates[0].expect("the exponential's bin holds its peak");
    let kernel = Frame::<f64>::new(LEN);
    let unit = Complex64::from_polar(1.0, tone.phase);
    let term = |m: f64| unit * kernel.kernel(tone.position - (127.0 + m));
    let denominator = term(0.0) * 2.0 - term(-1.0) - term(1.0);
    let exact_ratio = (term(-1.0) - term(1.0)) / denominator;
    let round = rounding::<T>(1.0);
    let ratio = (2.0 * round + exact_ratio.norm() * 4.0 * round)
        / (denominator.norm() * (1.0 - 4.0 * round / denominator.norm()));
    let correction = (PI / n).tan() / (PI / n);
    let eps = T::EPSILON.to_f64();
    let bound =
        correction * ratio + (correction * exact_ratio.re - 0.4).abs() + 8.0 * eps + n * eps;
    let error = (estimate.position().to_f64() - tone.position).abs();
    assert!(error <= bound, "offset error {error:e} exceeds {bound:e}");
}

#[test]
fn complex_exponential_offset_is_candans_for_every_scalar() {
    complex_exponential_offset_is_candans::<f64>();
    complex_exponential_offset_is_candans::<f32>();
}

/// Two tones, the weaker at half the stronger, resolved by three rounds of
/// estimate-and-subtract. The bound follows the rounds: the stronger tone is
/// read first with the weaker unsubtracted, then each tone is read with the
/// other's latest estimate subtracted, its residual bounded from the other's
/// latest error bound.
fn separated_tones_resolve_after_peeling<T: RealField>() {
    for (weak_phase, strong_phase) in [(-0.6, 1.2), (2.5, -2.9), (0.1, 0.4)] {
        let strong = Tone {
            position: 20.35,
            amplitude: 1.0,
            phase: strong_phase,
        };
        let weak = Tone {
            position: 52.1,
            amplitude: 0.5,
            phase: weak_phase,
        };
        let spectrum = real_spectrum::<T>(&[strong, weak]);
        let bins = [20, 52];
        assert!(
            spectrum[20].norm().to_f64() > spectrum[52].norm().to_f64(),
            "the stronger tone must be estimated first"
        );
        let round = rounding::<T>(1.5);
        let at = |k: usize, m: usize| k as f64 + m as f64 - 1.0;
        let whole = |tone: &Tone, p: f64| {
            let (term, image) = tone.terms(p);
            term.norm() + image.norm()
        };
        let mut strong_bound = bound::<T>(
            &strong,
            20,
            core::array::from_fn(|m| whole(&weak, at(20, m)) + round),
        );
        let mut weak_bound = bound::<T>(
            &weak,
            52,
            core::array::from_fn(|m| strong.residual(&strong_bound, at(52, m)) + round),
        );
        for _ in 1..3 {
            strong_bound = bound::<T>(
                &strong,
                20,
                core::array::from_fn(|m| weak.residual(&weak_bound, at(20, m)) + round),
            );
            weak_bound = bound::<T>(
                &weak,
                52,
                core::array::from_fn(|m| strong.residual(&strong_bound, at(52, m)) + round),
            );
        }
        let rounds = NonZeroUsize::new(3).expect("invariant: three is non-zero");
        let estimates = estimate_peaks(&spectrum, &bins, rounds)
            .expect("invariant: a 128-bin frame and distinct in-range bins are valid");
        for (estimate, tone, bound) in [
            (estimates[0], strong, strong_bound),
            (estimates[1], weak, weak_bound),
        ] {
            let estimate = estimate.expect("a separated tone resolves after peeling");
            check(&estimate, &tone, &bound, "peeled tone");
        }
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
    let spectrum = real_spectrum::<T>(&[tone]);
    let estimates = estimate_peaks(&spectrum, &[12, 8], NonZeroUsize::MIN)
        .expect("invariant: in-range bins are valid");
    assert_eq!(estimates, vec![None, None], "bins two away from the tone");

    for (label, position, bin) in [("DC", 0.0, 0), ("Nyquist", LEN as f64 / 2.0, LEN / 2)] {
        let spectrum = real_spectrum::<T>(&[Tone {
            position,
            amplitude: 1.0,
            phase: 0.3,
        }]);
        let estimates = estimate_peaks(&spectrum, &[bin], NonZeroUsize::MIN)
            .expect("invariant: in-range bins are valid");
        assert_eq!(estimates, vec![None], "a tone at {label} is its own image");
    }
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
    let spectrum = real_spectrum::<f64>(&[Tone {
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
    assert_eq!(
        estimate_peaks(&spectrum, &[10, 30, LEN - 10], NonZeroUsize::MIN),
        Err(PeakEstimationError::MirrorPeak {
            bin: LEN - 10,
            mirror: 10
        })
    );
    // DC and Nyquist are their own mirrors and stay readable.
    let readable = estimate_peaks(&spectrum, &[0, LEN / 2, 10], NonZeroUsize::MIN)
        .expect("self-mirrored bins are not a mirror pair");
    assert!(readable[2].is_some(), "bin 10 holds the tone");
    // `N ε = ½` at `N = 2^22` in f32.
    let long = vec![Complex::new(0.0_f32, 0.0); 1 << 22];
    assert_eq!(
        estimate_peaks(&long, &[1], NonZeroUsize::MIN),
        Err(PeakEstimationError::FrameTooLong { len: 1 << 22 })
    );
}

/// The closed-form kernel against `Σ_{t<N} e^{2πi u t / N}` summed directly,
/// out to positions thousands of bins away, the integer part of `u` reduced
/// exactly, both at `u` as the scalar holds it. The sum of `N` unit terms
/// carries at most `(N + 3) ε N` of rounding; the closed form, with its phases
/// reduced to a turn, a few `ε` relative to magnitudes up to `N`.
fn closed_form_kernel_matches_the_sum<T: RealField>() {
    let frame = Frame::<T>::new(LEN);
    let n = LEN as f64;
    let bound = (n + 8.0) * T::EPSILON.to_f64() * n;
    for u in [0.0, 0.37, -0.63, 3.5, 49.23, -98.63, 1_279.5, -17_802.37] {
        let held = T::from_f64(u);
        let closed = frame.kernel(held);
        let u = held.to_f64();
        let whole = u.round();
        let summed = (0..LEN).fold(Complex::new(T::ZERO, T::ZERO), |sum, t| {
            let turns = (u - whole) * t as f64 / n + (whole * t as f64).rem_euclid(n) / n;
            let angle = TAU * turns;
            sum + Complex::new(T::from_f64(angle.cos()), T::from_f64(angle.sin()))
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
