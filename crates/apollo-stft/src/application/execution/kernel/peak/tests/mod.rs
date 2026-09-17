//! The estimator against spectra summed directly from their samples, at every
//! shipped scalar, each estimate checked against a bound computed from the
//! model's exact terms and the other estimates it actually subtracted.
//!
//! **Spectra.** Samples and phasors are generated in f64 and narrowed to `T`.
//! A sample `A cos θ` with `|θ| ≤ Θ` carries `(Θ + 2) ε A` from its argument
//! and cosine, a phasor of an angle within `2π` carries `(2π + 2) ε`, the
//! narrowings and the product `3ε`, and recursive summation adds at most
//! `(N − 1) ε Σ|s_t|` (Higham, *Accuracy and Stability of Numerical
//! Algorithms*, §4.2): a bin is within `N ε Σ A (N + Θ + 2π + 7)` of its exact
//! value ([`Model::rounding`]).
//!
//! **Offset.** The estimator reads bins `k − 1, k, k + 1`, each the tone's term
//! `t_m` plus a perturbation of magnitude at most `e_m` (its own image, other
//! tones' residuals, rounding). With `D = 2t₀ − t₋₁ − t₊₁` and
//! `r₀ = (t₋₁ − t₊₁)/D`, the perturbed ratio differs from `r₀` by at most
//! `(e₋₁ + e₊₁ + |r₀| (2e₀ + e₋₁ + e₊₁)) / (|D| (1 − q))`,
//! `q = (2e₀ + e₋₁ + e₊₁) / |D| < 1`, by the triangle inequality. The offset
//! errs by at most the correction factor times that, plus `|c · Re r₀ − δ|`
//! (Candan's residual, `r₀` being exact for the tone alone), plus `8ε` for the
//! ratio's arithmetic and `N ε` for the position's.
//!
//! **Amplitude and phase.** Bin `k` holds `X = a R(δ) + ā I(δ)`, `I(s) =
//! R(−(2k + s))`, and the estimator solves at its offset `s = δ̂`. Expanding
//! `X R̄(s) − X̄ I(s) − a (|R(s)|² − |I(s)|²)` gives
//! `a R̄(s) (R(δ) − R(s)) + ā ((I(δ) − I(s)) R̄(s) + I(s) (R̄(s) − R̄(δ)))
//! − a I(s) (Ī(δ) − Ī(s))`, so
//! `|â − a| ≤ |a| (|R(δ) − R(s)| + |I(δ) − I(s)|) / (|R(s)| − |I(s)|)`, and a
//! bin error `E` adds `E / (|R(s)| − |I(s)|)`. The kernel's product form and
//! the solve's arithmetic add `16ε |a|`. The bound is evaluated at the
//! estimator's own `δ̂`.
//!
//! **Another tone's residual.** Where the estimator subtracted an estimate
//! `(f̂, â)` of a tone `(f, a)`, the bin keeps `|ĉ(p) − c(p)|` for
//! `c(p) = a R(f − p) + ā R(−f − p)`, evaluated exactly here, plus the `16ε`
//! relative rounding of computing `ĉ` in `T`; where it subtracted nothing,
//! the whole `|c(p)|`.

use core::num::NonZeroUsize;
use std::f64::consts::{PI, TAU};

use eunomia::{Complex, Complex64, RealField};

use super::{estimate_peaks, Frame, PeakEstimate};
use crate::domain::contracts::error::PeakEstimationError;

mod scene;

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

    /// The same tone read at its mirror bin `N − k`: position `N − f`, phase
    /// `−φ`.
    fn mirrored(&self, len: usize) -> Self {
        Self {
            position: len as f64 - self.position,
            amplitude: self.amplitude,
            phase: -self.phase,
        }
    }
}

/// A frame length and the f64 kernel the bounds are evaluated with; the
/// kernel is checked against the defining sum in
/// [`closed_form_kernel_matches_the_sum_for_every_scalar`].
struct Model {
    len: usize,
    frame: Frame<f64>,
}

impl Model {
    fn new(len: usize) -> Self {
        Self {
            len,
            frame: Frame::new(len),
        }
    }

    fn kernel(&self, u: f64) -> Complex64 {
        self.frame.kernel(u)
    }

    /// A tone's contribution at position `p`.
    fn contribution(&self, position: f64, a: Complex64, p: f64) -> Complex64 {
        a * self.kernel(position - p) + a.conj() * self.kernel(-position - p)
    }

    /// The bin error spectra summed by [`summed`] carry for tones of total
    /// amplitude `total` whose sample arguments stay within `argument`.
    fn rounding<T: RealField>(&self, total: f64, argument: f64) -> f64 {
        let n = self.len as f64;
        n * T::EPSILON.to_f64() * total * (n + argument + TAU + 7.0)
    }

    /// What `tone` leaves in bin `p` after the estimator subtracted
    /// `estimate`, or subtracted nothing.
    fn residual<T: RealField>(&self, tone: &Tone, estimate: Option<Estimate>, p: f64) -> f64 {
        let truth = self.contribution(tone.position, tone.a(), p);
        match estimate {
            None => truth.norm(),
            Some(estimate) => {
                let subtracted = self.contribution(estimate.position, estimate.a, p);
                let size = estimate.a.norm()
                    * (self.kernel(estimate.position - p).norm()
                        + self.kernel(-estimate.position - p).norm());
                (subtracted - truth).norm() + 16.0 * T::EPSILON.to_f64() * size
            }
        }
    }

    /// The bound on the offset error of `tone` read at bin `k`, with
    /// `extra[m]` bounding bin `k + m − 1` beyond the tone and its own image;
    /// `None` where the perturbation swamps the ratio and no bound follows.
    fn offset_bound<T: RealField>(&self, tone: &Tone, k: usize, extra: [f64; 3]) -> Option<f64> {
        let len = self.len as f64;
        let eps = T::EPSILON.to_f64();
        let a = tone.a();
        let term = |m: usize| a * self.kernel(tone.position - (k as f64 + m as f64 - 1.0));
        let spill: Vec<f64> = (0..3)
            .map(|m| {
                let p = k as f64 + m as f64 - 1.0;
                (a.conj() * self.kernel(-tone.position - p)).norm() + extra[m]
            })
            .collect();
        let denominator = term(1) * 2.0 - term(0) - term(2);
        let exact_ratio = (term(0) - term(2)) / denominator;
        let spill_in_denominator = 2.0 * spill[1] + spill[0] + spill[2];
        let swamp = spill_in_denominator / denominator.norm();
        if swamp >= 1.0 {
            return None;
        }
        let ratio = (spill[0] + spill[2] + exact_ratio.norm() * spill_in_denominator)
            / (denominator.norm() * (1.0 - swamp));
        let correction = (PI / len).tan() / (PI / len);
        let delta = tone.position - k as f64;
        let candan = (correction * exact_ratio.re - delta).abs();
        Some(correction * ratio + candan + 8.0 * eps + len * eps)
    }

    /// The bound on `|â − a|` for the estimator's solve at offset `delta_hat`
    /// with bin error `extra` beyond the tone and its image; `None` where the
    /// image outweighs the tone at that offset.
    fn solve_bound<T: RealField>(
        &self,
        tone: &Tone,
        k: usize,
        delta_hat: f64,
        extra: f64,
    ) -> Option<f64> {
        let delta = tone.position - k as f64;
        let image = |s: f64| self.kernel(-(2.0 * k as f64 + s));
        let (direct_hat, image_hat) = (self.kernel(delta_hat), image(delta_hat));
        let gap = direct_hat.norm() - image_hat.norm();
        if gap <= 0.0 {
            return None;
        }
        let a = tone.a().norm();
        let moved = (self.kernel(delta) - direct_hat).norm() + (image(delta) - image_hat).norm();
        Some((a * moved + extra) / gap + 16.0 * T::EPSILON.to_f64() * a)
    }

    /// Asserts `estimate` of `tone` read at bin `k` within both bounds.
    fn check<T: RealField>(
        &self,
        estimate: Option<PeakEstimate<T>>,
        tone: &Tone,
        k: usize,
        extra: [f64; 3],
        label: &str,
    ) {
        let estimate = Estimate::from(
            estimate.unwrap_or_else(|| panic!("{label}: {tone:?} at bin {k} was rejected")),
        );
        let offset = self
            .offset_bound::<T>(tone, k, extra)
            .unwrap_or_else(|| panic!("{label}: no offset bound for {tone:?} at bin {k}"));
        let offset_error = (estimate.position - tone.position).abs();
        assert!(
            offset_error <= offset,
            "{label}: offset error {offset_error:e} exceeds {offset:e} for {tone:?}"
        );
        let solve = self
            .solve_bound::<T>(tone, k, estimate.position - k as f64, extra[1])
            .unwrap_or_else(|| panic!("{label}: no solve bound for {tone:?} at bin {k}"));
        let solve_error = (estimate.a - tone.a()).norm();
        assert!(
            solve_error <= solve,
            "{label}: |â − a| = {solve_error:e} exceeds {solve:e} for {tone:?}"
        );
    }

    /// Checks every read of `rounds` rounds of estimate-and-subtract over
    /// `tones` at `bins`, following the order the estimator uses: strongest
    /// bin first, each read with the earlier tones' estimates from its own
    /// round and the later tones' from the round before. `bin_error(p)`
    /// bounds bin `p` beyond the tones (rounding, noise). A read whose bound
    /// does not exist is skipped; the last round must resolve every tone.
    fn check_peeled<T: RealField>(
        &self,
        spectrum: &[Complex<T>],
        tones: &[Tone],
        bins: &[usize],
        rounds: usize,
        bin_error: &impl Fn(f64) -> f64,
        label: &str,
    ) {
        let mut order: Vec<usize> = (0..bins.len()).collect();
        let strength = |i: usize| spectrum[bins[i]].norm().to_f64();
        order.sort_by(|&a, &b| strength(b).total_cmp(&strength(a)));
        let by_round: Vec<Vec<Option<PeakEstimate<T>>>> = (0..=rounds)
            .map(|round| match NonZeroUsize::new(round) {
                None => vec![None; bins.len()],
                Some(round) => estimate_peaks(spectrum, bins, round)
                    .expect("invariant: the checked bins are valid"),
            })
            .collect();
        for round in 1..=rounds {
            for (place, &i) in order.iter().enumerate() {
                let seen = |j: usize| {
                    let earlier = order.iter().position(|&o| o == j) < Some(place);
                    by_round[if earlier { round } else { round - 1 }][j].map(Estimate::from)
                };
                let extra: [f64; 3] = core::array::from_fn(|m| {
                    let p = bins[i] as f64 + m as f64 - 1.0;
                    bin_error(p)
                        + (0..tones.len())
                            .filter(|&j| j != i)
                            .map(|j| self.residual::<T>(&tones[j], seen(j), p))
                            .sum::<f64>()
                });
                let resolvable = self.offset_bound::<T>(&tones[i], bins[i], extra).is_some();
                let estimate = by_round[round][i];
                if round == rounds || (resolvable && estimate.is_some()) {
                    self.check::<T>(
                        estimate,
                        &tones[i],
                        bins[i],
                        extra,
                        &format!("{label}, round {round}, tone {i}"),
                    );
                }
            }
        }
    }
}

/// An estimate in f64, where the bounds are evaluated.
#[derive(Clone, Copy, Debug)]
struct Estimate {
    position: f64,
    a: Complex64,
}

impl<T: RealField> From<PeakEstimate<T>> for Estimate {
    fn from(estimate: PeakEstimate<T>) -> Self {
        Self {
            position: estimate.position().to_f64(),
            a: Complex64::from_polar(
                estimate.amplitude().to_f64() / 2.0,
                estimate.phase().to_f64(),
            ),
        }
    }
}

/// The unnormalized DFT of `samples`, summed term by term in `T` from samples
/// and phasors generated in f64.
fn summed<T: RealField>(samples: &[Complex64]) -> Vec<Complex<T>> {
    let n = samples.len();
    (0..n)
        .map(|p| {
            samples
                .iter()
                .enumerate()
                .fold(Complex::new(T::ZERO, T::ZERO), |sum, (t, s)| {
                    let angle = -TAU * ((p * t) % n) as f64 / n as f64;
                    let phasor = Complex::new(T::from_f64(angle.cos()), T::from_f64(angle.sin()));
                    sum + Complex::new(T::from_f64(s.re), T::from_f64(s.im)) * phasor
                })
        })
        .collect()
}

fn real_samples(tones: &[Tone], len: usize) -> Vec<f64> {
    let n = len as f64;
    (0..len)
        .map(|t| {
            tones.iter().fold(0.0, |sum, tone| {
                sum + tone.amplitude * (TAU * tone.position * t as f64 / n + tone.phase).cos()
            })
        })
        .collect()
}

fn real_spectrum<T: RealField>(tones: &[Tone]) -> Vec<Complex<T>> {
    let samples: Vec<Complex64> = real_samples(tones, LEN)
        .into_iter()
        .map(|value| Complex64::new(value, 0.0))
        .collect();
    summed(&samples)
}

/// The largest sample argument `2π f t / N + φ` of `tones` over a frame.
fn argument(tones: &[Tone]) -> f64 {
    tones
        .iter()
        .map(|tone| TAU * tone.position.abs() + tone.phase.abs())
        .fold(0.0, f64::max)
}

const PHASES: [f64; 8] = [-3.1, -2.618, -2.5307, -1.4, -0.2, 0.7, 1.9, 3.1];

/// Lone real tones across the phase circle, near DC and Nyquist where the image
/// is strongest and on an integer bin, each read at its own bin and again at
/// its mirror.
fn single_tones_within_their_bounds<T: RealField>() {
    let model = Model::new(LEN);
    for (position, amplitude) in [
        (1.7, 0.6),
        (10.3, 1.0),
        (31.0, 0.25),
        (40.00003, 1.0),
        (47.49, 3.0),
        (62.2, 1.5),
    ] {
        for phase in PHASES {
            let tone = Tone {
                position,
                amplitude,
                phase,
            };
            let spectrum = real_spectrum::<T>(&[tone]);
            let round = model.rounding::<T>(amplitude, argument(&[tone]));
            for tone in [tone, tone.mirrored(LEN)] {
                let k = tone.position.round() as usize;
                let estimates = estimate_peaks(&spectrum, &[k], NonZeroUsize::MIN)
                    .expect("invariant: a 128-bin frame and an in-range bin are valid");
                model.check::<T>(estimates[0], &tone, k, [round; 3], "single tone");
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
    let model = Model::new(LEN);
    let n = LEN as f64;
    let (position, phase) = (127.4, 0.9);
    let samples: Vec<Complex64> = (0..LEN)
        .map(|t| Complex64::from_polar(1.0, TAU * position * t as f64 / n + phase))
        .collect();
    let spectrum = summed::<T>(&samples);
    let estimates = estimate_peaks(&spectrum, &[127], NonZeroUsize::MIN)
        .expect("invariant: a 128-bin frame and an in-range bin are valid");
    let estimate = estimates[0].expect("the exponential's bin holds its peak");
    let unit = Complex64::from_polar(1.0, phase);
    let term = |m: f64| unit * model.kernel(position - (127.0 + m));
    let denominator = term(0.0) * 2.0 - term(-1.0) - term(1.0);
    let exact_ratio = (term(-1.0) - term(1.0)) / denominator;
    let round = model.rounding::<T>(1.0, TAU * position + phase);
    let ratio = (2.0 * round + exact_ratio.norm() * 4.0 * round)
        / (denominator.norm() * (1.0 - 4.0 * round / denominator.norm()));
    let correction = (PI / n).tan() / (PI / n);
    let eps = T::EPSILON.to_f64();
    let bound =
        correction * ratio + (correction * exact_ratio.re - 0.4).abs() + 8.0 * eps + n * eps;
    let error = (estimate.position().to_f64() - position).abs();
    assert!(error <= bound, "offset error {error:e} exceeds {bound:e}");
}

#[test]
fn complex_exponential_offset_is_candans_for_every_scalar() {
    complex_exponential_offset_is_candans::<f64>();
    complex_exponential_offset_is_candans::<f32>();
}

/// Two tones eight bins apart, the weaker at half the stronger, over three
/// rounds: the first read of the stronger tone carries the weaker one whole
/// (the direct pass), and every later read carries only the residual of the
/// estimate it subtracted.
fn separated_tones_resolve_after_peeling<T: RealField>() {
    let model = Model::new(LEN);
    for (strong_phase, weak_phase) in [(1.2, -0.6), (-2.9, 2.5), (0.4, 0.1)] {
        let tones = [
            Tone {
                position: 20.35,
                amplitude: 1.0,
                phase: strong_phase,
            },
            Tone {
                position: 28.1,
                amplitude: 0.5,
                phase: weak_phase,
            },
        ];
        let spectrum = real_spectrum::<T>(&tones);
        let round = model.rounding::<T>(1.5, argument(&tones));
        model.check_peeled::<T>(&spectrum, &tones, &[20, 28], 3, &|_| round, "peeled pair");
    }
}

#[test]
fn separated_tones_resolve_after_peeling_for_every_scalar() {
    separated_tones_resolve_after_peeling::<f64>();
    separated_tones_resolve_after_peeling::<f32>();
}

/// Bins that hold no peak of their own are rejected, not misread: a bin two
/// away from a lone tone reads an offset near two bins, and a tone read at DC
/// or Nyquist is its own image. For a real signal the offset read there is
/// exactly zero, the spectrum being conjugate-symmetric, and so is the
/// determinant; a complex exponential near Nyquist reads a non-zero offset,
/// and the determinant is zero only up to the rounding of `2k + δ̂`, which the
/// estimator's margin absorbs.
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

    for (label, position, bin) in [
        ("DC", 0.0, 0),
        ("near DC", 0.3, 0),
        ("Nyquist", 64.0, 64),
        ("near Nyquist", 64.2, 64),
        ("near Nyquist, below", 63.7, 64),
    ] {
        for phase in PHASES {
            let spectrum = real_spectrum::<T>(&[Tone {
                position,
                amplitude: 1.0,
                phase,
            }]);
            let estimates = estimate_peaks(&spectrum, &[bin], NonZeroUsize::MIN)
                .expect("invariant: in-range bins are valid");
            assert_eq!(
                estimates,
                vec![None],
                "a tone {label} read at bin {bin} is its own image (phase {phase})"
            );
        }
    }

    let n = LEN as f64;
    for position in [64.2, 63.7, 64.45] {
        for phase in PHASES {
            let samples: Vec<Complex64> = (0..LEN)
                .map(|t| Complex64::from_polar(1.0, TAU * position * t as f64 / n + phase))
                .collect();
            let spectrum = summed::<T>(&samples);
            let estimates = estimate_peaks(&spectrum, &[LEN / 2], NonZeroUsize::MIN)
                .expect("invariant: in-range bins are valid");
            assert_eq!(
                estimates,
                vec![None],
                "an exponential at {position} read at Nyquist has a singular solve (phase {phase})"
            );
        }
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

/// The kernel against `Σ_{t<N} e^{2πi u t / N}` summed directly, at `u` as the
/// scalar holds it: near zero, near integers and multiples of `N` (where a
/// quotient form cancels), and thousands of bins away. The sum of `N` unit
/// phasors, each within `(2π + 3) ε`, carries at most `N (N + 2π + 3) ε`; the
/// product form a few `ε` relative to magnitudes up to `N`.
fn closed_form_kernel_matches_the_sum<T: RealField>() {
    let frame = Frame::<T>::new(LEN);
    let n = LEN as f64;
    let eps = T::EPSILON.to_f64();
    for u in [
        0.0, 3.1e-5, -0.37, 0.63, 1.00003, 3.5, 49.23, 127.9999, 128.00002, -98.63, 1_279.5,
        -17_802.37,
    ] {
        let held = T::from_f64(u);
        let closed = frame.kernel(held);
        let u = held.to_f64();
        let whole = u.round();
        let summed = (0..LEN).fold(Complex::new(T::ZERO, T::ZERO), |sum, t| {
            let turns = (u - whole) * t as f64 / n + (whole * t as f64).rem_euclid(n) / n;
            let angle = TAU * turns;
            sum + Complex::new(T::from_f64(angle.cos()), T::from_f64(angle.sin()))
        });
        let bound = n * (n + TAU + 3.0) * eps + 16.0 * eps * summed.norm().to_f64();
        let error = (closed - summed).norm().to_f64();
        assert!(
            error <= bound,
            "kernel at {u}: {closed:?} against the sum {summed:?}, {error:e} > {bound:e}"
        );
    }
}

#[test]
fn closed_form_kernel_matches_the_sum_for_every_scalar() {
    closed_form_kernel_matches_the_sum::<f64>();
    closed_form_kernel_matches_the_sum::<f32>();
}
