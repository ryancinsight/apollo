//! The estimator against spectra summed directly from their samples, at every
//! shipped scalar, each estimate checked against a bound computed from the
//! model's exact terms and the other estimates it actually subtracted.
//!
//! **Spectra.** Samples and reduced-angle phasors are generated in f64, then
//! narrowed to `T`. For unit roundoff `u = ε/2`, `γ_m = mu/(1-mu)` composes
//! `m` rounded operations. A sample angle formed by five operations carries
//! `γ₅ Θ`; each elementary sine or cosine is assumed within one epsilon in
//! absolute error. [`Model::rounding`] composes those source errors with the
//! narrowing, complex product, and `N - 1` recursive additions. The elementary
//! function assumption is explicit evidence for these tests, not a portable
//! proof of every platform's libm.
//!
//! **Offset.** The estimator reads bins `k − 1, k, k + 1`, each the tone's term
//! `t_m` including its image, plus a perturbation of magnitude at most `e_m`
//! (other tones' residuals, rounding). With `D = 2t₀ − t₋₁ − t₊₁` and
//! `r₀ = (t₋₁ − t₊₁)/D`, the perturbed ratio differs from `r₀` by at most
//! `(e₋₁ + e₊₁ + |r₀| (2e₀ + e₋₁ + e₊₁)) / (|D| (1 − q))`,
//! `q = (2e₀ + e₋₁ + e₊₁) / |D| < 1`, by the triangle inequality. The offset
//! errs by at most `tan(π/N)/(π/N)` times that (atan is 1-Lipschitz), plus
//! the exact image-induced bias of the inverse relation, plus `8ε` for the
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

mod boundary;
mod scene;
mod tones;

const LEN: usize = 128;

fn gamma(epsilon: f64, operations: usize) -> f64 {
    let unit_roundoff = epsilon / 2.0;
    let accumulated = operations as f64 * unit_roundoff;
    accumulated / (1.0 - accumulated)
}

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
/// [`tones::closed_form_kernel_matches_the_sum_for_every_scalar`].
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
        let source_epsilon = f64::EPSILON;
        let source_unit = source_epsilon / 2.0;
        let source_sample = total
            * ((1.0 + source_unit)
                * (gamma(source_epsilon, 5) * argument + 2.0_f64.sqrt() * source_epsilon)
                + source_unit);
        let source_samples = (1.0 + gamma(source_epsilon, 3)) * n * source_sample
            + gamma(source_epsilon, 3) * n * total;

        let target_epsilon = T::EPSILON.to_f64();
        let target_unit = target_epsilon / 2.0;
        let sample_narrowing = if target_epsilon == source_epsilon {
            0.0
        } else {
            2.0_f64.sqrt() * target_unit
        };
        let samples = (1.0 + sample_narrowing) * source_samples + sample_narrowing * n * total;
        let source_phasor = gamma(source_epsilon, 3) * TAU + 2.0_f64.sqrt() * source_epsilon;
        let phasor = if target_epsilon == source_epsilon {
            source_phasor
        } else {
            let phasor_narrowing = 2.0_f64.sqrt() * target_unit;
            (1.0 + phasor_narrowing) * source_phasor + phasor_narrowing
        };
        // Each component of a complex product is a two-term dot product.
        let multiplication = 2.0_f64.sqrt() * gamma(target_epsilon, 2);
        let product = (1.0 + multiplication) * phasor + multiplication;
        let summation = gamma(target_epsilon, self.len - 1);
        (1.0 + summation) * (1.0 + product) * samples
            + (product + summation * (1.0 + product)) * n * total
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
                // |R'(u)| <= (2π/N) sum(t) = π(N-1). Forming each
                // argument ±f-p incurs at most ε(|f|+|p|) bin error.
                let argument_error = 2.0
                    * estimate.a.norm()
                    * PI
                    * (self.len - 1) as f64
                    * T::EPSILON.to_f64()
                    * (estimate.position.abs() + p.abs());
                (subtracted - truth).norm() + 16.0 * T::EPSILON.to_f64() * size + argument_error
            }
        }
    }

    /// The bound on the offset error of `tone` read at bin `k`, with
    /// `extra[m]` bounding bin `k + m − 1` beyond the tone and its own image;
    /// `None` where the perturbation swamps the ratio and no bound follows.
    fn offset_bound<T: RealField>(&self, tone: &Tone, k: usize, extra: [f64; 3]) -> Option<f64> {
        let len = self.len as f64;
        let eps = T::EPSILON.to_f64();
        let term = |m: usize| self.contribution(tone.position, tone.a(), k as f64 + m as f64 - 1.0);
        let spill = extra;
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
        let step = PI / len;
        let closure = ((exact_ratio.re * step.tan()).atan() / step - delta).abs();
        Some(correction * ratio + closure + 8.0 * eps + len * eps)
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
        // The public position rounds bin+delta; recovering delta for this
        // oracle adds at most ε(k+1). Bound both kernel derivatives by πN.
        let position_rounding =
            2.0 * PI * self.len as f64 * T::EPSILON.to_f64() * (k + 1) as f64 * a;
        Some((a * moved + extra + position_rounding) / gap + 16.0 * T::EPSILON.to_f64() * a)
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
                let resolvable = self
                    .offset_bound::<T>(&tones[i], bins[i], extra)
                    .is_some_and(|bound| (tones[i].position - bins[i] as f64).abs() + bound <= 0.5);
                let estimate = by_round[round][i];
                if round == rounds || resolvable {
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
