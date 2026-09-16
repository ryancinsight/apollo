//! Spike: sub-bin spectral peak estimators against a three-tone oracle
//! (`backlog.md#apollo-spectral-peak-estimation-spike`, ADR 0066).
//!
//! Test-only candidates, each a resolved published estimator of the
//! fractional bin offset `δ` of a tone from its peak DFT bin `k`:
//!
//! - Jacobsen and Kootsookos, "Fast, accurate frequency estimators", IEEE
//!   Signal Processing Magazine 24(3), 2007, equations (3), (4) and (5) with
//!   the Hann constants `P = 1.36` and `Q = 0.55` of its Table 1;
//! - Candan, "A method for fine resolution frequency estimation from three
//!   DFT samples", IEEE SPL 18(6), 2011: (3) scaled by `tan(π/N) / (π/N)`;
//!   and the 2013 closure `arctan(δ · π/N) · N/π`;
//! - Aboutanios and Mulgrew, "Iterative frequency estimation by
//!   interpolation on Fourier coefficients", IEEE Trans. Signal Processing
//!   53(4), 2005, Table I, Alg1: the DFT evaluated at `k + δ̂ ± 1/2`,
//!   `δ̂ ← δ̂ + ½ Re[(X₊ + X₋) / (X₊ − X₋)]`, converged in two iterations
//!   (Section III-B), at 1.0147 times the asymptotic Cramér–Rao bound
//!   (Section III-A, equation (19) at `δ = 0`).
//!
//! Every estimator reads bin `k` as the peak and is valid on `|δ| ≤ ½`; an
//! offset outside that half-bin means the bin holds no peak (the middle
//! tone under the outer tones' leakage) and the estimate is rejected.
//!
//! Amplitude and phase follow from the window's spectral kernel at the
//! estimated offset. A real tone `A cos(2π f t / f_s + φ)` is
//! `a e^{2πi (k+δ) t / N} + ā e^{−2πi (k+δ) t / N}` with `a = (A/2) e^{iφ}`
//! and `δ = f N / f_s − k`, so with `W̃(u) = Σ_t w[t] e^{2πi u t / N}` bin `k`
//! holds `X_k = a W̃(δ) + ā W̃(−(2k + δ))`; the second term is the
//! negative-frequency image, `2e-5` of the first for this scene, and
//! solving the pair `X_k`, `X̄_k` for `a` gives
//! `a = (X_k W̄̃(δ) − X̄_k W̃(−(2k + δ))) / (|W̃(δ)|² − |W̃(−(2k + δ))|²)`,
//! `A = 2|a|`, `φ = arg a`. The kernels are closed forms — the rectangular
//! `R(u) = (e^{2πiu} − 1) / (e^{2πiu/N} − 1)` and the symmetric Hann
//! `½ − ½ cos(2π t / (N − 1))` as `½ R(u) − ¼ R(u + N/(N−1)) − ¼ R(u − N/(N−1))`
//! — checked against the direct sum for the windows in use.
//!
//! Multi-tone scenes add estimate-and-subtract: tones are estimated
//! strongest first on the residual of the earlier estimates, then every tone
//! is re-estimated on the residual of all the others, for a fixed number of
//! rounds. A tone estimated with relative errors `ε` leaves a residual of
//! order `ε` that leaks into a bin `u` away at the kernel ratio
//! `|W̃(u)| / |W̃(δ)|`, so each round divides the interference by the
//! previous round's error. The DFT is linear, so a residual is read at any
//! position as the raw spectrum minus the subtracted tones' kernel terms.
//!
//! The oracle is the paper's three-tone scene (Henry, Science Talks 4, 2022,
//! Figs. 6-24): 48 kHz, 48 000 samples, tones near 8950 Hz about 50 Hz
//! apart, the outer two at 1 V and the middle at 1e-6 V, white noise of
//! 1e-10 V, random phases per seed.

use crate::application::execution::kernel::hann::hann_window;
use eunomia::Complex64;
use std::f64::consts::{PI, TAU};

const SAMPLE_RATE: f64 = 48_000.0;
const LEN: usize = 48_000;
const BIN_WIDTH: f64 = SAMPLE_RATE / LEN as f64;
const NOISE_STD: f64 = 1e-10;
const SEEDS: u64 = 8;
/// Aboutanios–Mulgrew iterations: Section III-B of the paper shows the
/// residual is below the Cramér–Rao bound after two.
const REFINEMENT_ITERATIONS: usize = 2;
/// Estimate-and-subtract rounds reported; the interference model above
/// puts the middle tone inside the noise floor by the third.
const PEEL_ROUNDS: usize = 3;
/// Frequencies and amplitudes of the scene: the outer tones a fraction of a
/// bin off integer frequencies so every estimator has an offset to find.
const TONES: [(f64, f64); 3] = [(8901.37, 1.0), (8950.61, 1e-6), (9000.23, 1.0)];

#[derive(Clone, Copy, Debug)]
enum Window {
    Rectangular,
    Hann,
}

impl Window {
    fn samples(self) -> Vec<f64> {
        match self {
            Self::Rectangular => vec![1.0; LEN],
            Self::Hann => hann_window(LEN).iter().copied().collect(),
        }
    }

    /// `W̃(u) = Σ_t w[t] e^{2πi u t / N}` in closed form.
    fn kernel(self, u: f64) -> Complex64 {
        match self {
            Self::Rectangular => rectangular_kernel(u),
            Self::Hann => {
                let shift = LEN as f64 / (LEN - 1) as f64;
                0.5 * rectangular_kernel(u)
                    - 0.25 * rectangular_kernel(u + shift)
                    - 0.25 * rectangular_kernel(u - shift)
            }
        }
    }
}

/// `R(u) = Σ_{t<N} e^{2πi u t / N} = (e^{2πiu} − 1) / (e^{2πiu/N} − 1)`,
/// `N` at the multiples of `N` where the ratio is indeterminate.
fn rectangular_kernel(u: f64) -> Complex64 {
    let one = Complex64::new(1.0, 0.0);
    let denominator = Complex64::from_polar(1.0, TAU * u / LEN as f64) - one;
    if denominator.norm() < f64::EPSILON {
        return Complex64::new(LEN as f64, 0.0);
    }
    (Complex64::from_polar(1.0, TAU * u) - one) / denominator
}

/// `W̃(u)` by the direct sum over the window samples: the calibration of
/// the closed forms.
fn summed_kernel(window: &[f64], u: f64) -> Complex64 {
    let step = TAU * u / LEN as f64;
    window
        .iter()
        .enumerate()
        .map(|(t, &w)| w * Complex64::from_polar(1.0, step * t as f64))
        .sum()
}

/// The DFT of `samples` at the fractional bin `p`: `Σ_t s[t] e^{−2πi p t / N}`.
fn dft_at(samples: &[f64], p: f64) -> Complex64 {
    let step = -TAU * p / LEN as f64;
    samples
        .iter()
        .enumerate()
        .map(|(t, &s)| s * Complex64::from_polar(1.0, step * t as f64))
        .sum()
}

#[derive(Clone, Copy, Debug)]
struct Tone {
    frequency: f64,
    amplitude: f64,
    phase: f64,
}

impl Tone {
    /// The tone's sample at index `t` of the scene.
    fn sample(&self, t: usize) -> f64 {
        self.amplitude * (TAU * self.frequency * t as f64 / SAMPLE_RATE + self.phase).cos()
    }

    /// The integer bin nearest the tone: the search a caller would run over
    /// the magnitude spectrum, restricted to the tone's neighbourhood so the
    /// low-amplitude tone is scored at its own bin rather than at a
    /// neighbour's leakage.
    fn nominal_bin(&self) -> usize {
        (self.frequency / BIN_WIDTH).round() as usize
    }

    /// The tone's contribution at position `p` of the windowed spectrum:
    /// `a W̃(f − p) + ā W̃(−f − p)` for `f` the frequency in bins.
    fn spectrum_at(&self, window: Window, p: f64) -> Complex64 {
        let a = Complex64::from_polar(self.amplitude / 2.0, self.phase);
        let f = self.frequency / BIN_WIDTH;
        a * window.kernel(f - p) + a.conj() * window.kernel(-f - p)
    }

    /// Cramér–Rao bound on the frequency standard deviation of the tone in
    /// the scene's noise: Aboutanios–Mulgrew (3) for the complex exponential,
    /// `σ_f² = 6 f_s² / ((2π)² ρ N (N² − 1))` with `ρ = A² / σ²`; a real tone
    /// in real noise has half the Fisher information of the complex one at
    /// half its noise per band, four times the variance (ADR 0066).
    fn frequency_bound(&self) -> f64 {
        let n = LEN as f64;
        let rho = self.amplitude * self.amplitude / (NOISE_STD * NOISE_STD);
        SAMPLE_RATE / TAU * (24.0 / (rho * n * (n * n - 1.0))).sqrt()
    }
}

/// Gaussian noise and phases from a seeded xorshift, Box–Muller.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gaussian(&mut self) -> f64 {
        let u = self.uniform().max(f64::MIN_POSITIVE);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (TAU * v).cos()
    }
}

fn scene(seed: u64) -> (Vec<Tone>, Vec<f64>) {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let tones: Vec<Tone> = TONES
        .iter()
        .map(|&(frequency, amplitude)| Tone {
            frequency,
            amplitude,
            phase: TAU * rng.uniform() - PI,
        })
        .collect();
    let signal: Vec<f64> = (0..LEN)
        .map(|t| tones.iter().map(|tone| tone.sample(t)).sum::<f64>() + NOISE_STD * rng.gaussian())
        .collect();
    (tones, signal)
}

/// The scene's signal analysed under one window.
struct Analysis {
    window: Window,
    windowed: Vec<f64>,
    spectrum: Vec<Complex64>,
}

impl Analysis {
    fn new(window: Window, signal: &[f64]) -> Self {
        let windowed: Vec<f64> = signal
            .iter()
            .zip(window.samples())
            .map(|(s, w)| s * w)
            .collect();
        let spectrum = apollo_fft::fft_1d_slice::<f64>(&windowed);
        Self {
            window,
            windowed,
            spectrum,
        }
    }

    /// The raw windowed spectrum at an integer or fractional position.
    fn raw_at(&self, p: f64) -> Complex64 {
        if p.fract() == 0.0 {
            self.spectrum[p as usize]
        } else {
            dft_at(&self.windowed, p)
        }
    }

    /// The windowed spectrum with `removed` tones subtracted, by linearity.
    fn residual_at<'a>(&self, p: f64, removed: impl Iterator<Item = &'a Tone>) -> Complex64 {
        self.raw_at(p)
            - removed
                .map(|tone| tone.spectrum_at(self.window, p))
                .sum::<Complex64>()
    }
}

#[derive(Clone, Copy, Debug)]
enum Estimator {
    /// Jacobsen (3), rectangular window.
    JacobsenComplex,
    /// Candan 2011: Jacobsen (3) scaled by `tan(π/N) / (π/N)`.
    Candan2011,
    /// Candan 2013: the 2011 estimate closed by `arctan(δ π/N) N/π`.
    Candan2013,
    /// Aboutanios–Mulgrew Alg1, rectangular window, two iterations.
    AboutaniosMulgrew,
    /// Jacobsen (5) on the Hann window, `Q = 0.55`.
    JacobsenHann,
    /// Jacobsen (4) on Hann magnitudes, `P = 1.36`.
    ParabolicHann,
}

const ESTIMATORS: [Estimator; 6] = [
    Estimator::JacobsenComplex,
    Estimator::Candan2011,
    Estimator::Candan2013,
    Estimator::AboutaniosMulgrew,
    Estimator::JacobsenHann,
    Estimator::ParabolicHann,
];

impl Estimator {
    fn window(self) -> Window {
        match self {
            Self::JacobsenHann | Self::ParabolicHann => Window::Hann,
            Self::JacobsenComplex
            | Self::Candan2011
            | Self::Candan2013
            | Self::AboutaniosMulgrew => Window::Rectangular,
        }
    }

    fn windowed(self) -> bool {
        matches!(self.window(), Window::Hann)
    }

    /// The three-bin complex-ratio family on the rectangular window.
    fn three_bin(self) -> bool {
        matches!(
            self,
            Self::JacobsenComplex | Self::Candan2011 | Self::Candan2013
        )
    }

    fn label(self) -> &'static str {
        match self {
            Self::JacobsenComplex => "Jacobsen (3), rectangular",
            Self::Candan2011 => "Candan 2011, rectangular",
            Self::Candan2013 => "Candan 2013, rectangular",
            Self::AboutaniosMulgrew => "Aboutanios-Mulgrew, rectangular",
            Self::JacobsenHann => "Jacobsen (5), Hann",
            Self::ParabolicHann => "Jacobsen (4), Hann magnitudes",
        }
    }

    /// The fractional offset of the peak from bin `k`, reading the spectrum
    /// through `read`.
    fn offset(self, read: &impl Fn(f64) -> Complex64, k: usize) -> f64 {
        let n = LEN as f64;
        let (before, peak, after) = (read(k as f64 - 1.0), read(k as f64), read(k as f64 + 1.0));
        match self {
            Self::JacobsenComplex => ((before - after) / (2.0 * peak - before - after)).re,
            Self::Candan2011 => Self::JacobsenComplex.offset(read, k) * (PI / n).tan() / (PI / n),
            Self::Candan2013 => {
                let first = Self::Candan2011.offset(read, k);
                (first * PI / n).atan() * n / PI
            }
            Self::AboutaniosMulgrew => {
                let mut delta = 0.0;
                for _ in 0..REFINEMENT_ITERATIONS {
                    let upper = read(k as f64 + delta + 0.5);
                    let lower = read(k as f64 + delta - 0.5);
                    delta += 0.5 * ((upper + lower) / (upper - lower)).re;
                }
                delta
            }
            Self::JacobsenHann => (0.55 * (before - after) / (2.0 * peak + before + after)).re,
            Self::ParabolicHann => {
                1.36 * (after.norm() - before.norm()) / (peak.norm() + before.norm() + after.norm())
            }
        }
    }

    /// The tone at bin `k` of the spectrum read through `read`: offset by
    /// this estimator, amplitude and phase from the window kernel and its
    /// image at that offset; `None` when the offset leaves the half-bin, so
    /// bin `k` holds no peak.
    fn tone(self, read: &impl Fn(f64) -> Complex64, k: usize) -> Option<Tone> {
        let delta = self.offset(read, k);
        if delta.abs() > 0.5 {
            return None;
        }
        let window = self.window();
        let direct = window.kernel(delta);
        let image = window.kernel(-(2.0 * k as f64 + delta));
        let peak = read(k as f64);
        let a =
            (peak * direct.conj() - peak.conj() * image) / (direct.norm_sqr() - image.norm_sqr());
        Some(Tone {
            frequency: (k as f64 + delta) * BIN_WIDTH,
            amplitude: 2.0 * a.norm(),
            phase: a.arg(),
        })
    }

    /// Every tone estimated from the raw spectrum of the scene; `None`
    /// where the estimate is rejected.
    fn direct(self, tones: &[Tone], analysis: &Analysis) -> Vec<Option<Errors>> {
        let read = |p| analysis.raw_at(p);
        tones
            .iter()
            .map(|tone| {
                self.tone(&read, tone.nominal_bin())
                    .map(|estimate| Errors::between(&estimate, tone))
            })
            .collect()
    }

    /// Estimate-and-subtract over `PEEL_ROUNDS` rounds; the errors after
    /// each round, `None` where the round rejected the tone (nothing is
    /// subtracted for it).
    fn peeled(self, tones: &[Tone], analysis: &Analysis) -> Vec<Vec<Option<Errors>>> {
        let bins: Vec<usize> = tones.iter().map(Tone::nominal_bin).collect();
        let order = {
            let mut order: Vec<usize> = (0..tones.len()).collect();
            order.sort_by(|&a, &b| {
                analysis.spectrum[bins[b]]
                    .norm()
                    .total_cmp(&analysis.spectrum[bins[a]].norm())
            });
            order
        };
        let mut estimates: Vec<Option<Tone>> = vec![None; tones.len()];
        (0..PEEL_ROUNDS)
            .map(|_| {
                for &i in &order {
                    let read = |p| {
                        analysis.residual_at(
                            p,
                            estimates
                                .iter()
                                .enumerate()
                                .filter(|&(j, _)| j != i)
                                .filter_map(|(_, e)| e.as_ref()),
                        )
                    };
                    estimates[i] = self.tone(&read, bins[i]);
                }
                estimates
                    .iter()
                    .zip(tones)
                    .map(|(estimate, tone)| {
                        estimate
                            .as_ref()
                            .map(|estimate| Errors::between(estimate, tone))
                    })
                    .collect()
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Errors {
    frequency: f64,
    amplitude: f64,
    phase: f64,
}

impl Errors {
    fn between(estimate: &Tone, truth: &Tone) -> Self {
        let phase = (estimate.phase - truth.phase + PI).rem_euclid(TAU) - PI;
        Self {
            frequency: (estimate.frequency - truth.frequency).abs(),
            amplitude: (estimate.amplitude - truth.amplitude).abs(),
            phase: phase.abs(),
        }
    }
}

/// Errors over the seeds: the count of accepted estimates and their mean
/// absolute errors.
#[derive(Clone, Copy, Debug, Default)]
struct Outcome {
    accepted: usize,
    errors: Errors,
}

impl Outcome {
    fn accumulate(&mut self, errors: Option<Errors>) {
        if let Some(errors) = errors {
            self.accepted += 1;
            self.errors.frequency += errors.frequency;
            self.errors.amplitude += errors.amplitude;
            self.errors.phase += errors.phase;
        }
    }

    fn mean(&self) -> Errors {
        let n = self.accepted.max(1) as f64;
        Errors {
            frequency: self.errors.frequency / n,
            amplitude: self.errors.amplitude / n,
            phase: self.errors.phase / n,
        }
    }

    fn row(&self, label: &str, round: &str, tone: usize) {
        let mean = self.mean();
        eprintln!(
            "| {label} | {round} | {} | {}/{SEEDS} | {:.2e} | {:.2e} | {:.2e} |",
            tone + 1,
            self.accepted,
            mean.frequency,
            mean.amplitude,
            mean.phase
        );
    }
}

/// Relative interference at the positions an estimator reads around an
/// outer tone's bin, `k − 1 ..= k + 1` in half-bin steps: the other outer
/// tone's contribution over the tone's own, through the rectangular kernel.
/// The middle tone and the noise are six orders below and ignored.
fn rectangular_interference(k: usize, own: (f64, f64), other: (f64, f64)) -> f64 {
    [-1.0, -0.5, 0.0, 0.5, 1.0]
        .into_iter()
        .map(|p| {
            let position = k as f64 + p;
            let contribution = |(frequency, amplitude): (f64, f64)| {
                amplitude / 2.0
                    * Window::Rectangular
                        .kernel(frequency / BIN_WIDTH - position)
                        .norm()
            };
            contribution(other) / contribution(own)
        })
        .fold(0.0, f64::max)
}

/// Bin magnitudes at the middle tone's bin through the Hann window: the
/// outer tones' leakage against the tone's own contribution.
fn hann_middle_bin() -> (f64, f64) {
    let middle_bin = (TONES[1].0 / BIN_WIDTH).round();
    let contribution = |(frequency, amplitude): (f64, f64)| {
        amplitude / 2.0
            * Window::Hann
                .kernel(frequency / BIN_WIDTH - middle_bin)
                .norm()
    };
    (
        contribution(TONES[0]) + contribution(TONES[2]),
        contribution(TONES[1]),
    )
}

#[test]
fn closed_form_kernels_match_the_direct_sums() {
    // The direct sum of `N` unit terms at angles up to `2π · 17 802` carries
    // the angle rounding of `1e5` rad, `1.5e-11` per term, coherently at
    // worst: `N · 1.5e-11 ≈ 7e-7`; the bound is one decade above.
    let bound = 1e-5;
    for window in [Window::Rectangular, Window::Hann] {
        let samples = window.samples();
        for u in [0.0, 0.37, -0.63, 49.23, -98.63, -17_802.37] {
            let closed = window.kernel(u);
            let summed = summed_kernel(&samples, u);
            assert!(
                (closed - summed).norm() <= bound,
                "{window:?} kernel at {u}: closed form {closed:?} against the sum {summed:?}"
            );
        }
    }
}

#[test]
fn resolved_estimators_against_the_three_tone_scene() {
    let mut direct = vec![vec![Outcome::default(); TONES.len()]; ESTIMATORS.len()];
    let mut peeled =
        vec![vec![vec![Outcome::default(); TONES.len()]; PEEL_ROUNDS]; ESTIMATORS.len()];
    for seed in 1..=SEEDS {
        let (tones, signal) = scene(seed);
        let analyses = [
            Analysis::new(Window::Rectangular, &signal),
            Analysis::new(Window::Hann, &signal),
        ];
        for (e, &estimator) in ESTIMATORS.iter().enumerate() {
            let analysis = &analyses[usize::from(estimator.windowed())];
            for (t, errors) in estimator.direct(&tones, analysis).into_iter().enumerate() {
                direct[e][t].accumulate(errors);
            }
            for (round, errors) in estimator.peeled(&tones, analysis).into_iter().enumerate() {
                for (t, errors) in errors.into_iter().enumerate() {
                    peeled[e][round][t].accumulate(errors);
                }
            }
        }
    }
    eprintln!(
        "| estimator | pass | tone | accepted | frequency (Hz) | amplitude (V) | phase (rad) |"
    );
    eprintln!("|---|---|---|---|---|---|---|");
    for (e, &estimator) in ESTIMATORS.iter().enumerate() {
        for (t, outcome) in direct[e].iter().enumerate() {
            outcome.row(estimator.label(), "direct", t);
        }
        for (round, outcomes) in peeled[e].iter().enumerate() {
            for (t, outcome) in outcomes.iter().enumerate() {
                outcome.row(estimator.label(), &format!("peel {}", round + 1), t);
            }
        }
    }
    let (tones, _) = scene(1);
    for (t, tone) in tones.iter().enumerate() {
        eprintln!(
            "Cramér-Rao frequency bound, tone {}: {:.2e} Hz",
            t + 1,
            tone.frequency_bound()
        );
    }
    let (leakage, middle) = hann_middle_bin();
    eprintln!("Hann bin at the middle tone: outer-tone leakage {leakage:.2e} against the tone's own {middle:.2e}");

    // Direct rectangular estimates of the outer tones err by the other outer
    // tone's interference `ρ` at the positions read (the middle tone and the
    // noise are six orders below it). Interference `c` added to the samples
    // an estimator divides moves the offset by at most `ρ` bins: Jacobsen (3)
    // and Candan read `(c_{k−1} − c_{k+1}) / (2X_k − X_{k−1} − X_{k+1})`, at
    // most `δ ρ`; Aboutanios–Mulgrew reads `(c₊ + c₋) / (X₊ − X₋)`, at most
    // `(½ + |δ|) ρ`. The amplitude `2|X_k + c| / |W̃(δ̂)|` errs by `ρ` from
    // `c` and by `2 ρ` through the kernel slope (`|d ln|W̃| / dδ| ≤ 2` for
    // the rectangular kernel on `|δ| ≤ ½`); the phase by `ρ` from `c` and by
    // `π ρ` through the kernel phase slope `π (N − 1) / N`. The image term
    // of the solve is exact, so it adds nothing.
    let outer_bins = [tones[0].nominal_bin(), tones[2].nominal_bin()];
    let rho = rectangular_interference(outer_bins[0], TONES[0], TONES[2])
        .max(rectangular_interference(outer_bins[1], TONES[2], TONES[0]));
    eprintln!("Rectangular interference between the outer tones: {rho:.2e}");
    for (e, &estimator) in ESTIMATORS.iter().enumerate() {
        if estimator.windowed() {
            continue;
        }
        for t in [0, 2] {
            let outcome = direct[e][t];
            let errors = outcome.mean();
            assert!(
                outcome.accepted == SEEDS as usize
                    && errors.frequency <= rho * BIN_WIDTH
                    && errors.amplitude <= (1.0 + 2.0) * rho * TONES[t].1
                    && errors.phase <= (1.0 + PI) * rho,
                "{} tone {}: {errors:?} exceeds the interference bound {rho:.2e}",
                estimator.label(),
                t + 1
            );
        }
    }
    // The middle tone is below the outer tones' leakage at its bin even
    // through the Hann window, so every direct reading there is leakage, not
    // the tone: the estimate is rejected or its amplitude error exceeds the
    // tone's whole amplitude.
    assert!(
        leakage > middle,
        "the middle tone would be above the Hann leakage: {leakage:.2e} against {middle:.2e}"
    );
    for (e, &estimator) in ESTIMATORS.iter().enumerate() {
        let outcome = direct[e][1];
        assert!(
            outcome.accepted == 0 || outcome.mean().amplitude > TONES[1].1,
            "{} resolves the middle tone directly: {outcome:?}",
            estimator.label()
        );
    }
    // Estimate-and-subtract with the three-bin family resolves every tone in
    // every seed by the last round: resolved means the estimate is dominated
    // by the tone rather than by leakage, one decade inside the direct
    // failure above (amplitude within a tenth, offset within a tenth of a
    // bin, phase within a tenth of a radian). The middle tone is then
    // noise-limited: within a decade of its Cramér–Rao bound. The outer
    // tones are limited by their own image, whose leakage `c` differs
    // between bins `k − 1` and `k + 1` by `(2π/N) / sin(π (2k + δ) / N)` of
    // itself while the ratio's denominator `2X_k − X_{k−1} − X_{k+1}` is
    // `2 / (δ (1 − δ²))` of the tone term `b`, and
    // `|c| / |b| = (π/N) / sin(π (2k + δ) / N)` for the rectangular kernel
    // `e^{iπu(N−1)/N} sin(πu) / sin(πu/N)`: the offset errs by at most
    // `δ (1 − δ²) (π/N)² / sin²(π (2k + δ) / N)` bins, plus Jacobsen (3)'s
    // finite-length bias `δ (π/N)² / 3`, which Candan's factor removes.
    for (e, &estimator) in ESTIMATORS.iter().enumerate() {
        if !estimator.three_bin() {
            continue;
        }
        for (t, outcome) in peeled[e][PEEL_ROUNDS - 1].iter().enumerate() {
            let errors = outcome.mean();
            assert!(
                outcome.accepted == SEEDS as usize
                    && errors.frequency <= 0.1 * BIN_WIDTH
                    && errors.amplitude <= 0.1 * TONES[t].1
                    && errors.phase <= 0.1,
                "peeled {} leaves tone {} unresolved: {outcome:?}",
                estimator.label(),
                t + 1
            );
        }
        let middle = peeled[e][PEEL_ROUNDS - 1][1].mean();
        assert!(
            middle.frequency <= 10.0 * tones[1].frequency_bound(),
            "peeled {} leaves the middle tone above a decade of its bound: {middle:?}",
            estimator.label()
        );
        for t in [0, 2] {
            let tone = &tones[t];
            let k = tone.nominal_bin();
            let delta = tone.frequency / BIN_WIDTH - k as f64;
            let n = LEN as f64;
            let image_floor = delta.abs() * (1.0 - delta * delta) * (PI / n).powi(2)
                / (PI * (2.0 * k as f64 + delta) / n).sin().powi(2)
                + delta.abs() * (PI / n).powi(2) / 3.0;
            let errors = peeled[e][PEEL_ROUNDS - 1][t].mean();
            assert!(
                errors.frequency <= image_floor * BIN_WIDTH,
                "peeled {} tone {} exceeds its image floor {image_floor:.2e}: {errors:?}",
                estimator.label(),
                t + 1
            );
        }
    }
}
