//! `estimate_peaks` against the three-tone scene of ADR 0066
//! (`docs/adr/0066-sub-bin-peak-estimation.md`).
//!
//! The scene is Henry, Science Talks 4 (2022), Figs. 6-24: 48 kHz, 48 000
//! samples (a 1 Hz bin), three tones near 8950 Hz about 50 Hz apart, the outer
//! two at 1 V and the middle at 1e-6 V, white noise of 1e-10 V, random phases
//! per seed, mean absolute errors over eight seeds.
//!
//! It runs in f64 only. Its noise and its middle tone sit below what f32
//! resolves in a 1 V spectrum of this length: a bin of the outer tones carries
//! about `N ε₃₂ A` = 6e-3 of rounding against the middle tone's `A N / π` of
//! 1.5e-2, so the scene is not representable there. The per-scalar bounds are
//! the unit tests of `kernel::peak`.

use core::num::NonZeroUsize;
use std::f64::consts::{PI, TAU};

use apollo_stft::estimate_peaks;
use eunomia::Complex64;

const SAMPLE_RATE: f64 = 48_000.0;
const LEN: usize = 48_000;
const BIN_WIDTH: f64 = SAMPLE_RATE / LEN as f64;
const NOISE_STD: f64 = 1e-10;
const SEEDS: u64 = 8;
/// Estimate-and-subtract rounds: ADR 0066 measured the middle tone inside the
/// noise floor by the third.
const PEEL_ROUNDS: usize = 3;
const TONES: [(f64, f64); 3] = [(8901.37, 1.0), (8950.61, 1e-6), (9000.23, 1.0)];

#[derive(Clone, Copy, Debug)]
struct Tone {
    frequency: f64,
    amplitude: f64,
    phase: f64,
}

impl Tone {
    fn sample(&self, t: usize) -> f64 {
        self.amplitude * (TAU * self.frequency * t as f64 / SAMPLE_RATE + self.phase).cos()
    }

    /// The integer bin nearest the tone, the search a caller runs over the
    /// magnitude spectrum.
    fn bin(&self) -> usize {
        (self.frequency / BIN_WIDTH).round() as usize
    }

    /// Cramér–Rao bound on the frequency standard deviation of a real tone in
    /// real white noise: four times Aboutanios–Mulgrew (3) for the complex
    /// exponential, `σ_f = (f_s / 2π) √(24 / (ρ N (N² − 1)))`, `ρ = A² / σ²`.
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

fn scene(seed: u64) -> (Vec<Tone>, Vec<Complex64>) {
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
    (tones, apollo_fft::fft_1d_slice::<f64>(&signal))
}

#[derive(Clone, Copy, Debug, Default)]
struct Errors {
    frequency: f64,
    amplitude: f64,
    phase: f64,
}

/// Mean absolute errors of the accepted estimates, and how many were.
#[derive(Clone, Copy, Debug, Default)]
struct Outcome {
    accepted: usize,
    sum: Errors,
}

impl Outcome {
    fn accumulate(&mut self, estimate: Option<apollo_stft::PeakEstimate<f64>>, truth: &Tone) {
        let Some(estimate) = estimate else {
            return;
        };
        self.accepted += 1;
        let phase = (estimate.phase() - truth.phase + PI).rem_euclid(TAU) - PI;
        self.sum.frequency += (estimate.position() * BIN_WIDTH - truth.frequency).abs();
        self.sum.amplitude += (estimate.amplitude() - truth.amplitude).abs();
        self.sum.phase += phase.abs();
    }

    fn mean(&self) -> Errors {
        let n = self.accepted.max(1) as f64;
        Errors {
            frequency: self.sum.frequency / n,
            amplitude: self.sum.amplitude / n,
            phase: self.sum.phase / n,
        }
    }
}

#[test]
fn the_three_tone_scene_resolves_to_its_floors() {
    let mut direct = [Outcome::default(); 3];
    let mut peeled = [Outcome::default(); 3];
    let rounds = NonZeroUsize::new(PEEL_ROUNDS).expect("invariant: three is non-zero");
    let mut tones = Vec::new();
    for seed in 1..=SEEDS {
        let (scene_tones, spectrum) = scene(seed);
        let bins: Vec<usize> = scene_tones.iter().map(Tone::bin).collect();
        for (t, tone) in scene_tones.iter().enumerate() {
            let alone = estimate_peaks(&spectrum, &[bins[t]], NonZeroUsize::MIN)
                .expect("invariant: the scene's bins are in range");
            direct[t].accumulate(alone[0], tone);
        }
        let all = estimate_peaks(&spectrum, &bins, rounds)
            .expect("invariant: the scene's bins are in range and distinct");
        for (t, tone) in scene_tones.iter().enumerate() {
            peeled[t].accumulate(all[t], tone);
        }
        tones = scene_tones;
    }
    for (label, outcomes) in [("direct", &direct), ("peeled", &peeled)] {
        for (t, outcome) in outcomes.iter().enumerate() {
            let mean = outcome.mean();
            eprintln!(
                "{label} tone {}: {}/{SEEDS} accepted, {:.2e} Hz, {:.2e} V, {:.2e} rad",
                t + 1,
                outcome.accepted,
                mean.frequency,
                mean.amplitude,
                mean.phase
            );
        }
    }

    // Read alone, the middle tone's bins are the outer tones' leakage: the
    // estimate is rejected or its amplitude error exceeds the whole tone.
    assert!(
        direct[1].accepted == 0 || direct[1].mean().amplitude > TONES[1].1,
        "the middle tone resolved without subtraction: {:?}",
        direct[1]
    );

    // After three rounds every tone resolves in every seed, and the middle
    // tone is noise-limited: within a decade of its Cramér–Rao bound.
    for (t, outcome) in peeled.iter().enumerate() {
        assert_eq!(
            outcome.accepted,
            SEEDS as usize,
            "tone {} rejected after peeling",
            t + 1
        );
    }
    let middle = peeled[1].mean();
    assert!(
        middle.frequency <= 10.0 * tones[1].frequency_bound(),
        "the middle tone is above a decade of its bound {:.2e}: {middle:?}",
        tones[1].frequency_bound()
    );

    // The ADR's Candan 2011 row, reported to two digits: each figure is within
    // half a unit of its last digit, and the transform's rounding may move it
    // further. apollo-fft's bins carry at most `N log₂N ε` relative rounding
    // (one `ε` per butterfly level along each of the `N` paths into a bin), and
    // a relative bin error `η` moves an offset by at most `2η` bins, an
    // amplitude by `2η` of itself and a phase by `2η` (the kernel slopes of the
    // `kernel::peak` tests at these tones' negligible image). The middle tone's
    // amplitude and phase are set by the outer tones' rounding in its bins,
    // not by the estimator, so only its frequency is compared.
    let eta = LEN as f64 * (LEN as f64).log2() * f64::EPSILON;
    let reported = |value: f64, measured: f64, rounding: f64| {
        let unit = 10f64.powf(value.abs().log10().floor() - 1.0);
        (measured - value).abs() <= 0.5 * unit + rounding
    };
    for (t, (frequency, amplitude, phase)) in [
        (0, (8.4e-10, 1.1e-9, 2.6e-9)),
        (2, (7.8e-10, 5.1e-10, 2.4e-9)),
    ] {
        let mean = peeled[t].mean();
        assert!(
            reported(frequency, mean.frequency, 2.0 * eta * BIN_WIDTH)
                && reported(amplitude, mean.amplitude, 2.0 * eta * TONES[t].1)
                && reported(phase, mean.phase, 2.0 * eta),
            "tone {} after peeling: {mean:?} departs from ADR 0066's ({frequency:e}, {amplitude:e}, {phase:e})",
            t + 1
        );
    }
    assert!(
        reported(
            6.0e-7,
            middle.frequency,
            2.0 * eta * BIN_WIDTH * TONES[0].1 / TONES[1].1
        ),
        "the middle tone's frequency {:.3e} departs from ADR 0066's 6.0e-7",
        middle.frequency
    );
}
