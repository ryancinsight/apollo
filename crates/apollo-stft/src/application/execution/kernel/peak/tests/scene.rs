//! ADR 0066's three-tone scene against `estimate_peaks`.
//!
//! The scene is Henry, Science Talks 4 (2022), Figs. 6-24: 48 kHz, 48 000
//! samples (a 1 Hz bin), three tones near 8950 Hz about 50 Hz apart, the outer
//! two at 1 V and the middle at 1e-6 V, white noise of 1e-10 V, random phases
//! per seed, eight seeds.
//!
//! It runs in f64 only: its noise and its middle tone sit below what f32
//! resolves in a 1 V spectrum of this length, where a bin of the outer tones
//! carries about `N ε₃₂ A` = 6e-3 of rounding against the middle tone's
//! `A N / π` of 1.5e-2.
//!
//! Every read of the three rounds is checked by [`Model::check_peeled`], the
//! outer tones' first read carrying the other two tones whole (the direct
//! pass). The bins beyond the tones hold the seed's noise, computed here
//! exactly at each bin read. Sample construction is bounded with `γ_m` factors
//! from its operation counts; direct DFTs use integer-reduced phases and the
//! same composed phasor/summation bound. `fft_1d_slice` adds its existing
//! `(N + log₂ N + 4) ε Σ|s_t|` transform bound. The assumed absolute error of
//! each elementary sine or cosine is one epsilon; this is not a portable libm
//! proof.

use core::num::NonZeroUsize;
use std::f64::consts::{PI, TAU};

use eunomia::Complex64;

use super::{Estimate, Model, Tone};
use crate::estimate_peaks;

mod reference;

use reference::{
    assert_within_truth_bounds, check_geometric_rounds, production_rounds, reference_rounds,
};

const SAMPLE_RATE: f64 = 48_000.0;
const LEN: usize = 48_000;
const BIN_WIDTH: f64 = SAMPLE_RATE / LEN as f64;
const NOISE_STD: f64 = 1e-10;
const SEEDS: u64 = 8;
/// Estimate-and-subtract rounds: ADR 0066 measured the middle tone inside the
/// noise floor by the third.
const PEEL_ROUNDS: usize = 3;
const TONES: [(f64, f64); 3] = [(8901.37, 1.0), (8950.61, 1e-6), (9000.23, 1.0)];
const ORDERINGS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

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

struct Scene {
    tones: Vec<Tone>,
    noise: Vec<f64>,
    samples: Vec<f64>,
    spectrum: Vec<Complex64>,
}

impl Scene {
    fn new(seed: u64) -> Self {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let tones: Vec<Tone> = TONES
            .iter()
            .map(|&(frequency, amplitude)| Tone {
                position: frequency / BIN_WIDTH,
                amplitude,
                phase: TAU * rng.uniform() - PI,
            })
            .collect();
        let noise: Vec<f64> = (0..LEN).map(|_| NOISE_STD * rng.gaussian()).collect();
        let samples: Vec<f64> = super::real_samples(&tones, LEN)
            .into_iter()
            .zip(&noise)
            .map(|(tone, noise)| tone + noise)
            .collect();
        let spectrum = apollo_fft::fft_1d_slice::<f64>(&samples);
        Self {
            tones,
            noise,
            samples,
            spectrum,
        }
    }

    /// Replaces exactly the nine bins consumed by the estimator with their
    /// defining sample sums. The untouched entries cannot affect a read.
    fn direct_spectrum(&self, bins: &[usize]) -> Vec<Complex64> {
        let mut spectrum = self.spectrum.clone();
        for &bin in bins {
            for read in [bin - 1, bin, bin + 1] {
                spectrum[read] = self.samples.iter().enumerate().fold(
                    Complex64::new(0.0, 0.0),
                    |sum, (time, &sample)| {
                        let angle = -TAU * ((read * time) % LEN) as f64 / LEN as f64;
                        sum + Complex64::from_polar(sample, angle)
                    },
                );
            }
        }
        spectrum
    }

    fn sample_rounding(&self) -> f64 {
        let epsilon = f64::EPSILON;
        let unit_roundoff = epsilon / 2.0;
        let tone_error = self
            .tones
            .iter()
            .map(|tone| {
                let theta = TAU * tone.position.abs() + tone.phase.abs();
                tone.amplitude.abs()
                    * ((1.0 + unit_roundoff) * (super::gamma(epsilon, 5) * theta + epsilon)
                        + unit_roundoff)
            })
            .sum::<f64>();
        let amplitude = self
            .tones
            .iter()
            .map(|tone| tone.amplitude.abs())
            .sum::<f64>();
        let noise = self.noise.iter().map(|value| value.abs()).sum::<f64>();
        let summation = super::gamma(epsilon, 3);
        (1.0 + summation) * LEN as f64 * tone_error + summation * (LEN as f64 * amplitude + noise)
    }

    fn phasor_rounding() -> f64 {
        let epsilon = f64::EPSILON;
        let unit_roundoff = epsilon / 2.0;
        (1.0 + unit_roundoff) * (super::gamma(epsilon, 3) * TAU + 2.0_f64.sqrt() * epsilon)
            + unit_roundoff
    }

    fn direct_rounding(&self) -> f64 {
        let phasor = Self::phasor_rounding();
        let summation = super::gamma(f64::EPSILON, LEN - 1);
        self.sample_rounding()
            + (phasor + summation * (1.0 + phasor))
                * self.samples.iter().map(|value| value.abs()).sum::<f64>()
    }

    /// The noise's DFT at bin `p`, summed in f64, with its own summation
    /// rounding.
    fn noise_at(&self, p: f64) -> f64 {
        let bin = p as usize;
        let (sum, magnitude) = self.noise.iter().enumerate().fold(
            (Complex64::new(0.0, 0.0), 0.0),
            |(sum, magnitude), (t, &s)| {
                let angle = -TAU * ((bin * t) % LEN) as f64 / LEN as f64;
                (sum + Complex64::from_polar(s, angle), magnitude + s.abs())
            },
        );
        let phasor = Self::phasor_rounding();
        let summation = super::gamma(f64::EPSILON, LEN - 1);
        sum.norm() + (phasor + summation * (1.0 + phasor)) * magnitude
    }

    /// Cramér–Rao bound on the frequency standard deviation of a real tone in
    /// real white noise, in Hz: four times Aboutanios–Mulgrew (3) for the
    /// complex exponential, `σ_f = (f_s / 2π) √(24 / (ρ N (N² − 1)))`,
    /// `ρ = A² / σ²`.
    fn frequency_crlb(tone: &Tone) -> f64 {
        let n = LEN as f64;
        let rho = tone.amplitude * tone.amplitude / (NOISE_STD * NOISE_STD);
        SAMPLE_RATE / TAU * (24.0 / (rho * n * (n * n - 1.0))).sqrt()
    }
}

fn phase_delta(lhs: f64, rhs: f64) -> f64 {
    (lhs - rhs + PI).rem_euclid(TAU) - PI
}

#[derive(Clone, Copy, Default)]
struct Errors {
    frequency: f64,
    amplitude: f64,
    phase: f64,
}

impl Errors {
    fn add(&mut self, estimate: Estimate, tone: &Tone) {
        self.frequency += (estimate.position - tone.position).abs() * BIN_WIDTH;
        self.amplitude += (2.0 * estimate.a.norm() - tone.amplitude).abs();
        self.phase += phase_delta(estimate.a.arg(), tone.phase).abs();
    }
}

#[test]
fn the_three_tone_scene_resolves_to_its_bounds() {
    let model = Model::new(LEN);
    let rounds = NonZeroUsize::new(PEEL_ROUNDS).expect("invariant: three is non-zero");
    let mut fft_errors = [Errors::default(); TONES.len()];
    let mut direct_errors = [Errors::default(); TONES.len()];
    for seed in 1..=SEEDS {
        let scene = Scene::new(seed);
        let bins: Vec<usize> = scene
            .tones
            .iter()
            .map(|tone| tone.position.round() as usize)
            .collect();
        let n = LEN as f64;
        let sample_magnitude = scene.samples.iter().map(|value| value.abs()).sum::<f64>();
        let transform =
            scene.sample_rounding() + (n + n.log2() + 4.0) * f64::EPSILON * sample_magnitude;
        // The estimator reads only the bins beside the three peaks, so their
        // noise is summed once rather than at every bound evaluation.
        let noise: Vec<(usize, f64)> = bins
            .iter()
            .flat_map(|&bin| [bin - 1, bin, bin + 1])
            .map(|p| (p, scene.noise_at(p as f64)))
            .collect();
        let noise_at = |p: f64| {
            let bin = p as usize;
            noise.iter().find(|&&(read, _)| read == bin).map_or_else(
                || panic!("bin {bin} is not beside a scene peak"),
                |&(_, value)| value,
            )
        };
        let bin_error = |p: f64| noise_at(p) + transform;
        model.check_peeled::<f64>(
            &scene.spectrum,
            &scene.tones,
            &bins,
            PEEL_ROUNDS,
            &bin_error,
            &format!("seed {seed}"),
        );
        let direct_spectrum = scene.direct_spectrum(&bins);
        let direct_rounding = scene.direct_rounding();
        let direct_bin_error = |p: f64| noise_at(p) + direct_rounding;
        model.check_peeled::<f64>(
            &direct_spectrum,
            &scene.tones,
            &bins,
            PEEL_ROUNDS,
            &direct_bin_error,
            &format!("seed {seed}, direct DFT"),
        );

        // Read alone, the middle tone's bins are the outer tones' leakage: the
        // estimate is rejected or misses the tone by more than its amplitude.
        let alone = estimate_peaks(&scene.spectrum, &[bins[1]], NonZeroUsize::MIN)
            .expect("invariant: the scene's bins are valid");
        if let Some(estimate) = alone[0] {
            assert!(
                (estimate.amplitude() - scene.tones[1].amplitude).abs() > scene.tones[1].amplitude,
                "seed {seed}: the middle tone resolved without subtraction: {estimate:?}"
            );
        }

        let canonical_fft = estimate_peaks(&scene.spectrum, &bins, rounds)
            .expect("invariant: the scene's bins are valid")
            .into_iter()
            .map(|estimate| estimate.map(Estimate::from))
            .collect::<Vec<_>>();
        let canonical_direct = estimate_peaks(&direct_spectrum, &bins, rounds)
            .expect("invariant: the scene's bins are valid")
            .into_iter()
            .map(|estimate| estimate.map(Estimate::from))
            .collect::<Vec<_>>();
        for i in 0..TONES.len() {
            fft_errors[i].add(
                canonical_fft[i].expect("the FFT tone resolves after peeling"),
                &scene.tones[i],
            );
            direct_errors[i].add(
                canonical_direct[i].expect("the direct-DFT tone resolves after peeling"),
                &scene.tones[i],
            );
        }

        for ordering in ORDERINGS {
            let ordered_bins = ordering.map(|i| bins[i]);
            let ordered_tones = ordering.map(|i| scene.tones[i]);
            let fft_rounds = production_rounds(&scene.spectrum, &ordered_bins);
            let direct_rounds = production_rounds(&direct_spectrum, &ordered_bins);
            let reference_fft_rounds =
                reference_rounds(&scene.spectrum, &ordered_bins, PEEL_ROUNDS);
            let reference_direct_rounds =
                reference_rounds(&direct_spectrum, &ordered_bins, PEEL_ROUNDS);
            let label = format!("seed {seed}, ordering {ordering:?}");
            let fft_bounds = check_geometric_rounds(
                &scene.spectrum,
                &ordered_tones,
                &ordered_bins,
                &fft_rounds,
                &bin_error,
                &format!("{label}, FFT production"),
            );
            let direct_bounds = check_geometric_rounds(
                &direct_spectrum,
                &ordered_tones,
                &ordered_bins,
                &direct_rounds,
                &direct_bin_error,
                &format!("{label}, direct-DFT production"),
            );
            let reference_fft_bounds = check_geometric_rounds(
                &scene.spectrum,
                &ordered_tones,
                &ordered_bins,
                &reference_fft_rounds,
                &bin_error,
                &format!("{label}, FFT reference"),
            );
            let reference_direct_bounds = check_geometric_rounds(
                &direct_spectrum,
                &ordered_tones,
                &ordered_bins,
                &reference_direct_rounds,
                &direct_bin_error,
                &format!("{label}, direct-DFT reference"),
            );
            for place in 0..TONES.len() {
                let tone_index = ordering[place];
                let tone = &scene.tones[tone_index];
                let fft = fft_rounds[PEEL_ROUNDS][place]
                    .expect("the FFT production estimate resolves after peeling");
                let direct = direct_rounds[PEEL_ROUNDS][place]
                    .expect("the direct-DFT production estimate resolves after peeling");
                let reference_fft = reference_fft_rounds[PEEL_ROUNDS][place]
                    .expect("the FFT reference estimate resolves after peeling");
                let reference_direct = reference_direct_rounds[PEEL_ROUNDS][place]
                    .expect("the direct-DFT reference estimate resolves after peeling");
                let tone_label = format!("{label}, tone {tone_index}");
                assert_within_truth_bounds(
                    fft,
                    fft_bounds[place],
                    reference_fft,
                    reference_fft_bounds[place],
                    tone,
                    &format!("{tone_label}, FFT production/reference"),
                );
                assert_within_truth_bounds(
                    direct,
                    direct_bounds[place],
                    reference_direct,
                    reference_direct_bounds[place],
                    tone,
                    &format!("{tone_label}, direct production/reference"),
                );
                assert_within_truth_bounds(
                    fft,
                    fft_bounds[place],
                    direct,
                    direct_bounds[place],
                    tone,
                    &format!("{tone_label}, FFT/direct production"),
                );
                assert_within_truth_bounds(
                    reference_fft,
                    reference_fft_bounds[place],
                    reference_direct,
                    reference_direct_bounds[place],
                    tone,
                    &format!("{tone_label}, FFT/direct reference"),
                );

                let canonical_fft =
                    canonical_fft[tone_index].expect("the canonical FFT estimate resolved above");
                let canonical_direct = canonical_direct[tone_index]
                    .expect("the canonical direct estimate resolved above");
                assert_eq!(fft.position, canonical_fft.position, "{tone_label}");
                assert_eq!(fft.a, canonical_fft.a, "{tone_label}");
                assert_eq!(direct.position, canonical_direct.position, "{tone_label}");
                assert_eq!(direct.a, canonical_direct.a, "{tone_label}");
            }
        }
    }
    for (i, tone) in TONES.iter().enumerate() {
        let count = SEEDS as f64;
        let fft = fft_errors[i];
        let direct = direct_errors[i];
        eprintln!(
            "tone {i} ({:.2e} V): FFT {:.2e} Hz, {:.2e} V, {:.2e} rad; direct DFT {:.2e} Hz, {:.2e} V, {:.2e} rad; frequency CRLB {:.2e} Hz",
            tone.1,
            fft.frequency / count,
            fft.amplitude / count,
            fft.phase / count,
            direct.frequency / count,
            direct.amplitude / count,
            direct.phase / count,
            Scene::frequency_crlb(&Tone {
                position: tone.0 / BIN_WIDTH,
                amplitude: tone.1,
                phase: 0.0,
            })
        );
    }
}
