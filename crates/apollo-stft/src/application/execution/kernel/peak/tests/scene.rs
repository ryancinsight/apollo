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
//! exactly at each bin read, and the transform's rounding: `fft_1d_slice`
//! reaches each bin through at most `⌈log₂ N⌉ + 3` scalings by rounded
//! unit-modulus factors per sample and a summation of the `N` samples, so a
//! bin is within `(N + log₂ N + 4) ε Σ|s_t|` of the exact sum, plus the
//! samples' own `(Θ + 2) ε A` for arguments within `Θ`.

use core::num::NonZeroUsize;
use std::f64::consts::{PI, TAU};

use eunomia::Complex64;

use super::{Model, Tone};
use crate::estimate_peaks;

const SAMPLE_RATE: f64 = 48_000.0;
const LEN: usize = 48_000;
const BIN_WIDTH: f64 = SAMPLE_RATE / LEN as f64;
const NOISE_STD: f64 = 1e-10;
const SEEDS: u64 = 8;
/// Estimate-and-subtract rounds: ADR 0066 measured the middle tone inside the
/// noise floor by the third.
const PEEL_ROUNDS: usize = 3;
const TONES: [(f64, f64); 3] = [(8901.37, 1.0), (8950.61, 1e-6), (9000.23, 1.0)];

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
        let signal: Vec<f64> = super::real_samples(&tones, LEN)
            .into_iter()
            .zip(&noise)
            .map(|(tone, noise)| tone + noise)
            .collect();
        let spectrum = apollo_fft::fft_1d_slice::<f64>(&signal);
        Self {
            tones,
            noise,
            spectrum,
        }
    }

    /// The noise's DFT at bin `p`, summed in f64, with its own summation
    /// rounding.
    fn noise_at(&self, p: f64) -> f64 {
        let n = LEN as f64;
        let step = -TAU * p / n;
        let (sum, magnitude) = self.noise.iter().enumerate().fold(
            (Complex64::new(0.0, 0.0), 0.0),
            |(sum, magnitude), (t, &s)| {
                (
                    sum + Complex64::from_polar(s, step * t as f64),
                    magnitude + s.abs(),
                )
            },
        );
        sum.norm() + (n + TAU * p + 7.0) * f64::EPSILON * magnitude
    }

    /// Cramér–Rao bound on the frequency standard deviation of a real tone in
    /// real white noise, in Hz: four times Aboutanios–Mulgrew (3) for the
    /// complex exponential, `σ_f = (f_s / 2π) √(24 / (ρ N (N² − 1)))`,
    /// `ρ = A² / σ²`.
    fn frequency_bound(tone: &Tone) -> f64 {
        let n = LEN as f64;
        let rho = tone.amplitude * tone.amplitude / (NOISE_STD * NOISE_STD);
        SAMPLE_RATE / TAU * (24.0 / (rho * n * (n * n - 1.0))).sqrt()
    }
}

#[test]
fn the_three_tone_scene_resolves_to_its_bounds() {
    let model = Model::new(LEN);
    let rounds = NonZeroUsize::new(PEEL_ROUNDS).expect("invariant: three is non-zero");
    let mut middle_error = 0.0;
    let mut middle_bound = 0.0;
    for seed in 1..=SEEDS {
        let scene = Scene::new(seed);
        let bins: Vec<usize> = scene
            .tones
            .iter()
            .map(|tone| tone.position.round() as usize)
            .collect();
        let total: f64 = scene.tones.iter().map(|tone| tone.amplitude).sum::<f64>()
            + scene.noise.iter().map(|s| s.abs()).sum::<f64>() / LEN as f64;
        let argument = super::argument(&scene.tones);
        let n = LEN as f64;
        let transform = n * f64::EPSILON * total * (n + n.log2() + 4.0 + argument + 2.0);
        let bin_error = |p: f64| scene.noise_at(p) + transform;
        model.check_peeled::<f64>(
            &scene.spectrum,
            &scene.tones,
            &bins,
            PEEL_ROUNDS,
            &bin_error,
            &format!("seed {seed}"),
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

        let peeled = estimate_peaks(&scene.spectrum, &bins, rounds)
            .expect("invariant: the scene's bins are valid");
        let middle = peeled[1].expect("the middle tone resolves after peeling");
        middle_error += (middle.position() - scene.tones[1].position).abs() * BIN_WIDTH;
        middle_bound = Scene::frequency_bound(&scene.tones[1]);
    }
    // Noise-limited: the mean absolute frequency error over the seeds, about
    // 0.8 σ for a Gaussian, within a decade of the bound.
    let middle_error = middle_error / SEEDS as f64;
    eprintln!("middle tone: {middle_error:.2e} Hz against its bound {middle_bound:.2e} Hz");
    assert!(
        middle_error <= 10.0 * middle_bound,
        "the middle tone's {middle_error:.2e} Hz is above a decade of its bound {middle_bound:.2e}"
    );
}
