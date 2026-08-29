//! Real-to-complex split verified against an analytical oracle and RealFFT.
//!
//! Apollo's real-input forward transform computes a size-`N/2` complex
//! transform and untangles it, rather than widening the input to complex and
//! running a size-`N` transform. These tests pin that the shorter route
//! produces the same spectrum.
//!
//! ## Oracles, in order of authority
//!
//! 1. **Analytical.** A sum of integer-frequency tones has an exactly known
//!    spectrum, so the first test needs no reference implementation at all.
//! 2. **Conjugate symmetry.** A real signal's spectrum satisfies
//!    `X[N-k] = conj(X[k])`; this is a property of the input, not of any
//!    implementation, and a split that mismatched its halves would break it.
//! 3. **Differential against RealFFT**, an independently authored real-FFT
//!    implementation with a different internal algorithm.
//!
//! ## Tolerance
//!
//! Bounds derive from the `O(log N · u)` FFT forward-error bound (Higham,
//! *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section 24.1)
//! with `|X_k| <= ||x||_1`, not from observed error.

use eunomia::Complex64;
use realfft::RealFftPlanner;
use std::f64::consts::TAU;

/// `2c` from the forward-error bound: covers both engines plus their differing
/// twiddle generation.
const TOLERANCE_FACTOR: f64 = 16.0;

fn tolerance(n: usize, l1: f64) -> f64 {
    let stages = f64::from(u32::try_from(n.trailing_zeros()).expect("power of two fits u32"));
    TOLERANCE_FACTOR * stages * (f64::EPSILON / 2.0) * l1
}

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            (0.017 * x).sin() + 0.4 * (0.083 * x).cos()
        })
        .collect()
}

/// Sizes covering the split path and the lengths that fall back to widening.
const SIZES: [usize; 8] = [4, 8, 16, 64, 256, 1024, 4096, 16384];

#[test]
fn matches_exact_spectrum_of_a_known_tone_sum() {
    // A sum of integer-frequency tones has an exactly known DFT, so this test
    // depends on no reference implementation.
    for n in SIZES {
        let tones: [(usize, f64); 3] = [(1, 1.0), (3, -0.5), (7, 0.25)];
        let applicable: Vec<_> = tones.into_iter().filter(|(k, _)| *k < n / 2).collect();
        let src: Vec<f64> = (0..n)
            .map(|i| {
                applicable
                    .iter()
                    .map(|(k, a)| a * (TAU * ((k * i) % n) as f64 / n as f64).cos())
                    .sum()
            })
            .collect();

        let spectrum = apollo_fft::fft_1d_slice::<f64>(&src);
        assert_eq!(spectrum.len(), n, "full spectrum length");

        let l1: f64 = src.iter().map(|v| v.abs()).sum();
        let bound = tolerance(n, l1);
        // A real cosine of amplitude a at bin k contributes a·N/2 to bins k and N-k.
        for (bin, value) in spectrum.iter().enumerate() {
            let expected = applicable
                .iter()
                .find(|(k, _)| *k == bin || n - *k == bin)
                .map_or(0.0, |(_, a)| a * n as f64 / 2.0);
            let err = (value.re - expected).hypot(value.im);
            assert!(
                err <= bound,
                "N={n} bin {bin}: |{value:?} - {expected}| = {err:.3e} exceeds {bound:.3e}"
            );
        }
    }
}

#[test]
fn spectrum_is_conjugate_symmetric() {
    // A property of any real input, independent of how the transform is done.
    for n in SIZES {
        let src = signal(n);
        let spectrum = apollo_fft::fft_1d_slice::<f64>(&src);
        let l1: f64 = src.iter().map(|v| v.abs()).sum();
        let bound = tolerance(n, l1);
        for k in 1..n / 2 {
            let (a, b) = (spectrum[k], spectrum[n - k]);
            let err = (a.re - b.re).hypot(a.im + b.im);
            assert!(
                err <= bound,
                "N={n}: X[{k}] and conj(X[{}]) differ by {err:.3e} > {bound:.3e}",
                n - k
            );
        }
        // The DC and Nyquist bins are purely real for a real signal.
        assert!(spectrum[0].im.abs() <= bound, "N={n}: DC bin not real");
        assert!(
            spectrum[n / 2].im.abs() <= bound,
            "N={n}: Nyquist bin not real"
        );
    }
}

#[test]
fn agrees_with_realfft_on_the_independent_bins() {
    let mut planner = RealFftPlanner::<f64>::new();
    for n in SIZES {
        let src = signal(n);
        let apollo = apollo_fft::fft_1d_slice::<f64>(&src);

        let r2c = planner.plan_fft_forward(n);
        let mut input = src.clone();
        let mut reference = r2c.make_output_vec();
        r2c.process(&mut input, &mut reference).unwrap();
        assert_eq!(reference.len(), n / 2 + 1);

        let l1: f64 = src.iter().map(|v| v.abs()).sum();
        let bound = tolerance(n, l1);
        // Reduction order differs between the two engines, so the oracle is an
        // epsilon bound rather than equality.
        let worst = reference
            .iter()
            .enumerate()
            .map(|(k, r)| (apollo[k].re - r.re).hypot(apollo[k].im - r.im))
            .fold(0.0f64, f64::max);
        assert!(
            worst <= bound,
            "N={n}: Apollo and RealFFT differ by {worst:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn half_spectrum_entry_allocates_nothing_and_matches_the_full_one() {
    use apollo_fft::RealFftData;

    for n in SIZES
        .into_iter()
        .filter(|&n| <f64 as RealFftData>::real_split_applies(n))
    {
        let src = signal(n);
        let full = apollo_fft::fft_1d_slice::<f64>(&src);

        let half_plan =
            <f64 as apollo_fft::PlanCacheProvider>::get_1d_plan(apollo_fft::Shape1D { n: n / 2 });
        let mut half = vec![Complex64::default(); n / 2 + 1];
        <f64 as RealFftData>::forward_1d_half_into(half_plan.as_ref(), &src, &mut half);

        for (k, value) in half.iter().enumerate() {
            assert_eq!(
                (value.re, value.im),
                (full[k].re, full[k].im),
                "N={n} bin {k}: half-spectrum entry disagrees with the full spectrum"
            );
        }
    }
}

#[test]
fn split_spectrum_round_trips_through_the_public_inverse() {
    for n in [128, 256, 512] {
        let src = signal(n);
        let spectrum = apollo_fft::fft_1d_slice::<f64>(&src);
        let reconstructed = apollo_fft::ifft_1d_slice::<f64>(&spectrum);
        let input_l1 = src.iter().map(|value| value.abs()).sum();
        let spectrum_l1 = spectrum.iter().map(|value| value.re.hypot(value.im)).sum();
        // The inverse propagates each forward-bin error through a 1/N-scaled
        // sum and adds its own O(log N * u) transform error.
        let bound = tolerance(n, input_l1) + tolerance(n, spectrum_l1) / n as f64;
        let worst = src
            .iter()
            .zip(&reconstructed)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= bound,
            "N={n}: public real round trip differs by {worst:.3e} > {bound:.3e}"
        );
    }
}
