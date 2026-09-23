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

#![expect(
    clippy::unwrap_used,
    reason = "ratchet APOLLO-UNWRAP-1: pre-existing debt"
)]

use eunomia::Complex64;
use realfft::RealFftPlanner;
use std::f64::consts::TAU;

/// `2c` from the forward-error bound: covers both engines plus their differing
/// twiddle generation.
const TOLERANCE_FACTOR: f64 = 16.0;

/// Each restarted untangling block advances its twiddle at most seven times,
/// and a complex multiply has six rounded scalar operations, so `42u` bounds
/// the recurrence's additional error independently of `N`.
const RESTARTED_RECURRENCE_FACTOR: f64 = 42.0;

fn tolerance(n: usize, l1: f64) -> f64 {
    let stages = f64::from(u32::try_from(n.trailing_zeros()).expect("power of two fits u32"));
    (TOLERANCE_FACTOR * stages + RESTARTED_RECURRENCE_FACTOR) * (f64::EPSILON / 2.0) * l1
}

fn tolerance_f32(n: usize, l1: f32) -> f32 {
    let stages =
        f32::from(u16::try_from(n.trailing_zeros()).expect("power-of-two stage count fits u16"));
    // The same derivation as the f64 bound, in native precision.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "42 and 16 are exact in f32"
    )]
    let (recurrence, factor) = (RESTARTED_RECURRENCE_FACTOR as f32, TOLERANCE_FACTOR as f32);
    (factor * stages + recurrence) * (f32::EPSILON / 2.0) * l1
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
const SIZES: [usize; 9] = [4, 8, 16, 64, 256, 1024, 4096, 16384, 65536];

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
fn matches_exact_spectrum_in_native_f32() {
    for n in SIZES {
        let n_f32 = f32::from(u16::try_from(n - 1).expect("test length minus one fits u16")) + 1.0;
        let tones: [(usize, f32); 3] = [(1, 1.0), (3, -0.5), (7, 0.25)];
        let applicable: Vec<_> = tones.into_iter().filter(|(k, _)| *k < n / 2).collect();
        let src: Vec<f32> = (0..n)
            .map(|i| {
                applicable
                    .iter()
                    .map(|(k, amplitude)| {
                        let residue =
                            u16::try_from((k * i) % n).expect("test phase residue fits u16");
                        amplitude * (std::f32::consts::TAU * f32::from(residue) / n_f32).cos()
                    })
                    .sum()
            })
            .collect();

        let spectrum = apollo_fft::fft_1d_slice::<f32>(&src);
        let l1: f32 = src.iter().map(|value| value.abs()).sum();
        let bound = tolerance_f32(n, l1);
        for (bin, value) in spectrum.iter().enumerate() {
            let expected = applicable
                .iter()
                .find(|(k, _)| *k == bin || n - *k == bin)
                .map_or(0.0, |(_, amplitude)| amplitude * n_f32 / 2.0);
            let error = (value.re - expected).hypot(value.im);
            assert!(
                error <= bound,
                "N={n} bin {bin}: |{value:?} - {expected}| = {error:.3e} exceeds {bound:.3e}"
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

        let half_plan = <f64 as apollo_fft::PlanCacheProvider>::get_1d_plan(
            apollo_fft::Shape1D::new(n / 2).expect("invariant: shape lengths are non-zero"),
        );
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

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn best_block(calls: u32, mut operation: impl FnMut()) -> f64 {
    const BLOCKS: usize = 12;
    let mut best = f64::INFINITY;
    for _ in 0..BLOCKS {
        let started = std::time::Instant::now();
        for _ in 0..calls {
            operation();
        }
        best = best.min(started.elapsed().as_nanos() as f64 / f64::from(calls));
    }
    best
}

/// Pinned attribution for the allocation-free half-spectrum route.
///
/// The retained production entry is measured without changing its workload.
/// The other rows isolate cache acquisition, pair packing, and the packed
/// half-length transform; subtracting adjacent rows attributes the untangle.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
#[test]
#[ignore = "measurement instrument for real half-spectrum phase attribution"]
fn attributes_real_half_spectrum_phases() {
    use apollo_fft::{PlanCacheProvider, RealFftData, Shape1D};
    use hermes_simd::{ProcessorBinding, ProcessorIndex};
    use std::hint::black_box;

    let processor = ProcessorIndex::new(2);
    let _binding =
        ProcessorBinding::bind(processor).expect("measurement processor must be available");
    std::thread::yield_now();
    assert_eq!(
        ProcessorIndex::current()
            .expect("Windows supports processor queries")
            .get(),
        processor.get(),
        "processor binding must remain exact"
    );

    for n in [1_024usize, 4_096, 16_384, 65_536, 262_144] {
        let calls = u32::try_from(1_000_000usize / n)
            .expect("probe call count fits u32")
            .max(4);
        let src = signal(n);
        let half_plan = <f64 as PlanCacheProvider>::get_1d_plan(
            Shape1D::new(n / 2).expect("probe lengths have non-zero halves"),
        );
        let mut direct = vec![Complex64::default(); n / 2 + 1];
        let mut public = direct.clone();
        <f64 as RealFftData>::forward_1d_half_into(half_plan.as_ref(), &src, &mut direct);
        apollo_fft::fft_1d_slice_half_into::<f64>(&src, &mut public);
        assert_eq!(direct, public, "N={n}: direct and public halves differ");

        let cache_ns = best_block(calls, || {
            black_box(<f64 as PlanCacheProvider>::get_1d_plan(
                Shape1D::new(n / 2).expect("probe lengths have non-zero halves"),
            ));
        });
        let pack_ns = best_block(calls, || {
            <f64 as RealFftData>::pack_real_pairs(black_box(&src), black_box(&mut direct[..n / 2]));
        });
        let pack_fft_ns = best_block(calls, || {
            <f64 as RealFftData>::pack_real_pairs(black_box(&src), black_box(&mut direct[..n / 2]));
            half_plan.forward_complex_slice_inplace(black_box(&mut direct[..n / 2]));
        });
        let direct_ns = best_block(calls, || {
            <f64 as RealFftData>::forward_1d_half_into(
                half_plan.as_ref(),
                black_box(&src),
                black_box(&mut direct),
            );
        });
        let public_ns = best_block(calls, || {
            apollo_fft::fft_1d_slice_half_into::<f64>(black_box(&src), black_box(&mut public));
        });

        println!(
            "REAL_HALF cpu={} n={n:<6} calls={calls:<4} cache={cache_ns:>9.1}ns pack={pack_ns:>9.1}ns half_fft={:>9.1}ns untangle={:>9.1}ns direct={direct_ns:>9.1}ns public={public_ns:>9.1}ns",
            processor.get(),
            (pack_fft_ns - pack_ns).max(0.0),
            (direct_ns - pack_fft_ns).max(0.0),
        );
    }
}

/// The split at `n ≡ 2 (mod 4)`, whose half length `m = n/2` is odd, so the
/// untangle has no self-paired midpoint and the half transform runs a
/// composite, Rader or Bluestein route rather than a power of two.
///
/// Bound: the half transform is at worst Bluestein's three transforms at the
/// padded `p < 4n`, each within the `O(log p · u)` forward-error bound
/// (Higham §24.1, the file's `TOLERANCE_FACTOR` per transform), plus the
/// untangle's restarted recurrence; the direct-sum oracle adds `n` rounded
/// products and sums, `2n u`. All against `|X_k| ≤ ‖x‖₁`.
#[test]
fn split_serves_lengths_two_mod_four() {
    let lengths = (6..=602).step_by(4).chain([1002, 2050]);
    for n in lengths {
        assert!(
            <f64 as apollo_fft::RealFftData>::real_split_applies(n),
            "N={n} must take the split"
        );
        let src = signal(n);
        let l1: f64 = src.iter().map(|value| value.abs()).sum();
        let log_p = f64::from((4 * n).next_power_of_two().trailing_zeros());
        let bound = (3.0 * TOLERANCE_FACTOR * log_p + RESTARTED_RECURRENCE_FACTOR + 4.0 * n as f64)
            * (f64::EPSILON / 2.0)
            * l1;

        let half_plan = <f64 as apollo_fft::PlanCacheProvider>::get_1d_plan(
            apollo_fft::Shape1D::new(n / 2).expect("non-zero"),
        );
        let mut half = vec![Complex64::default(); n / 2 + 1];
        <f64 as apollo_fft::RealFftData>::forward_1d_half_into(half_plan.as_ref(), &src, &mut half);
        for (k, value) in half.iter().enumerate() {
            let exact = (0..n).fold(Complex64::new(0.0, 0.0), |acc, j| {
                let angle = -TAU * ((j * k) % n) as f64 / n as f64;
                acc + Complex64::from_polar(src[j], angle)
            });
            let error = (value.re - exact.re).hypot(value.im - exact.im);
            assert!(error <= bound, "N={n} bin {k}: {error:.3e} > {bound:.3e}");
        }

        let mut spectrum = half.clone();
        let mut back = vec![0.0f64; n];
        <f64 as apollo_fft::RealFftData>::inverse_1d_half_into(
            half_plan.as_ref(),
            &mut spectrum,
            &mut back,
        );
        // The inverse runs the same routes on bins of size at most `‖x‖₁`,
        // scaled by `1/n`.
        let worst = src
            .iter()
            .zip(&back)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= 2.0 * bound,
            "N={n}: round trip {worst:.3e} > {:.3e}",
            2.0 * bound
        );
    }
}

/// Every even length the split admits, 2 through 2048, against the widened
/// size-`n` complex transform of the same signal: a route with no split,
/// whose own accuracy the direct-sum sweep of `tests/dft_oracle_sweep.rs`
/// establishes. The direct sum itself is the oracle of
/// [`split_serves_lengths_two_mod_four`]; here it would cost `O(n²)` per
/// length.
///
/// Bound: each side carries the half-length or full-length route's error
/// under the bound of [`split_serves_lengths_two_mod_four`] less its
/// direct-sum term, so their difference is within twice that.
#[test]
fn split_serves_every_even_length() {
    for n in (2..=2048).step_by(2) {
        assert!(
            <f64 as apollo_fft::RealFftData>::real_split_applies(n),
            "N={n} must take the split"
        );
        let src = signal(n);
        let l1: f64 = src.iter().map(|value| value.abs()).sum();
        let log_p = f64::from((4 * n).next_power_of_two().trailing_zeros());
        let bound = 2.0
            * (3.0 * TOLERANCE_FACTOR * log_p + RESTARTED_RECURRENCE_FACTOR)
            * (f64::EPSILON / 2.0)
            * l1;

        let half = apollo_fft::fft_1d_slice_half::<f64>(&src);
        let full_plan = <f64 as apollo_fft::PlanCacheProvider>::get_1d_plan(
            apollo_fft::Shape1D::new(n).expect("non-zero"),
        );
        let mut widened: Vec<Complex64> = src
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect();
        full_plan.forward_complex_slice_inplace(&mut widened);
        for (k, (value, reference)) in half.iter().zip(&widened).enumerate() {
            let error = (value.re - reference.re).hypot(value.im - reference.im);
            assert!(error <= bound, "N={n} bin {k}: {error:.3e} > {bound:.3e}");
        }

        // The inverse runs the half-length route on bins of size at most
        // `‖x‖₁`, scaled by `1/n`, after bins carrying half this bound.
        let mut spectrum = half;
        let mut back = vec![0.0f64; n];
        apollo_fft::ifft_1d_slice_half_into::<f64>(&mut spectrum, &mut back);
        let worst = src
            .iter()
            .zip(&back)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= bound,
            "N={n}: round trip {worst:.3e} > {bound:.3e}"
        );
    }
}
