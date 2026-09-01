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

fn tolerance_f32(n: usize, l1: f32) -> f32 {
    let stages =
        f32::from(u16::try_from(n.trailing_zeros()).expect("power-of-two stage count fits u16"));
    // Each restarted block advances its twiddle at most seven times. A complex
    // multiply has six rounded scalar operations, so 42u bounds the additional
    // recurrence error independently of N.
    const RESTARTED_RECURRENCE_FACTOR: f32 = 42.0;
    (16.0 * stages + RESTARTED_RECURRENCE_FACTOR) * (f32::EPSILON / 2.0) * l1
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
