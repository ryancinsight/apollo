//! Correctness for the N = 16 register-resident codelet.
//!
//! The primary oracle is a compensated direct DFT — analytical, independent of
//! every other route in the crate. The differential against the incumbent
//! sized path is secondary: it pins that a dispatch swap cannot change results
//! beyond rounding, with the bound derived from the stage counts rather than
//! observed.

use super::try_transform_16;
use crate::application::execution::kernel::components::test_support::executed_or_declined_untouched;
use eunomia::Complex64;
use std::f64::consts::TAU;

fn signal() -> Vec<Complex64> {
    (0..16)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.4 * x + 0.3).sin(), (0.7 * x - 0.1).cos())
        })
        .collect()
}

/// Direct DFT accumulated in f64; at N = 16 its `O(N * u)` error is far below
/// the assertion bound.
fn dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (t, v) in input.iter().enumerate() {
                let (s, c) = (sign * TAU * ((k * t) % n) as f64 / n as f64).sin_cos();
                re += v.re * c - v.im * s;
                im += v.re * s + v.im * c;
            }
            Complex64::new(re, im)
        })
        .collect()
}

/// `O(stages * u * l1)` with the shared factor the crate's other accuracy
/// tests use.
fn bound(input: &[Complex64]) -> f64 {
    let l1: f64 = input.iter().map(|v| v.re.hypot(v.im)).sum();
    16.0 * 4.0 * (f64::EPSILON / 2.0) * l1
}

fn worst(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.re - y.re).hypot(x.im - y.im))
        .fold(0.0f64, f64::max)
}

#[test]
fn forward_matches_the_direct_transform_when_width_is_supported() {
    let src = signal();
    let mut data = src.clone();
    let executed = try_transform_16::<f64, false, false>(&mut data);
    if !executed_or_declined_untouched(&src, &data, executed) {
        return;
    }
    let expected = dft(&src, false);
    let err = worst(&data, &expected);
    let limit = bound(&src);
    assert!(err <= limit, "forward differs by {err:.3e} > {limit:.3e}");
}

#[test]
fn inverse_matches_the_direct_transform_when_width_is_supported() {
    let src = signal();
    let mut data = src.clone();
    let executed = try_transform_16::<f64, true, false>(&mut data);
    if !executed_or_declined_untouched(&src, &data, executed) {
        return;
    }
    let expected = dft(&src, true);
    let err = worst(&data, &expected);
    let limit = bound(&src);
    assert!(err <= limit, "inverse differs by {err:.3e} > {limit:.3e}");
}

#[test]
fn normalized_round_trip_recovers_the_input_when_width_is_supported() {
    let src = signal();
    let mut data = src.clone();
    let executed = try_transform_16::<f64, false, false>(&mut data);
    if !executed_or_declined_untouched(&src, &data, executed) {
        return;
    }
    assert!(try_transform_16::<f64, true, true>(&mut data));
    let err = worst(&data, &src);
    let limit = 2.0 * bound(&src);
    assert!(
        err <= limit,
        "round trip differs by {err:.3e} > {limit:.3e}"
    );
}

#[test]
fn matches_the_incumbent_route_when_width_is_supported() {
    let src = signal();
    let mut ours = src.clone();
    let executed = try_transform_16::<f64, false, false>(&mut ours);
    if !executed_or_declined_untouched(&src, &ours, executed) {
        return;
    }

    let mut theirs = src.clone();
    crate::FftPlan1D::<f64>::new(
        crate::Shape1D::new(16).expect("invariant: shape lengths are non-zero"),
    )
    .forward_complex_slice_inplace(&mut theirs);

    let err = worst(&ours, &theirs);
    // Two different evaluation orders, each within the stage bound.
    let limit = 2.0 * bound(&src);
    assert!(
        err <= limit,
        "codelet differs from the route by {err:.3e} > {limit:.3e}"
    );
}

/// The width the codelet asks for must be answered by hardware, not emulation.
///
/// `vectorize_lanes` falls through to Hermes' scalar backend when no ISA
/// backend provides exactly the requested width, and the codelet's own
/// `LANE_COUNT != 4` guard passes there too, because the scalar backend
/// provides any width asked of it. The f64 request matches AVX2's four lanes
/// on this host, so the probe measures a vectorized kernel; the same request
/// at f32 does not, and lands on the emulation whose register width is zero.
/// Pinning both keeps a future width change from turning the instrument into a
/// scalar one without saying so.
#[test]
fn the_codelet_width_request_is_answered_by_hardware_for_f64_only() {
    use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel, SimdStorage};
    struct RegisterWidth;
    impl<T: hermes_simd::LaneScalar> LaneKernel<T> for RegisterWidth {
        type Output = (usize, u32);
        fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> (usize, u32) {
            (<A as SimdStorage<T>>::LANE_COUNT, A::REGISTER_WIDTH_BITS)
        }
    }

    let Some((lanes, width)) = hermes_simd::vectorize_lanes::<4, f64, _>(RegisterWidth) else {
        return;
    };
    assert_eq!(
        lanes, 4,
        "the four-lane request must be answered at width four"
    );
    assert_ne!(
        width, 0,
        "the codelet's f64 width request resolved to the scalar emulation, so          every figure the probe reports is for an unvectorized kernel"
    );

    // The f32 side of the same request: `vectorize_lanes` answers through
    // emulation, and only the hardware-only entry reports the truth.
    if let Some((_, f32_width)) = hermes_simd::vectorize_lanes::<4, f32, _>(RegisterWidth) {
        if f32_width == 0 {
            assert!(
                hermes_simd::vectorize_hardware_lanes::<4, f32, _>(RegisterWidth).is_none(),
                "the hardware-only entry must decline what only emulation answered"
            );
        }
    }
}

#[test]
#[ignore = "reconciliation instrument for APOLLO-PROBE-SCALE-DISAGREEMENT"]
fn scratch_reconcile_sweep_context() {
    use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
    use eunomia::Complex64;
    fn case(suite: &mut BenchmarkSuite, n: usize) {
        let src: Vec<Complex64> = (0..n)
            .map(|i| {
                let x = i as f64;
                Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect();
        let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D::new(n).expect("non-zero"));
        let mut work = src.clone();
        suite.run(BenchmarkCase::new("solo", "apollo-f64", n), || {
            work.copy_from_slice(&src);
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
        });
    }
    // One case alone.
    let mut solo = BenchmarkSuite::new(BenchmarkConfig::regression());
    case(&mut solo, 16);
    print!("SOLO {}", solo.report());
    // The same case reached through a sweep, warm-up pass included, as the
    // small-sizes probe does it.
    let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
    for n in [8usize, 16, 32, 64] {
        case(&mut warmup, n);
    }
    drop(warmup);
    let mut swept = BenchmarkSuite::new(BenchmarkConfig::regression());
    for n in [8usize, 16, 32, 64] {
        case(&mut swept, n);
    }
    print!("SWEPT {}", swept.report());
    panic!("reconciliation instrument: read the two reports above");
}

/// The register permutation must reproduce `BIT_REVERSED_16` exactly.
///
/// The direct-DFT oracles above would also fail on a wrong permutation, but
/// only as a diffuse numeric mismatch. This asserts the routing itself, on
/// index labels rather than values, so a swapped source pair names itself.
///
/// The model is the codelet's own shape: register `k` holds samples `2k` and
/// `2k + 1`, and each `deinterleave_pairs(natural[low], natural[low + 4])`
/// sends the two first samples to output `position` and the two second samples
/// to output `position + 4`.
#[test]
fn the_register_permutation_reproduces_the_bit_reversal() {
    let natural: [(usize, usize); 8] = core::array::from_fn(|k| (2 * k, 2 * k + 1));
    let mut reversed = [(0usize, 0usize); 8];
    for (position, low) in super::REGISTER_PAIR_ORDER.into_iter().enumerate() {
        reversed[position] = (natural[low].0, natural[low + 4].0);
        reversed[position + 4] = (natural[low].1, natural[low + 4].1);
    }

    let flat: Vec<usize> = reversed
        .into_iter()
        .flat_map(|(first, second)| [first, second])
        .collect();
    assert_eq!(
        flat,
        super::BIT_REVERSED_16.to_vec(),
        "register permutation does not reproduce the bit-reversed order"
    );
}
