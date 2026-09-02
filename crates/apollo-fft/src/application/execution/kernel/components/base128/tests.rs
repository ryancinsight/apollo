//! Correctness for the 64- and 128-point base butterflies. The direct DFT is
//! the analytical authority.

use super::instance_major::{transform_128, Plan128};
use super::instance_major::{transform_64, Plan64};
use eunomia::{Complex32, Complex64};
use std::f64::consts::TAU;

fn dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (t, v) in input.iter().enumerate() {
                let (s, c) = (sign * TAU * ((k * t) % n) as f64 / n as f64).sin_cos();
                re += v.re * c - v.im * s;
                im += v.re * s + v.im * c;
            }
            Complex64::new(re, im)
        })
        .collect()
}

fn signal(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.043 * x).sin(), 0.25 * (0.029 * x).cos())
        })
        .collect()
}

fn tolerance(input: &[Complex64]) -> f64 {
    let l1: f64 = input.iter().map(|v| v.re.hypot(v.im)).sum();
    // The direct oracle performs at most eight rounded scalar operations per
    // input term; the radix-2 FFT performs at most sixteen per stage.
    // Higham's gamma_k = ku / (1 - ku), u = epsilon/2, bounds their combined
    // first-order error against the input L1 norm.
    let operations = 8.0 * input.len() as f64 + 16.0 * f64::from(input.len().ilog2());
    let scaled_epsilon = operations * (f64::EPSILON / 2.0);
    scaled_epsilon / (1.0 - scaled_epsilon) * l1
}

fn worst(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.re - y.re).hypot(x.im - y.im))
        .fold(0.0f64, f64::max)
}

fn dft_reduced(input: &[Complex32], inverse: bool) -> Vec<Complex32> {
    let n = input.len();
    let sign = if inverse { 1.0_f32 } else { -1.0_f32 };
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0_f32, 0.0_f32);
            for (t, value) in input.iter().enumerate() {
                let angle = sign * std::f32::consts::TAU * ((k * t) % n) as f32 / n as f32;
                let (sine, cosine) = angle.sin_cos();
                re += value.re * cosine - value.im * sine;
                im += value.re * sine + value.im * cosine;
            }
            Complex32::new(re, im)
        })
        .collect()
}

fn reduced_tolerance(input: &[Complex32]) -> f32 {
    let l1: f32 = input.iter().map(|value| value.re.hypot(value.im)).sum();
    let operations = 8.0 * input.len() as f32 + 16.0 * input.len().ilog2() as f32;
    let scaled_epsilon = operations * (f32::EPSILON / 2.0);
    scaled_epsilon / (1.0 - scaled_epsilon) * l1
}

/// On a host that declines both base routes the plan still transforms through
/// the incumbent twiddle route; the fallback asserts that route's round-trip
/// correctness instead of merely asserting twiddles exist.
fn assert_incumbent_route_round_trips(plan: &crate::FftPlan1D<f64>, n: usize) {
    if plan.base128.is_some() || plan.base64.is_some() {
        return;
    }
    let source = signal(n);
    let mut data = source.clone();
    plan.forward_complex_slice_inplace(&mut data);
    plan.inverse_complex_slice_inplace(&mut data);
    let error = worst(&data, &source);
    let bound = 2.0 * tolerance(&source);
    assert!(
        error <= bound,
        "the incumbent route round trips by {error:.3e} > {bound:.3e}"
    );
}

fn assert_base64_matches_direct<const INVERSE: bool>() {
    let source = signal(64);
    let mut actual = source.clone();
    let Some(plan) = Plan64::<f64>::new_if_supported::<INVERSE>() else {
        assert_eq!(actual, source, "a width decline must not mutate the input");
        return;
    };

    assert!(transform_64::<f64, INVERSE>(&mut actual, &plan));
    let expected = dft(&source, INVERSE);
    let error = worst(&actual, &expected);
    let bound = tolerance(&source);
    assert!(
        error <= bound,
        "base-64 transform differs by {error:.3e} > {bound:.3e}"
    );
}

#[test]
fn base64_forward_matches_the_direct_transform() {
    assert_base64_matches_direct::<false>();
}

#[test]
fn base64_inverse_matches_the_direct_transform() {
    assert_base64_matches_direct::<true>();
}

#[test]
fn base64_forward_then_inverse_recovers_the_input() {
    let source = signal(64);
    let mut actual = source.clone();
    let Some(forward) = Plan64::<f64>::new_if_supported::<false>() else {
        assert_eq!(actual, source, "a width decline must not mutate the input");
        return;
    };
    let inverse = Plan64::<f64>::new_if_supported::<true>()
        .expect("the same exact-width capability serves both directions");

    assert!(transform_64::<f64, false>(&mut actual, &forward));
    assert!(transform_64::<f64, true>(&mut actual, &inverse));
    let scale = 64.0;
    let error = actual
        .iter()
        .zip(&source)
        .map(|(value, reference)| {
            (value.re - scale * reference.re).hypot(value.im - scale * reference.im)
        })
        .fold(0.0_f64, f64::max);
    let bound = 2.0 * scale * tolerance(&source);
    assert!(
        error <= bound,
        "base-64 round trip differs by {error:.3e} > {bound:.3e}"
    );
}

#[test]
fn base64_reduced_precision_matches_the_direct_transform() {
    let source: Vec<Complex32> = (0..64)
        .map(|index| {
            let x = index as f32;
            Complex32::new((0.043 * x).sin(), 0.25 * (0.029 * x).cos())
        })
        .collect();
    let mut actual = source.clone();
    let Some(plan) = Plan64::<f32>::new_if_supported::<false>() else {
        assert_eq!(actual, source, "a width decline must not mutate the input");
        return;
    };

    assert!(transform_64::<f32, false>(&mut actual, &plan));
    let expected = dft_reduced(&source, false);
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(value, reference)| (value.re - reference.re).hypot(value.im - reference.im))
        .fold(0.0_f32, f32::max);
    let bound = reduced_tolerance(&source);
    assert!(
        error <= bound,
        "reduced base-64 transform differs by {error:.3e} > {bound:.3e}"
    );
}

#[test]
fn dynamic_base64_plan_owns_only_the_selected_route() {
    let plan = crate::FftPlan1D::<f64>::new(
        crate::Shape1D::new(64).expect("invariant: shape lengths are non-zero"),
    );
    let Some(base) = plan.base64.as_ref() else {
        assert_incumbent_route_round_trips(&plan, 64);
        return;
    };

    assert!(plan.base128.is_none());
    assert!(
        plan.twiddle_fwd.is_none(),
        "the selected base route must not retain incumbent twiddles"
    );
    assert!(!base.inverse_is_initialized());
    let clone = plan.clone();
    assert!(std::sync::Arc::ptr_eq(
        base,
        clone
            .base64
            .as_ref()
            .expect("a clone preserves the selected route")
    ));

    let source = signal(64);
    let mut actual = source.clone();
    plan.forward_complex_slice_inplace(&mut actual);
    assert!(!base.inverse_is_initialized());
    plan.inverse_complex_slice_inplace(&mut actual);
    assert!(base.inverse_is_initialized());
    let error = worst(&actual, &source);
    let bound = 2.0 * tolerance(&source);
    assert!(
        error <= bound,
        "dynamic base-64 round trip differs by {error:.3e} > {bound:.3e}"
    );
}

#[test]
fn forward_matches_the_direct_transform() {
    let src = signal(128);
    let mut data = src.clone();
    let Some(plan) = Plan128::<f64>::new_if_supported::<false>() else {
        assert_eq!(data, src, "a width decline must not mutate the input");
        return;
    };
    assert!(transform_128::<f64, false>(&mut data, &plan));
    let (err, bound) = (worst(&data, &dft(&src, false)), tolerance(&src));
    assert!(err <= bound, "forward differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn inverse_matches_the_direct_transform() {
    let src = signal(128);
    let mut data = src.clone();
    let Some(plan) = Plan128::<f64>::new_if_supported::<true>() else {
        assert_eq!(data, src, "a width decline must not mutate the input");
        return;
    };
    assert!(transform_128::<f64, true>(&mut data, &plan));
    let (err, bound) = (worst(&data, &dft(&src, true)), tolerance(&src));
    assert!(err <= bound, "inverse differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn forward_then_inverse_recovers_the_input() {
    let src = signal(128);
    let mut data = src.clone();
    let Some(forward_plan) = Plan128::<f64>::new_if_supported::<false>() else {
        assert_eq!(data, src, "a width decline must not mutate the input");
        return;
    };
    let inverse_plan = Plan128::<f64>::new_if_supported::<true>()
        .expect("the same exact-width capability serves both directions");
    assert!(transform_128::<f64, false>(&mut data, &forward_plan));
    assert!(
        transform_128::<f64, true>(&mut data, &inverse_plan),
        "one direction cannot decline after the same width ran forward"
    );
    let n = 128.0;
    let bound = 2.0 * tolerance(&src) * n;
    let err = data
        .iter()
        .zip(src.iter())
        .map(|(a, b)| (a.re - b.re * n).hypot(a.im - b.im * n))
        .fold(0.0f64, f64::max);
    assert!(
        err <= bound,
        "round trip differs by {err:.3e} > {bound:.3e}"
    );
}

#[test]
fn matches_the_static_incumbent_route_within_rounding() {
    let src = signal(128);
    let mut ours = src.clone();
    let Some(plan) = Plan128::<f64>::new_if_supported::<false>() else {
        assert_eq!(ours, src, "a width decline must not mutate the input");
        return;
    };
    assert!(transform_128::<f64, false>(&mut ours, &plan));

    let mut theirs = src.clone();
    crate::StaticFftPlan1D::<f64, 128>::new().forward_complex_slice_inplace(&mut theirs);

    let bound = 2.0 * tolerance(&src);
    let err = worst(&ours, &theirs);
    assert!(err <= bound, "routes differ by {err:.3e} > {bound:.3e}");
}

#[test]
fn reduced_precision_computes_or_declines_without_mutation() {
    let src: Vec<Complex32> = (0..128)
        .map(|index| {
            let x = index as f32;
            Complex32::new((0.043 * x).sin(), 0.25 * (0.029 * x).cos())
        })
        .collect();
    let mut data = src.clone();
    let Some(plan) = Plan128::<f32>::new_if_supported::<false>() else {
        assert_eq!(data, src, "a width decline must not mutate the input");
        return;
    };
    assert!(transform_128::<f32, false>(&mut data, &plan));

    let expected = dft_reduced(&src, false);
    let error = data
        .iter()
        .zip(&expected)
        .map(|(actual, reference)| (actual.re - reference.re).hypot(actual.im - reference.im))
        .fold(0.0_f32, f32::max);
    let bound = reduced_tolerance(&src);
    assert!(
        error <= bound,
        "reduced-precision forward differs by {error:.3e} > {bound:.3e}"
    );
}

#[cfg(all(windows, target_arch = "x86_64"))]
#[test]
fn comparison_specialization_does_not_record_phases() {
    use super::instance_major::phase_meter::{CALLS, PHASES};
    use std::sync::atomic::Ordering;

    CALLS.store(0, Ordering::Relaxed);
    for phase in &PHASES {
        phase.store(0, Ordering::Relaxed);
    }

    let source = signal(128);
    let mut data = source.clone();
    let Some(plan) = Plan128::<f64>::new_if_supported::<false>() else {
        assert_eq!(data, source, "a width decline must not mutate the input");
        return;
    };
    assert!(transform_128::<f64, false>(&mut data, &plan));

    assert_eq!(CALLS.load(Ordering::Relaxed), 0);
    let recorded = std::array::from_fn(|index| PHASES[index].load(Ordering::Relaxed));
    assert_eq!(recorded, [0; 3]);
}

#[test]
fn dynamic_plan_owns_forward_and_lazily_initializes_inverse() {
    let plan = crate::FftPlan1D::<f64>::new(
        crate::Shape1D::new(128).expect("invariant: shape lengths are non-zero"),
    );
    let Some(base) = plan.base128.as_ref() else {
        assert_incumbent_route_round_trips(&plan, 128);
        return;
    };

    assert!(
        plan.twiddle_fwd.is_none(),
        "the selected base route must not retain incumbent twiddles"
    );
    assert!(!base.inverse_is_initialized());
    let clone = plan.clone();
    assert!(std::sync::Arc::ptr_eq(
        base,
        clone
            .base128
            .as_ref()
            .expect("a clone preserves the selected route")
    ));

    let source = signal(128);
    let mut data = source.clone();
    plan.forward_complex_slice_inplace(&mut data);
    assert!(
        !base.inverse_is_initialized(),
        "forward execution must not initialize inverse state"
    );
    plan.inverse_complex_slice_inplace(&mut data);
    assert!(base.inverse_is_initialized());
    let clone = plan.clone();
    assert!(std::sync::Arc::ptr_eq(
        base,
        clone
            .base128
            .as_ref()
            .expect("a clone preserves initialized inverse state")
    ));
}

#[test]
fn dynamic_split_plans_share_complete_twiddle_tables() {
    for n in [256usize, 512] {
        let plan = crate::FftPlan1D::<f64>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let Some(_) = plan.base128.as_ref() else {
            assert_incumbent_route_round_trips(&plan, n);
            continue;
        };
        let forward = plan
            .twiddle_fwd
            .as_ref()
            .expect("a split base route retains its complete forward table");
        assert_eq!(forward.len(), n - 1);
        let cached = <f64 as crate::application::execution::kernel::mixed_radix::MixedRadixScalar>::cached_twiddle_fwd(n);
        assert!(
            std::sync::Arc::ptr_eq(forward, &cached),
            "the plan must share the process cache allocation"
        );
        assert!(plan.twiddle_inv.get().is_none());

        let clone = plan.clone();
        assert!(std::sync::Arc::ptr_eq(
            forward,
            clone
                .twiddle_fwd
                .as_ref()
                .expect("a split-plan clone shares its forward table")
        ));

        let source = signal(n);
        let mut data = source.clone();
        plan.forward_complex_slice_inplace(&mut data);
        assert!(
            plan.twiddle_inv.get().is_none(),
            "forward execution must not initialize inverse twiddles"
        );
        plan.inverse_complex_slice_inplace(&mut data);
        let inverse = plan
            .twiddle_inv
            .get()
            .expect("inverse execution initializes the complete inverse table");
        assert_eq!(inverse.len(), n - 1);
        let initialized_clone = plan.clone();
        assert!(std::sync::Arc::ptr_eq(
            inverse,
            initialized_clone
                .twiddle_inv
                .get()
                .expect("an initialized split-plan clone shares inverse twiddles")
        ));
    }
}

#[test]
fn dynamic_split_plans_normalize_by_full_length() {
    for n in super::BASE_SPLIT_LENGTHS {
        let plan = crate::FftPlan1D::<f64>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let source = signal(n);
        let mut actual = source.clone();
        let Some(_) = plan.base128.as_ref() else {
            assert_incumbent_route_round_trips(&plan, n);
            continue;
        };

        plan.forward_complex_slice_inplace(&mut actual);
        plan.inverse_complex_slice_inplace(&mut actual);

        let error = worst(&actual, &source);
        let bound = 2.0 * tolerance(&source);
        assert!(
            error <= bound,
            "N={n} dynamic split round trip differs by {error:.3e} > {bound:.3e}"
        );
    }
}

fn assert_dynamic_split_matches_direct<const INVERSE: bool>() {
    for n in [256usize, 512] {
        let plan = crate::FftPlan1D::<f64>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let source = signal(n);
        let mut actual = source.clone();
        let Some(_) = plan.base128.as_ref() else {
            assert_eq!(actual, source, "a width decline must not mutate the input");
            continue;
        };

        if INVERSE {
            plan.inverse_complex_slice_inplace(&mut actual);
        } else {
            plan.forward_complex_slice_inplace(&mut actual);
        }
        let normalization = if INVERSE { n as f64 } else { 1.0 };
        let expected: Vec<_> = dft(&source, INVERSE)
            .into_iter()
            .map(|value| Complex64::new(value.re / normalization, value.im / normalization))
            .collect();
        let error = worst(&actual, &expected);
        let bound = 2.0 * tolerance(&source) / normalization;
        assert!(
            error <= bound,
            "N={n} split transform differs by {error:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn dynamic_split_forward_matches_the_direct_transform() {
    assert_dynamic_split_matches_direct::<false>();
}

#[test]
fn dynamic_split_inverse_matches_the_direct_transform() {
    assert_dynamic_split_matches_direct::<true>();
}

#[test]
fn reduced_dynamic_split_matches_the_direct_transform() {
    const N: usize = 512;
    let source: Vec<Complex32> = (0..N)
        .map(|index| {
            let x = index as f32;
            Complex32::new((0.043 * x).sin(), 0.25 * (0.029 * x).cos())
        })
        .collect();
    let plan = crate::FftPlan1D::<f32>::new(
        crate::Shape1D::new(N).expect("invariant: shape lengths are non-zero"),
    );
    let mut actual = source.clone();
    let Some(_) = plan.base128.as_ref() else {
        assert_eq!(actual, source, "a width decline must not mutate the input");
        return;
    };

    plan.forward_complex_slice_inplace(&mut actual);
    let expected = dft_reduced(&source, false);
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(value, reference)| (value.re - reference.re).hypot(value.im - reference.im))
        .fold(0.0_f32, f32::max);
    let bound = reduced_tolerance(&source);
    assert!(
        error <= bound,
        "reduced N=512 split differs by {error:.3e} > {bound:.3e}"
    );
}

#[test]
fn f64_dynamic_plan_clones_execute_inverse_concurrently() {
    let plan = crate::FftPlan1D::<f64>::new(
        crate::Shape1D::new(128).expect("invariant: shape lengths are non-zero"),
    );
    let first_input = signal(128);
    let mut second_input = first_input.clone();
    second_input.rotate_left(17);
    let first_expected: Vec<_> = dft(&first_input, true)
        .into_iter()
        .map(|value| Complex64::new(value.re / 128.0, value.im / 128.0))
        .collect();
    let second_expected: Vec<_> = dft(&second_input, true)
        .into_iter()
        .map(|value| Complex64::new(value.re / 128.0, value.im / 128.0))
        .collect();
    let first_bound = tolerance(&first_input) / 128.0;
    let second_bound = tolerance(&second_input) / 128.0;
    let barrier = std::sync::Barrier::new(3);
    let first_plan = plan.clone();
    let second_plan = plan.clone();

    let (first_actual, second_actual) = std::thread::scope(|scope| {
        let first_barrier = &barrier;
        let first = scope.spawn(move || {
            let mut actual = first_input;
            first_barrier.wait();
            first_plan.inverse_complex_slice_inplace(&mut actual);
            actual
        });
        let second_barrier = &barrier;
        let second = scope.spawn(move || {
            let mut actual = second_input;
            second_barrier.wait();
            second_plan.inverse_complex_slice_inplace(&mut actual);
            actual
        });
        barrier.wait();
        (
            first.join().expect("first clone execution must complete"),
            second.join().expect("second clone execution must complete"),
        )
    });

    let first_error = worst(&first_actual, &first_expected);
    let second_error = worst(&second_actual, &second_expected);
    assert!(
        first_error <= first_bound,
        "first concurrent clone differs by {first_error:.3e} > {first_bound:.3e}"
    );
    assert!(
        second_error <= second_bound,
        "second concurrent clone differs by {second_error:.3e} > {second_bound:.3e}"
    );
}

#[test]
fn f32_dynamic_plan_clones_execute_inverse_concurrently() {
    let plan = crate::FftPlan1D::<f32>::new(
        crate::Shape1D::new(128).expect("invariant: shape lengths are non-zero"),
    );
    let first_input: Vec<_> = (0..128)
        .map(|index| {
            let x = index as f32;
            Complex32::new((0.043 * x).sin(), 0.25 * (0.029 * x).cos())
        })
        .collect();
    let mut second_input = first_input.clone();
    second_input.rotate_right(23);
    let first_expected: Vec<_> = dft_reduced(&first_input, true)
        .into_iter()
        .map(|value| Complex32::new(value.re / 128.0, value.im / 128.0))
        .collect();
    let second_expected: Vec<_> = dft_reduced(&second_input, true)
        .into_iter()
        .map(|value| Complex32::new(value.re / 128.0, value.im / 128.0))
        .collect();
    let first_bound = reduced_tolerance(&first_input) / 128.0;
    let second_bound = reduced_tolerance(&second_input) / 128.0;
    let barrier = std::sync::Barrier::new(3);
    let first_plan = plan.clone();
    let second_plan = plan.clone();

    let (first_actual, second_actual) = std::thread::scope(|scope| {
        let first_barrier = &barrier;
        let first = scope.spawn(move || {
            let mut actual = first_input;
            first_barrier.wait();
            first_plan.inverse_complex_slice_inplace(&mut actual);
            actual
        });
        let second_barrier = &barrier;
        let second = scope.spawn(move || {
            let mut actual = second_input;
            second_barrier.wait();
            second_plan.inverse_complex_slice_inplace(&mut actual);
            actual
        });
        barrier.wait();
        (
            first.join().expect("first clone execution must complete"),
            second.join().expect("second clone execution must complete"),
        )
    });

    let first_error = first_actual
        .iter()
        .zip(&first_expected)
        .map(|(actual, expected)| (actual.re - expected.re).hypot(actual.im - expected.im))
        .fold(0.0_f32, f32::max);
    let second_error = second_actual
        .iter()
        .zip(&second_expected)
        .map(|(actual, expected)| (actual.re - expected.re).hypot(actual.im - expected.im))
        .fold(0.0_f32, f32::max);
    assert!(
        first_error <= first_bound,
        "first concurrent clone differs by {first_error:.3e} > {first_bound:.3e}"
    );
    assert!(
        second_error <= second_bound,
        "second concurrent clone differs by {second_error:.3e} > {second_bound:.3e}"
    );
}

/// The gather must run at the plan's native width, not merely produce the
/// right answer: a wrong-width dispatch falls back and still passes value
/// checks, so this asserts the dispatched width handled the pass and that
/// its output matches the scalar strided reference for both block counts.
fn assert_gather_matches_reference<T>(blocks: usize)
where
    T: crate::application::execution::kernel::mixed_radix::MixedRadixScalar
        + hermes_simd::LaneScalar,
{
    let n = blocks * 128;
    let lanes: Vec<T> = (0..2 * n)
        .map(|i| T::from_precise(((i * 37) % 97) as f64 * 0.125 - 4.0))
        .collect();
    let mut reference = vec![T::from_precise(0.0); 2 * n];
    let bits = blocks.trailing_zeros();
    for b in 0..blocks {
        let row = b.reverse_bits() >> (usize::BITS - bits);
        for j in 0..128 {
            reference[(row * 128 + j) * 2] = lanes[(j * blocks + b) * 2];
            reference[(row * 128 + j) * 2 + 1] = lanes[(j * blocks + b) * 2 + 1];
        }
    }
    let mut narrow = vec![T::from_precise(0.0); 2 * n];
    let mut wide = vec![T::from_precise(0.0); 2 * n];
    let (narrow_handled, wide_handled) = if blocks == 2 {
        (
            hermes_simd::vectorize_lanes::<4, T, _>(super::split_boundary::GatherBlocks::<T, 2> {
                src: &lanes,
                dst: &mut narrow,
            })
            .unwrap_or(false),
            hermes_simd::vectorize_lanes::<8, T, _>(super::split_boundary::GatherBlocks::<T, 2> {
                src: &lanes,
                dst: &mut wide,
            })
            .unwrap_or(false),
        )
    } else {
        (
            hermes_simd::vectorize_lanes::<4, T, _>(super::split_boundary::GatherBlocks::<T, 4> {
                src: &lanes,
                dst: &mut narrow,
            })
            .unwrap_or(false),
            hermes_simd::vectorize_lanes::<8, T, _>(super::split_boundary::GatherBlocks::<T, 4> {
                src: &lanes,
                dst: &mut wide,
            })
            .unwrap_or(false),
        )
    };
    // The four-lane request lands on a native or emulated four-lane frame
    // everywhere this suite runs; it must handle and match bit-exactly
    // (the pass moves values, computing nothing).
    assert!(narrow_handled, "four-lane gather must be handled");
    assert_eq!(narrow, reference, "four-lane gather output mismatch");
    // The eight-lane request is handled exactly where the base plan selects
    // the eight-lane layout; there it must match the same reference.
    let plan_is_wide = super::instance_major::Plan128::<T>::new_if_supported::<false>()
        .is_some_and(|plan| plan.native_eight_lanes());
    if plan_is_wide {
        assert!(
            wide_handled,
            "the eight-lane gather must handle where the plan is eight-lane"
        );
        assert_eq!(wide, reference, "eight-lane gather output mismatch");
    }
}

#[test]
fn gather_matches_the_strided_reference_at_both_widths() {
    assert_gather_matches_reference::<f64>(2);
    assert_gather_matches_reference::<f64>(4);
    assert_gather_matches_reference::<f32>(2);
    assert_gather_matches_reference::<f32>(4);
}
