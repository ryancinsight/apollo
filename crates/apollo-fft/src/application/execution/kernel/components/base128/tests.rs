//! Correctness for the 128-point base butterfly. The direct DFT is the
//! analytical authority.

use super::butterfly::{transform_128, Plan128};
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

fn signal() -> Vec<Complex64> {
    (0..128)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.043 * x).sin(), 0.25 * (0.029 * x).cos())
        })
        .collect()
}

fn tolerance(input: &[Complex64]) -> f64 {
    let l1: f64 = input.iter().map(|v| v.re.hypot(v.im)).sum();
    // The direct oracle performs at most eight rounded scalar operations per
    // input term; the radix-2 FFT performs at most sixteen per one of seven
    // stages. Higham's gamma_k = ku / (1 - ku), u = epsilon/2, bounds their
    // combined first-order error against the input L1 norm.
    let operations = 8.0 * input.len() as f64 + 16.0 * 7.0;
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
    let operations = 8.0 * input.len() as f32 + 16.0 * 7.0;
    let scaled_epsilon = operations * (f32::EPSILON / 2.0);
    scaled_epsilon / (1.0 - scaled_epsilon) * l1
}

#[test]
fn forward_matches_the_direct_transform() {
    let src = signal();
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
    let src = signal();
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
    let src = signal();
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
    let src = signal();
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
    use super::butterfly::phase_meter::{CALLS, PHASES};
    use std::sync::atomic::Ordering;

    CALLS.store(0, Ordering::Relaxed);
    for phase in &PHASES {
        phase.store(0, Ordering::Relaxed);
    }

    let source = signal();
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
    let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D { n: 128 });
    let Some(base) = plan.base128.as_ref() else {
        assert!(
            plan.twiddle_fwd.is_some(),
            "the incumbent route retains its forward twiddles"
        );
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

    let source = signal();
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
fn f64_dynamic_plan_clones_execute_inverse_concurrently() {
    let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D { n: 128 });
    if plan.base128.is_none() {
        assert!(
            plan.twiddle_fwd.is_some(),
            "the incumbent route retains its forward twiddles"
        );
        return;
    }

    let first_input = signal();
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
    let first_bound = 2.0 * tolerance(&first_input) / 128.0;
    let second_bound = 2.0 * tolerance(&second_input) / 128.0;
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
    let plan = crate::FftPlan1D::<f32>::new(crate::Shape1D { n: 128 });
    if plan.base128.is_none() {
        assert!(
            plan.twiddle_fwd.is_some(),
            "the incumbent route retains its forward twiddles"
        );
        return;
    }

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
    let first_bound = 2.0 * reduced_tolerance(&first_input) / 128.0;
    let second_bound = 2.0 * reduced_tolerance(&second_input) / 128.0;
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
