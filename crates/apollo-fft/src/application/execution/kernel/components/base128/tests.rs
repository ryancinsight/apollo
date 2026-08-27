//! Correctness for the 128-point base butterfly. The direct DFT is the
//! analytical authority.

use super::butterfly::transform_128;
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

fn dft_reduced(input: &[Complex32]) -> Vec<Complex32> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0_f32, 0.0_f32);
            for (t, value) in input.iter().enumerate() {
                let angle = -std::f32::consts::TAU * ((k * t) % n) as f32 / n as f32;
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
    if !transform_128::<f64, false>(&mut data) {
        assert_eq!(data, src, "a width decline must not mutate the input");
        return;
    }
    let (err, bound) = (worst(&data, &dft(&src, false)), tolerance(&src));
    assert!(err <= bound, "forward differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn inverse_matches_the_direct_transform() {
    let src = signal();
    let mut data = src.clone();
    if !transform_128::<f64, true>(&mut data) {
        assert_eq!(data, src, "a width decline must not mutate the input");
        return;
    }
    let (err, bound) = (worst(&data, &dft(&src, true)), tolerance(&src));
    assert!(err <= bound, "inverse differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn forward_then_inverse_recovers_the_input() {
    let src = signal();
    let mut data = src.clone();
    if !transform_128::<f64, false>(&mut data) {
        assert_eq!(data, src, "a width decline must not mutate the input");
        return;
    }
    assert!(
        transform_128::<f64, true>(&mut data),
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
fn matches_the_production_route_within_rounding() {
    let src = signal();
    let mut ours = src.clone();
    if !transform_128::<f64, false>(&mut ours) {
        assert_eq!(ours, src, "a width decline must not mutate the input");
        return;
    }

    let mut theirs = src.clone();
    let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D { n: 128 });
    plan.forward_complex_slice_inplace(&mut theirs);

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
    if !transform_128::<f32, false>(&mut data) {
        assert_eq!(data, src, "a width decline must not mutate the input");
        return;
    }

    let expected = dft_reduced(&src);
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
    if !transform_128::<f64, false>(&mut data) {
        assert_eq!(data, source, "a width decline must not mutate the input");
    }

    assert_eq!(CALLS.load(Ordering::Relaxed), 0);
    let recorded = std::array::from_fn(|index| PHASES[index].load(Ordering::Relaxed));
    assert_eq!(recorded, [0; 3]);
}
