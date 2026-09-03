//! Correctness for the register-resident four-step at N = 1024.
//!
//! The direct DFT is the analytical authority; the differential against the
//! batched route pins interchangeability behind one gate.

use super::four_step_resident;
use crate::application::execution::kernel::components::test_support::executed_or_declined_untouched;
use eunomia::Complex64;
use std::f64::consts::TAU;

pub(super) fn dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
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

pub(super) fn signal() -> Vec<Complex64> {
    (0..1024)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

pub(super) fn tolerance(input: &[Complex64]) -> f64 {
    let l1: f64 = input.iter().map(|v| v.re.hypot(v.im)).sum();
    16.0 * 10.0 * (f64::EPSILON / 2.0) * l1
}

pub(super) fn worst(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.re - y.re).hypot(x.im - y.im))
        .fold(0.0f64, f64::max)
}

#[test]
fn forward_matches_the_direct_transform_when_width_is_supported() {
    let src = signal();
    let mut data = src.clone();
    let executed = four_step_resident::<f64, false>(&mut data);
    if !executed_or_declined_untouched(&src, &data, executed) {
        return;
    }
    let (err, bound) = (worst(&data, &dft(&src, false)), tolerance(&src));
    assert!(err <= bound, "forward differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn inverse_matches_the_direct_transform_when_width_is_supported() {
    let src = signal();
    let mut data = src.clone();
    let executed = four_step_resident::<f64, true>(&mut data);
    if !executed_or_declined_untouched(&src, &data, executed) {
        return;
    }
    let (err, bound) = (worst(&data, &dft(&src, true)), tolerance(&src));
    assert!(err <= bound, "inverse differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn forward_then_inverse_recovers_the_input_when_width_is_supported() {
    let src = signal();
    let mut data = src.clone();
    let executed = four_step_resident::<f64, false>(&mut data);
    if !executed_or_declined_untouched(&src, &data, executed) {
        return;
    }
    assert!(four_step_resident::<f64, true>(&mut data));
    let n = 1024.0;
    let bound = tolerance(&src) * n;
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
fn matches_the_batched_route_when_width_is_supported() {
    use crate::application::execution::kernel::components::batched;
    let src = signal();
    let mut ours = src.clone();
    let executed = four_step_resident::<f64, false>(&mut ours);
    if !executed_or_declined_untouched(&src, &ours, executed) {
        return;
    }

    let mut theirs = src.clone();
    let mut scratch = vec![Complex64::default(); batched::scratch_len(1024)];
    batched::four_step_batched::<f64, false>(&mut theirs, &mut scratch);

    let bound = 2.0 * tolerance(&src);
    let err = worst(&ours, &theirs);
    assert!(err <= bound, "routes differ by {err:.3e} > {bound:.3e}");
}

#[test]
fn lengths_other_than_1024_report_unhandled_untouched() {
    let mut data = vec![Complex64::new(1.0, -2.0); 256];
    let before = data.clone();
    assert!(!four_step_resident::<f64, false>(&mut data));
    assert_eq!(data, before, "a declined length must not be mutated");
}

#[test]
fn resident_plans_are_shared_across_threads() {
    // The cache is keyed per thread. If a thread that misses builds its own
    // `ResidentPlan` rather than taking the shared one, the stage twiddles and
    // the 32 x 32 four-step matrix exist once per thread and nothing evicts
    // them. Pointer identity is the direct evidence: one allocation, or one
    // per thread.
    use super::ResidentPlanCache;

    let addresses: Vec<usize> = (0..2)
        .map(|_| {
            std::thread::spawn(|| {
                std::sync::Arc::as_ptr(&<f64 as ResidentPlanCache>::cached_resident_plan::<false>(
                    super::ROW * super::ROW,
                )) as usize
            })
        })
        .map(|handle| handle.join().expect("plan builder thread must not panic"))
        .collect();

    assert_eq!(
        addresses[0],
        addresses[1],
        "each thread built its own resident plan for N = {}",
        super::ROW * super::ROW
    );
}
