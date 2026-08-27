//! Correctness for the planar-register four-step at N = 1024.
//!
//! The direct DFT is the analytical authority; the differential against the
//! interleaved resident kernel pins that the two register shapes compute the
//! same transform through independent shuffle networks.

use super::super::tests::{dft, signal, tolerance, worst};
use super::four_step_planar;
use eunomia::Complex64;

#[test]
fn forward_matches_the_direct_transform() {
    let src = signal();
    let mut data = src.clone();
    assert!(
        four_step_planar::<f64, false>(&mut data),
        "this host's dispatched width must run the planar rows"
    );
    let (err, bound) = (worst(&data, &dft(&src, false)), tolerance(&src));
    assert!(err <= bound, "forward differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn inverse_matches_the_direct_transform() {
    let src = signal();
    let mut data = src.clone();
    assert!(four_step_planar::<f64, true>(&mut data));
    let (err, bound) = (worst(&data, &dft(&src, true)), tolerance(&src));
    assert!(err <= bound, "inverse differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn forward_then_inverse_recovers_the_input() {
    let src = signal();
    let mut data = src.clone();
    assert!(four_step_planar::<f64, false>(&mut data));
    assert!(four_step_planar::<f64, true>(&mut data));
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
fn matches_the_interleaved_resident_kernel_within_rounding() {
    let src = signal();
    let mut planar = src.clone();
    assert!(four_step_planar::<f64, false>(&mut planar));

    let mut interleaved = src.clone();
    assert!(super::super::four_step_resident::<f64, false>(
        &mut interleaved
    ));

    let bound = 2.0 * tolerance(&src);
    let err = worst(&planar, &interleaved);
    assert!(err <= bound, "shapes differ by {err:.3e} > {bound:.3e}");
}

#[test]
fn lengths_other_than_1024_report_unhandled_untouched() {
    let mut data = vec![Complex64::new(1.0, -2.0); 256];
    let before = data.clone();
    assert!(!four_step_planar::<f64, false>(&mut data));
    assert_eq!(data, before, "a declined length must not be mutated");
}
