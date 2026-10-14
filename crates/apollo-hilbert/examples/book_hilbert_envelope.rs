//! Hilbert envelope of a pure tone.
//!
//! The analytic signal of `A·cos(2π·k·n/N)` is `A·e^{2πi·k·n/N}`, so its
//! envelope is the constant amplitude `A` and the quadrature component is
//! the sine at the same amplitude.

extern crate apollo_hilbert;

use apollo_hilbert::HilbertPlan;
use std::f64::consts::PI;

fn main() {
    let plan = HilbertPlan::new(16).expect("valid plan");
    let signal: Vec<f64> = (0..16)
        .map(|n| 1.5 * (2.0 * PI * 2.0 * f64::from(n) / 16.0).cos())
        .collect();
    let analytic = plan.analytic_signal(&signal).expect("analytic signal");
    assert_eq!(analytic.len(), 16);

    let envelope = analytic.envelope();
    let mut max_error = 0.0f64;
    for value in envelope.iter() {
        max_error = max_error.max((value - 1.5).abs());
    }
    println!("max envelope error : {max_error:e}");
    assert!(max_error < 1.0e-9);

    // The quadrature component carries the sine at the same amplitude.
    let quadrature = analytic.quadrature();
    assert!((quadrature[0] - 0.0).abs() < 1.0e-9);
    assert!((quadrature[2] - 1.5).abs() < 1.0e-9);

    println!("Hilbert envelope assertions passed");
}
