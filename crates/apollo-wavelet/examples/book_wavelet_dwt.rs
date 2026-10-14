//! Discrete wavelet round-trip with Daubechies-4.
//!
//! Two decomposition levels over eight samples, then the inverse
//! reconstructs the signal from the approximation and detail coefficients.

extern crate apollo_wavelet;

use apollo_wavelet::{DiscreteWavelet, DwtPlan};

fn main() {
    let plan = DwtPlan::new(8, 2, DiscreteWavelet::Daubechies4).expect("valid plan");
    let signal = [1.0, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let coeffs = plan.forward(&signal).expect("forward");
    println!("approximation length : {}", coeffs.approximation().len());
    println!("detail levels        : {}", coeffs.detail_levels().count());
    assert_eq!(coeffs.approximation().len(), 2);
    assert_eq!(coeffs.detail_levels().count(), 2);

    let recovered = plan.inverse(&coeffs).expect("inverse");
    let mut max_error = 0.0f64;
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        max_error = max_error.max((actual - expected).abs());
    }
    println!("max round-trip error : {max_error:e}");
    assert!(max_error < 1.0e-12);

    println!("DWT round-trip assertions passed");
}
