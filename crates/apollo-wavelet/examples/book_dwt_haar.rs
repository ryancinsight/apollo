//! Single-level Haar DWT with closed-form coefficients.
//!
//! The orthonormal Haar stage maps each pair to its scaled sum and
//! difference, and the inverse reassembles the signal exactly.

extern crate apollo_wavelet;

use apollo_wavelet::{DiscreteWavelet, DwtPlan};

fn main() {
    let signal = [4.0, 6.0, 10.0, 12.0];
    let plan = DwtPlan::new(signal.len(), 1, DiscreteWavelet::Haar).expect("valid plan");
    let coefficients = plan.forward(&signal).expect("forward DWT");
    let root_two = 2.0_f64.sqrt();
    for (actual, expected) in coefficients
        .approximation()
        .iter()
        .zip([5.0 * root_two, 11.0 * root_two])
    {
        assert!(
            (actual - expected).abs() < 1e-12,
            "Haar approximation is the scaled pair sum: {actual} vs {expected}"
        );
    }
    for (actual, expected) in coefficients.details().iter().zip([-root_two, -root_two]) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "Haar detail is the scaled pair difference: {actual} vs {expected}"
        );
    }
    let recovered = plan.inverse(&coefficients).expect("inverse DWT");
    for ((recovered, original), index) in recovered.iter().zip(signal.iter()).zip(0..) {
        assert!(
            (recovered - original).abs() < 1e-12,
            "round-trip recovers sample {index}: {recovered} vs {original}"
        );
    }
    println!("ok");
}
