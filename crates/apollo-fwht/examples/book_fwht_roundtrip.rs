//! Fast Walsh-Hadamard round-trip.
//!
//! The FWHT expands the signal in Walsh functions with only additions and
//! subtractions; the transform is self-inverse up to a `1/N` scale, so the
//! inverse recovers the input.

extern crate apollo_fwht;
extern crate leto;

use apollo_fwht::FwhtPlan;
use leto::Array1;

fn main() {
    let plan = FwhtPlan::new(8).expect("valid plan");
    let signal: Array1<f64> =
        Array1::from_vec([8], vec![1.0, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75])
            .expect("valid signal");
    let spectrum = plan.forward(&signal).expect("forward");
    println!("spectrum length : {}", spectrum.size());
    assert_eq!(spectrum.size(), 8);
    // DC bin: the input sum.
    let sum: f64 = signal.iter().sum();
    assert!((spectrum[[0]] - sum).abs() < 1.0e-12);

    let recovered = plan.inverse(&spectrum).expect("inverse");
    let mut max_error = 0.0f64;
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        max_error = max_error.max((actual - expected).abs());
    }
    println!("max round-trip error : {max_error:e}");
    assert!(max_error < 1.0e-12);

    println!("FWHT round-trip assertions passed");
}
