//! STFT round-trip through overlap-add inversion.
//!
//! The inverse redistributes each frame's energy to its source samples and
//! normalizes by the summed window weights, recovering the signal the
//! forward transform analyzed.

extern crate apollo_stft;
extern crate leto;

use apollo_stft::StftPlan;
use leto::Array1;

fn main() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from(vec![
        1.0, -1.0, 0.5, 2.0, -0.75, 0.25, 1.5, -0.5, 0.125, 0.875, -1.25, 0.75,
    ]);
    let spectrum = plan.forward(&signal).expect("forward");
    let recovered = plan.inverse(&spectrum, signal.size()).expect("inverse");
    assert_eq!(recovered.size(), signal.size());

    let mut max_error = 0.0f64;
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        max_error = max_error.max((actual - expected).abs());
    }
    println!("max round-trip error : {max_error:e}");
    assert!(max_error < 1.0e-8);

    println!("STFT round-trip assertions passed");
}
