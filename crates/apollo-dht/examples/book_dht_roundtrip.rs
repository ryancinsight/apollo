//! Discrete Hartley round-trip.
//!
//! The DHT is real-to-real and self-inverse: the same transform with a
//! `1/N` scale recovers the input.

extern crate apollo_dht;

use apollo_dht::DhtPlan;

fn main() {
    let plan = DhtPlan::new(4).expect("valid plan");
    let signal = [1.0, -2.0, 0.5, 2.25];
    let spectrum = plan.forward(&signal).expect("forward");
    println!("spectrum length : {}", spectrum.len());
    assert_eq!(spectrum.len(), 4);

    let recovered = plan.inverse(&spectrum).expect("inverse");
    let mut max_error = 0.0f64;
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        max_error = max_error.max((actual - expected).abs());
    }
    println!("max round-trip error : {max_error:e}");
    assert!(max_error < 1.0e-12);

    println!("DHT round-trip assertions passed");
}
