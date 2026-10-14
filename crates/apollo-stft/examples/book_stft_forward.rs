//! STFT forward transform.
//!
//! Each frame is Hann-windowed and Fourier-transformed; the spectrum holds
//! `frame_count * frame_len` complex bins.

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
    println!("signal length   : {}", signal.size());
    println!("spectrum length : {}", spectrum.size());
    assert_eq!(
        spectrum.size(),
        plan.frame_count(signal.size()) * plan.frame_len()
    );
    assert_eq!(spectrum.size(), 32);

    println!("STFT forward assertions passed");
}
