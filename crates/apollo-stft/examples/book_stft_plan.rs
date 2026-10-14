//! STFT plan construction.
//!
//! `StftPlan::new` validates the frame and hop lengths and builds the
//! Hann-windowed COLA plan; `frame_count` reports how many frames a
//! signal of a given length produces.

extern crate apollo_stft;

use apollo_stft::StftPlan;

fn main() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    println!("frame length : {}", plan.frame_len());
    println!("hop length   : {}", plan.hop_len());
    assert_eq!(plan.frame_len(), 8);
    assert_eq!(plan.hop_len(), 4);

    // 1 + ceil(12 / 4) frames cover a 12-sample signal.
    assert_eq!(plan.frame_count(12), 4);
    println!("frames for 12 samples : {}", plan.frame_count(12));

    println!("STFT plan assertions passed");
}
