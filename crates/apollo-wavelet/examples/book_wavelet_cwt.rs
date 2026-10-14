//! Continuous wavelet scalogram with the Morlet wavelet.
//!
//! The CWT correlates the signal against the analysis wavelet at each
//! scale, producing a `[scales, time]` coefficient matrix.

extern crate apollo_wavelet;

use apollo_wavelet::{ContinuousWavelet, CwtPlan};

fn main() {
    let plan = CwtPlan::new(4, vec![1.0, 2.0], ContinuousWavelet::Morlet { omega0: 5.0 })
        .expect("valid plan");
    let signal = [1.0, -0.5, 0.25, 2.0];
    let scalogram = plan.transform(&signal).expect("transform");
    println!("scales : {:?}", scalogram.scales());
    println!("shape  : {:?}", scalogram.values().shape());
    assert_eq!(scalogram.scales(), &[1.0, 2.0]);
    assert_eq!(scalogram.values().shape(), [2, 4]);

    println!("CWT scalogram assertions passed");
}
