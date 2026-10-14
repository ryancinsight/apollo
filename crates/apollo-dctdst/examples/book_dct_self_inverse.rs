//! DCT-IV self-inversion on a four-sample signal.
//!
//! DCT-IV is self-inverse up to a scale factor: applying the forward
//! transform twice recovers `(N/2) * x`.

extern crate apollo_dctdst;

use apollo_dctdst::{DctDstPlan, RealTransformKind};

fn main() {
    let signal = [1.0, 2.0, 3.0, 4.0_f64];
    let plan = DctDstPlan::new(signal.len(), RealTransformKind::DctIV).expect("valid plan");
    let spectrum = plan.forward(&signal).expect("forward DCT-IV");
    let recovered = plan.forward(&spectrum).expect("second DCT-IV");
    for ((recovered, original), index) in recovered.iter().zip(signal.iter()).zip(0..) {
        let expected = 2.0 * original;
        assert!(
            (recovered - expected).abs() < 1e-12,
            "DCT-IV twice scales by N/2 at index {index}: {recovered} vs {expected}"
        );
    }
    println!("ok");
}
