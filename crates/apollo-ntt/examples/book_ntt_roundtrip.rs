//! Number-theoretic round-trip over the default prime.
//!
//! `NttPlan::new` builds the twiddle tables for a power-of-two length over
//! the crate default NTT-friendly prime; the inverse recovers the input
//! residues exactly.

extern crate apollo_ntt;
extern crate leto;

use apollo_ntt::NttPlan;
use leto::Array1;

fn main() {
    let plan = NttPlan::new(4).expect("valid plan");
    println!("modulus : {}", plan.modulus());
    let input = Array1::from(vec![1u64, 2, 3, 4]);
    let spectrum = plan.forward(&input).expect("forward");
    assert_eq!(spectrum.size(), 4);

    let recovered = plan.inverse(&spectrum).expect("inverse");
    assert_eq!(
        recovered.as_slice().expect("contiguous"),
        input.as_slice().expect("contiguous")
    );

    println!("NTT round-trip assertions passed");
}
