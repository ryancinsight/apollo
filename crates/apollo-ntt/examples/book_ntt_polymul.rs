//! Exact polynomial multiplication through the NTT.
//!
//! Forward-transform both operands, multiply pointwise modulo the prime,
//! and inverse-transform: `(1 + 2x)(3 + 4x) = 3 + 10x + 8x^2` with no
//! rounding error.

extern crate apollo_ntt;
extern crate leto;

use apollo_ntt::NttPlan;
use leto::Array1;

fn main() {
    let plan = NttPlan::new(4).expect("valid plan");
    let modulus = plan.modulus();
    let poly_a = Array1::from(vec![1u64, 2, 0, 0]);
    let poly_b = Array1::from(vec![3u64, 4, 0, 0]);

    let ntt_a = plan.forward(&poly_a).expect("forward a");
    let ntt_b = plan.forward(&poly_b).expect("forward b");
    let product_ntt = Array1::from(
        ntt_a
            .as_slice()
            .expect("contiguous")
            .iter()
            .zip(ntt_b.as_slice().expect("contiguous").iter())
            .map(|(a, b)| u64::try_from((u128::from(*a) * u128::from(*b)) % u128::from(modulus)).expect("invariant: remainder below u64 modulus"))
            .collect::<Vec<_>>(),
    );
    let product = plan.inverse(&product_ntt).expect("inverse");

    assert_eq!(product.as_slice().expect("contiguous"), &[3u64, 10, 8, 0]);
    println!(
        "product coefficients : {:?}",
        product.as_slice().expect("contiguous")
    );

    println!("NTT polynomial-multiplication assertions passed");
}
