//! Fractional Fourier transform at integer orders.
//!
//! Order zero is the identity, and order one is the Fourier transform, so
//! forward-then-inverse at order one recovers the signal.

extern crate apollo_frft;
extern crate eunomia;
extern crate leto;

use apollo_frft::FrftPlan;
use eunomia::Complex64;
use leto::Array1;

fn main() {
    let input = Array1::from(vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]);
    let identity = FrftPlan::new(2, 0.0).expect("valid plan");
    assert_eq!(
        identity.forward(&input).expect("order-zero forward"),
        input,
        "order zero is the identity"
    );

    let n = 8_usize;
    let plan = FrftPlan::new(n, 1.0).expect("valid plan");
    let signal = Array1::from_shape_fn([n], |[i]| {
        Complex64::new((i as f64 * 0.17).cos(), (i as f64 * 0.23).sin())
    });
    let recovered = plan
        .inverse(&plan.forward(&signal).expect("forward"))
        .expect("inverse");
    for ((recovered, original), index) in recovered.iter().zip(signal.iter()).zip(0..) {
        assert!(
            (recovered.re - original.re).abs() < 1e-12,
            "order-one round-trip recovers sample {index}: {recovered} vs {original}"
        );
        assert!(
            (recovered.im - original.im).abs() < 1e-12,
            "order-one round-trip has no imaginary drift at {index}"
        );
    }
    println!("ok");
}
