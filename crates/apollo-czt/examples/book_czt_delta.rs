//! Chirp Z-transform of a delta signal on the unit circle.
//!
//! With start `a = 1`, the delta `[1, 0, 0, 0]` has CZT value 1 at every
//! output point regardless of the spiral step `w`.

extern crate apollo_czt;
extern crate eunomia;
extern crate leto;

use apollo_czt::CztPlan;
use eunomia::Complex64;
use leto::Array1;

fn main() {
    let plan = CztPlan::new(4, 4, Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0))
        .expect("valid CZT parameters");
    let input = Array1::from(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let output = plan.forward(&input).expect("forward CZT");
    assert_eq!(output.len(), 4, "CZT emits one value per output point");
    for (index, value) in output.iter().enumerate() {
        assert!(
            (value.re - 1.0).abs() < 1e-12,
            "delta CZT is 1 at point {index}, got {value}"
        );
        assert!(value.im.abs() < 1e-12, "delta CZT has no imaginary part");
    }
    println!("ok");
}
