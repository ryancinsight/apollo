//! Spherical harmonic transform of a constant field.
//!
//! A constant `1/sqrt(4 pi)` field carries only the `Y_00` harmonic with
//! unit coefficient; every other coefficient vanishes and the inverse
//! recovers the field.

extern crate apollo_sht;
extern crate leto;

use apollo_sht::ShtPlan;
use leto::Array2;
use std::f64::consts::PI;

fn main() {
    let plan = ShtPlan::new(6, 13, 2).expect("valid grid");
    let samples = Array2::from_elem(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        1.0 / (4.0 * PI).sqrt(),
    );
    let coefficients = plan.forward_real(&samples).expect("forward SHT");
    let dc = coefficients.get(0, 0);
    assert!(
        (dc.re - 1.0).abs() < 1e-12,
        "constant field has unit Y_00, got {dc}"
    );
    assert!(dc.im.abs() < 1e-12, "Y_00 of a real field is real");
    for degree in 0..=2_isize {
        for order in -degree..=degree {
            if degree == 0 && order == 0 {
                continue;
            }
            let value = coefficients.get(degree as usize, order);
            assert!(
                value.norm() < 1e-12,
                "constant field has no Y_{degree}{order}, got {value}"
            );
        }
    }
    let recovered = plan.inverse_real(&coefficients).expect("inverse SHT");
    for (recovered, original) in recovered.iter().zip(samples.iter()) {
        assert!(
            (recovered - original).abs() < 1e-12,
            "round-trip recovers the field: {recovered} vs {original}"
        );
    }
    println!("ok");
}
