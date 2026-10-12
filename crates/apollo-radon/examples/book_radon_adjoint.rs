//! Parallel-beam Radon forward projection and its adjoint.
//!
//! A 3x3 image projects to a 2-angle sinogram, and backprojection is the
//! exact adjoint of forward projection: `<Ax, Ax> == <x, A'Ax>`.

extern crate apollo_radon;
extern crate leto;

use apollo_radon::RadonPlan;
use leto::Array2;
use std::f64::consts::FRAC_PI_2;

fn main() {
    let plan = RadonPlan::new(3, 3, vec![0.0, FRAC_PI_2], 3, 1.0).expect("valid plan");
    let image = Array2::from_shape_vec([3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .expect("valid image shape");
    let sinogram = plan.forward(&image).expect("forward Radon");
    assert_eq!(
        sinogram.shape(),
        (2, 3),
        "sinogram spans angles by detectors"
    );
    let round_trip = plan.backproject(&sinogram).expect("backproject");
    let sino_energy: f64 = sinogram.values().iter().map(|value| value * value).sum();
    let adjoint_energy: f64 = image
        .iter()
        .zip(round_trip.iter())
        .map(|(pixel, back)| pixel * back)
        .sum();
    assert!(
        (sino_energy - adjoint_energy).abs() < 1e-9,
        "backprojection is the adjoint of projection: {sino_energy} vs {adjoint_energy}"
    );
    println!("ok");
}
