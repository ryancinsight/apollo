//! Tests for the 2D FFT plan.

use super::*;
use approx::assert_abs_diff_eq;

#[test]
fn roundtrip_recovers_asymmetric_inputs() {
    for &(nx, ny) in &[(7, 13), (13, 7), (16, 24), (24, 16), (120, 360)] {
        let shape = Shape2D::new(nx, ny).expect("test dimensions are non-zero");
        let plan = FftPlan2D::new(shape);
        let input = Array2::from_shape_fn((nx, ny), |(i, j)| {
            let x = i as f64 / nx as f64;
            let y = j as f64 / ny as f64;
            (std::f64::consts::TAU * x).sin()
                + 0.5 * (2.0 * std::f64::consts::TAU * y).cos()
                + 0.125 * ((i * 3 + j * 5) as f64).sin()
        });

        let spectrum = plan.forward(&input);
        let recovered = plan.inverse(&spectrum);

        for ((i, j), expected) in input.indexed_iter() {
            assert_abs_diff_eq!(recovered[(i, j)], *expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn complex_inverse_inplace_is_normalized() {
    let nx = 5;
    let ny = 9;
    let shape = Shape2D::new(nx, ny).expect("test dimensions are non-zero");
    let plan = FftPlan2D::new(shape);
    let input = Array2::from_shape_fn((nx, ny), |(i, j)| {
        Complex64::new((i + 2 * j) as f64, (2 * i + j) as f64 / 3.0)
    });

    let mut spectrum = input.clone();
    plan.forward_complex_inplace(&mut spectrum);
    plan.inverse_complex_inplace(&mut spectrum);

    for ((i, j), expected) in input.indexed_iter() {
        assert_abs_diff_eq!(spectrum[(i, j)].re, expected.re, epsilon = 1e-9);
        assert_abs_diff_eq!(spectrum[(i, j)].im, expected.im, epsilon = 1e-9);
    }
}
