use super::*;
use apollo_fft::PrecisionProfile;
use approx::{assert_abs_diff_eq, assert_relative_eq};
use leto::{SliceArg, Storage};
use ndarray::Array1;
use num_complex::{Complex32, Complex64};

fn reference_plan(n: usize, m: usize) -> CztPlan {
    CztPlan::new(
        n,
        m,
        Complex64::from_polar(1.0, 0.125),
        Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0),
    )
    .expect("valid CZT plan")
}

fn reference_input(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new((i as f64 * 0.31).sin(), (i as f64 * 0.19).cos()))
        .collect()
}

#[test]
fn leto_forward_matches_ndarray_reference() {
    let input = reference_input(5);
    let ndarray_input = Array1::from_vec(input.clone());
    let plan = reference_plan(input.len(), 7);
    let expected = plan.forward(&ndarray_input).expect("ndarray forward");

    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("leto input");
    let actual = plan.forward_leto(leto_input.view()).expect("leto forward");

    assert_eq!(actual.shape(), [plan.output_len()]);
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_direct_forward_matches_ndarray_reference() {
    let input = reference_input(5);
    let ndarray_input = Array1::from_vec(input.clone());
    let plan = reference_plan(input.len(), 7);
    let expected = plan
        .forward_direct(&ndarray_input)
        .expect("ndarray direct forward");

    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("leto input");
    let actual = plan
        .forward_direct_leto(leto_input.view())
        .expect("leto direct forward");

    assert_eq!(actual.shape(), [plan.output_len()]);
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_strided_forward_matches_ndarray_reference() {
    let input = reference_input(5);
    let ndarray_input = Array1::from_vec(input.clone());
    let plan = reference_plan(input.len(), 7);
    let expected = plan.forward(&ndarray_input).expect("ndarray forward");

    let mut interleaved = Vec::with_capacity(input.len() * 2);
    for value in input {
        interleaved.push(value);
        interleaved.push(Complex64::new(-999.0, 999.0));
    }
    let leto_input =
        leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto input");
    let strided = leto_input
        .view()
        .slice_with::<1>(&[SliceArg::range(
            Some(0),
            Some(leto_input.shape()[0] as isize),
            2,
        )])
        .expect("strided Leto input");
    let actual = plan.forward_leto(strided).expect("leto forward");

    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_strided_direct_forward_matches_ndarray_reference() {
    let input = reference_input(5);
    let ndarray_input = Array1::from_vec(input.clone());
    let plan = reference_plan(input.len(), 7);
    let expected = plan
        .forward_direct(&ndarray_input)
        .expect("ndarray direct forward");

    let mut interleaved = Vec::with_capacity(input.len() * 2);
    for value in input {
        interleaved.push(value);
        interleaved.push(Complex64::new(-999.0, 999.0));
    }
    let leto_input =
        leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto input");
    let strided = leto_input
        .view()
        .slice_with::<1>(&[SliceArg::range(
            Some(0),
            Some(leto_input.shape()[0] as isize),
            2,
        )])
        .expect("strided Leto input");
    let actual = plan
        .forward_direct_leto(strided)
        .expect("leto direct forward");

    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_typed_complex32_forward_matches_ndarray_reference() {
    let input64 = reference_input(5);
    let input32: Vec<Complex32> = input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect();
    let ndarray_input = Array1::from_vec(input32.clone());
    let plan = reference_plan(input32.len(), 7);
    let mut expected = Array1::<Complex32>::zeros(plan.output_len());
    plan.forward_typed_into(
        &ndarray_input,
        &mut expected,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("ndarray typed forward");

    let leto_input =
        leto::Array1::from_shape_vec([input32.len()], input32).expect("leto typed input");
    let actual = plan
        .forward_leto_typed(leto_input.view(), PrecisionProfile::LOW_PRECISION_F32)
        .expect("leto typed forward");

    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-6);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-6);
    }
}

#[test]
fn leto_inverse_matches_ndarray_reference() {
    let n = 5usize;
    let input = reference_input(n);
    let ndarray_input = Array1::from_vec(input.clone());
    let plan = CztPlan::new(
        n,
        n,
        Complex64::new(1.0, 0.0),
        Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64),
    )
    .expect("DFT-equivalent plan");
    let spectrum = plan.forward(&ndarray_input).expect("ndarray forward");
    let expected = plan.inverse(&spectrum).expect("ndarray inverse");

    let leto_spectrum = leto::Array1::from_shape_vec([n], spectrum.iter().copied().collect())
        .expect("leto spectrum");
    let actual = plan
        .inverse_leto(leto_spectrum.view())
        .expect("leto inverse");

    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-10);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-10);
    }
}
