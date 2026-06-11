//! Tests for inverse real FFT API.

use crate::*;
use half::f16;
use ndarray::{Array1, Array2, Array3};

#[test]
fn real_inverse_mutating_spectrum_wrappers_reuse_spectrum_storage() {
    let signal1 = Array1::from_shape_fn(18, |i| (i as f64 * 0.17).sin());
    let expected_spectrum1 = fft_1d_array(&signal1);
    let expected_recovered1 = ifft_1d_array(&expected_spectrum1);
    let mut spectrum1 = expected_spectrum1.clone();
    let spectrum_ptr1 = spectrum1.as_ptr();
    let mut actual_recovered1 = Array1::<f64>::zeros(18);
    ifft_1d_array_into_spectrum_scratch(&mut spectrum1, &mut actual_recovered1);
    assert_eq!(spectrum_ptr1, spectrum1.as_ptr());
    for (expected, actual) in expected_recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).abs() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn((6, 10), |(i, j)| {
        ((i as f64 * 0.13) - (j as f64 * 0.23)).cos()
    });
    let expected_spectrum2 = fft_2d_array(&signal2);
    let expected_recovered2 = ifft_2d_array(&expected_spectrum2);
    let mut spectrum2 = expected_spectrum2.clone();
    let spectrum_ptr2 = spectrum2.as_ptr();
    let mut actual_recovered2 = Array2::<f64>::zeros((6, 10));
    ifft_2d_array_into_spectrum_scratch(&mut spectrum2, &mut actual_recovered2);
    assert_eq!(spectrum_ptr2, spectrum2.as_ptr());
    for (expected, actual) in expected_recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).abs() < 1e-13);
    }
}

#[test]
fn typed_real_inverse_mutating_spectrum_wrappers_reuse_spectrum_storage() {
    let signal32 = Array1::from_shape_fn(20, |i| (i as f32 * 0.17).sin());
    let expected_spectrum32 = fft_1d_array_typed(&signal32);
    let expected_recovered32 = ifft_1d_array_typed::<f32>(&expected_spectrum32);
    let mut spectrum32 = expected_spectrum32.clone();
    let spectrum_ptr32 = spectrum32.as_ptr();
    let mut actual_recovered32 = Array1::<f32>::zeros(20);
    ifft_1d_array_typed_into_spectrum_scratch::<f32>(&mut spectrum32, &mut actual_recovered32);
    assert_eq!(spectrum_ptr32, spectrum32.as_ptr());
    for (expected, actual) in expected_recovered32.iter().zip(actual_recovered32.iter()) {
        assert!((expected - actual).abs() < 1e-5);
    }

    let field16 = Array2::from_shape_fn((4, 5), |(i, j)| {
        f16::from_f32(((i as f32 * 0.13) - (j as f32 * 0.19)).cos())
    });
    let expected_spectrum16 = fft_2d_array_typed(&field16);
    let expected_recovered16 = ifft_2d_array_typed::<f16>(&expected_spectrum16);
    let mut spectrum16 = expected_spectrum16.clone();
    let spectrum_ptr16 = spectrum16.as_ptr();
    let mut actual_recovered16 = Array2::<f16>::from_elem((4, 5), f16::from_f32(0.0));
    ifft_2d_array_typed_into_spectrum_scratch::<f16>(&mut spectrum16, &mut actual_recovered16);
    assert_eq!(spectrum_ptr16, spectrum16.as_ptr());
    for (expected, actual) in expected_recovered16.iter().zip(actual_recovered16.iter()) {
        assert!((expected.to_f32() - actual.to_f32()).abs() < 1e-3);
    }

    let field3 = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| {
        ((i as f32 * 0.11) + (j as f32 * 0.07) - (k as f32 * 0.05)).sin()
    });
    let expected_spectrum3 = fft_3d_array_typed(&field3);
    let expected_recovered3 = ifft_3d_array_typed::<f32>(&expected_spectrum3);
    let mut spectrum3 = expected_spectrum3.clone();
    let spectrum_ptr3 = spectrum3.as_ptr();
    let mut actual_recovered3 = Array3::<f32>::zeros((3, 4, 5));
    ifft_3d_array_typed_into_spectrum_scratch::<f32>(&mut spectrum3, &mut actual_recovered3);
    assert_eq!(spectrum_ptr3, spectrum3.as_ptr());
    for (expected, actual) in expected_recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).abs() < 2e-5);
    }
}
