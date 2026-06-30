//! Tests for forward real FFT API.

use crate::*;
use half::f16;
use leto::{Array1, Array2, Array3};
use eunomia::Complex64;

#[test]
fn fft_3d_array_into_matches_allocating_path() {
    let (nx, ny, nz) = (8, 8, 8);
    let field = Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| ((i + j + k) as f64 * 0.3).sin());
    let expected = fft_3d_array(&field);
    let mut actual = Array3::<Complex64>::zeros([nx, ny, nz]);
    fft_3d_array_into(&field, &mut actual);
    for (lhs, rhs) in expected.iter().zip(actual.iter()) {
        assert!((lhs - rhs).norm() < 1e-13);
    }

    let preserved_spectrum = expected.clone();
    let expected_recovered = ifft_3d_array(&expected);
    let mut actual_recovered = Array3::<f64>::zeros([nx, ny, nz]);
    let mut scratch = Array3::<Complex64>::zeros([nx, ny, nz]);
    ifft_3d_array_into_scratch(&expected, &mut actual_recovered, &mut scratch);
    for (lhs, rhs) in expected_recovered.iter().zip(actual_recovered.iter()) {
        assert!((lhs - rhs).abs() < 1e-12);
    }
    assert_eq!(expected, preserved_spectrum);
}

#[test]
fn real_1d_2d_into_wrappers_match_allocating_paths() {
    let signal1 = Array1::from_shape_fn(18, |i| (i as f64 * 0.17).sin());
    let expected1 = fft_1d_array(&signal1);
    let mut actual1 = Array1::<Complex64>::zeros(18);
    fft_1d_array_into(&signal1, &mut actual1);
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered1 = ifft_1d_array(&expected1);
    let mut actual_recovered1 = Array1::<f64>::zeros(18);
    let mut scratch1 = Array1::<Complex64>::zeros(18);
    ifft_1d_array_into(&expected1, &mut actual_recovered1, &mut scratch1);
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).abs() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn([6, 10], |[i, j]| {
        ((i as f64 * 0.13) - (j as f64 * 0.23)).cos()
    });
    let expected2 = fft_2d_array(&signal2);
    let mut actual2 = Array2::<Complex64>::zeros([6, 10]);
    fft_2d_array_into(&signal2, &mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered2 = ifft_2d_array(&expected2);
    let mut actual_recovered2 = Array2::<f64>::zeros([6, 10]);
    let mut scratch2 = Array2::<Complex64>::zeros([6, 10]);
    ifft_2d_array_into(&expected2, &mut actual_recovered2, &mut scratch2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).abs() < 1e-13);
    }
}

#[test]
fn static_real_1d_into_wrappers_match_dynamic_paths() {
    let signal64 = Array1::from_shape_fn(20, |i| (i as f64 * 0.11).sin());
    let expected64 = fft_1d_array(&signal64);
    let mut spectrum64 = Array1::<Complex64>::zeros(20);
    fft_1d_array_static_into::<20>(&signal64, &mut spectrum64);
    for (expected, actual) in expected64.iter().zip(spectrum64.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered64 = ifft_1d_array(&expected64);
    let mut actual_recovered64 = Array1::<f64>::zeros(20);
    let mut scratch64 = Array1::<Complex64>::zeros(20);
    ifft_1d_array_static_into::<20>(&expected64, &mut actual_recovered64, &mut scratch64);
    for (expected, actual) in recovered64.iter().zip(actual_recovered64.iter()) {
        assert!((expected - actual).abs() < 1e-13);
    }

    let signal32 = Array1::from_shape_fn(20, |i| (i as f32 * 0.13).cos());
    let expected32 = fft_1d_array_typed(&signal32);
    let mut spectrum32 = Array1::<Complex32>::zeros(20);
    fft_1d_array_static_typed_into::<f32, 20>(&signal32, &mut spectrum32);
    for (expected, actual) in expected32.iter().zip(spectrum32.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let recovered32 = ifft_1d_array_typed::<f32>(&expected32);
    let mut actual_recovered32 = Array1::<f32>::zeros(20);
    let mut scratch32 = Array1::<Complex32>::zeros(20);
    ifft_1d_array_static_typed_into::<f32, 20>(
        &expected32,
        &mut actual_recovered32,
        &mut scratch32,
    );
    for (expected, actual) in recovered32.iter().zip(actual_recovered32.iter()) {
        assert!((expected - actual).abs() < 1e-5);
    }

    let signal16 = signal64.mapv(|value| f16::from_f32(value as f32));
    let expected16 = fft_1d_array_typed(&signal16);
    let mut spectrum16 = Array1::<Complex32>::zeros(20);
    fft_1d_array_static_typed_into::<f16, 20>(&signal16, &mut spectrum16);
    for (expected, actual) in expected16.iter().zip(spectrum16.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let recovered16 = ifft_1d_array_typed::<f16>(&expected16);
    let mut actual_recovered16 = Array1::<f16>::from_elem(20, f16::from_f32(0.0));
    let mut scratch16 = Array1::<Complex32>::zeros(20);
    ifft_1d_array_static_typed_into::<f16, 20>(
        &expected16,
        &mut actual_recovered16,
        &mut scratch16,
    );
    for (expected, actual) in recovered16.iter().zip(actual_recovered16.iter()) {
        assert!((expected.to_f32() - actual.to_f32()).abs() < 1e-3);
    }
}

#[test]
fn static_real_2d_into_wrappers_match_dynamic_paths() {
    let field64 = Array2::from_shape_fn([4, 5], |[i, j]| {
        ((i as f64 * 0.11) - (j as f64 * 0.07)).sin()
    });
    let expected64 = fft_2d_array(&field64);
    let mut spectrum64 = Array2::<Complex64>::zeros([4, 5]);
    fft_2d_array_static_into::<4, 5>(&field64, &mut spectrum64);
    for (expected, actual) in expected64.iter().zip(spectrum64.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered64 = ifft_2d_array(&expected64);
    let mut actual_recovered64 = Array2::<f64>::zeros([4, 5]);
    let mut scratch64 = Array2::<Complex64>::zeros([4, 5]);
    ifft_2d_array_static_into::<4, 5>(&expected64, &mut actual_recovered64, &mut scratch64);
    for (expected, actual) in recovered64.iter().zip(actual_recovered64.iter()) {
        assert!((expected - actual).abs() < 1e-13);
    }

    let field32 = Array2::from_shape_fn([4, 5], |[i, j]| {
        ((i as f32 * 0.13) + (j as f32 * 0.17)).cos()
    });
    let expected32 = fft_2d_array_typed(&field32);
    let mut spectrum32 = Array2::<Complex32>::zeros([4, 5]);
    fft_2d_array_static_typed_into::<f32, 4, 5>(&field32, &mut spectrum32);
    for (expected, actual) in expected32.iter().zip(spectrum32.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let recovered32 = ifft_2d_array_typed::<f32>(&expected32);
    let mut actual_recovered32 = Array2::<f32>::zeros([4, 5]);
    let mut scratch32 = Array2::<Complex32>::zeros([4, 5]);
    ifft_2d_array_static_typed_into::<f32, 4, 5>(
        &expected32,
        &mut actual_recovered32,
        &mut scratch32,
    );
    for (expected, actual) in recovered32.iter().zip(actual_recovered32.iter()) {
        assert!((expected - actual).abs() < 1e-5);
    }

    let field16 = field64.mapv(|value| f16::from_f32(value as f32));
    let expected16 = fft_2d_array_typed(&field16);
    let mut spectrum16 = Array2::<Complex32>::zeros([4, 5]);
    fft_2d_array_static_typed_into::<f16, 4, 5>(&field16, &mut spectrum16);
    for (expected, actual) in expected16.iter().zip(spectrum16.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let recovered16 = ifft_2d_array_typed::<f16>(&expected16);
    let mut actual_recovered16 = Array2::<f16>::from_elem([4, 5], f16::from_f32(0.0));
    let mut scratch16 = Array2::<Complex32>::zeros([4, 5]);
    ifft_2d_array_static_typed_into::<f16, 4, 5>(
        &expected16,
        &mut actual_recovered16,
        &mut scratch16,
    );
    for (expected, actual) in recovered16.iter().zip(actual_recovered16.iter()) {
        assert!((expected.to_f32() - actual.to_f32()).abs() < 1e-3);
    }
}

#[test]
fn static_real_3d_into_wrappers_match_dynamic_paths() {
    let field64 = Array3::from_shape_fn([3, 4, 5], |[i, j, k]| {
        ((i as f64 * 0.11) + (j as f64 * 0.07) - (k as f64 * 0.05)).sin()
    });
    let expected64 = fft_3d_array(&field64);
    let mut spectrum64 = Array3::<Complex64>::zeros([3, 4, 5]);
    fft_3d_array_static_into::<3, 4, 5>(&field64, &mut spectrum64);
    for (expected, actual) in expected64.iter().zip(spectrum64.iter()) {
        assert!((expected - actual).norm() < 1e-12);
    }
    let recovered64 = ifft_3d_array(&expected64);
    let mut actual_recovered64 = Array3::<f64>::zeros([3, 4, 5]);
    let mut scratch64 = Array3::<Complex64>::zeros([3, 4, 5]);
    ifft_3d_array_static_into::<3, 4, 5>(&expected64, &mut actual_recovered64, &mut scratch64);
    for (expected, actual) in recovered64.iter().zip(actual_recovered64.iter()) {
        assert!((expected - actual).abs() < 1e-12);
    }

    let field32 = field64.mapv(|value| value as f32);
    let expected32 = fft_3d_array_typed(&field32);
    let mut spectrum32 = Array3::<Complex32>::zeros([3, 4, 5]);
    fft_3d_array_static_typed_into::<f32, 3, 4, 5>(&field32, &mut spectrum32);
    for (expected, actual) in expected32.iter().zip(spectrum32.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let recovered32 = ifft_3d_array_typed::<f32>(&expected32);
    let mut actual_recovered32 = Array3::<f32>::zeros([3, 4, 5]);
    let mut scratch32 = Array3::<Complex32>::zeros([3, 4, 5]);
    ifft_3d_array_static_typed_into::<f32, 3, 4, 5>(
        &expected32,
        &mut actual_recovered32,
        &mut scratch32,
    );
    for (expected, actual) in recovered32.iter().zip(actual_recovered32.iter()) {
        assert!((expected - actual).abs() < 2e-5);
    }

    let field16 = field64.mapv(|value| f16::from_f32(value as f32));
    let expected16 = fft_3d_array_typed(&field16);
    let mut spectrum16 = Array3::<Complex32>::zeros([3, 4, 5]);
    fft_3d_array_static_typed_into::<f16, 3, 4, 5>(&field16, &mut spectrum16);
    for (expected, actual) in expected16.iter().zip(spectrum16.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let recovered16 = ifft_3d_array_typed::<f16>(&expected16);
    let mut actual_recovered16 = Array3::<f16>::from_elem([3, 4, 5], f16::from_f32(0.0));
    let mut scratch16 = Array3::<Complex32>::zeros([3, 4, 5]);
    ifft_3d_array_static_typed_into::<f16, 3, 4, 5>(
        &expected16,
        &mut actual_recovered16,
        &mut scratch16,
    );
    for (expected, actual) in recovered16.iter().zip(actual_recovered16.iter()) {
        assert!((expected.to_f32() - actual.to_f32()).abs() < 1e-3);
    }
}

#[test]
fn typed_real_1d_2d_into_supports_f64_f32_and_f16_profiles() {
    let signal64 = Array1::from_shape_fn(20, |i| (i as f64 * 0.11).sin());
    let expected64 = fft_1d_array_typed(&signal64);
    let mut spectrum64 = Array1::<Complex64>::zeros(20);
    fft_1d_array_typed_into(&signal64, &mut spectrum64);
    for (expected, actual) in expected64.iter().zip(spectrum64.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered64 = ifft_1d_array_typed::<f64>(&spectrum64);
    let mut actual_recovered64 = Array1::<f64>::zeros(20);
    let mut scratch64 = Array1::<Complex64>::zeros(20);
    ifft_1d_array_typed_into(&spectrum64, &mut actual_recovered64, &mut scratch64);
    for (expected, actual) in recovered64.iter().zip(actual_recovered64.iter()) {
        assert!((expected - actual).abs() < 1e-13);
    }

    let field32 = Array2::from_shape_fn([4, 6], |[i, j]| {
        ((i as f32 * 0.19) + (j as f32 * 0.07)).sin()
    });
    let expected32 = fft_2d_array_typed(&field32);
    let mut spectrum32 = Array2::<Complex32>::zeros([4, 6]);
    fft_2d_array_typed_into(&field32, &mut spectrum32);
    for (expected, actual) in expected32.iter().zip(spectrum32.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let recovered32 = ifft_2d_array_typed::<f32>(&spectrum32);
    let mut actual_recovered32 = Array2::<f32>::zeros([4, 6]);
    let mut scratch32 = Array2::<Complex32>::zeros([4, 6]);
    ifft_2d_array_typed_into(&spectrum32, &mut actual_recovered32, &mut scratch32);
    for (expected, actual) in recovered32.iter().zip(actual_recovered32.iter()) {
        assert!((expected - actual).abs() < 1e-5);
    }

    let signal16 = signal64.mapv(|value| f16::from_f32(value as f32));
    let expected16 = fft_1d_array_typed(&signal16);
    let mut spectrum16 = Array1::<Complex32>::zeros(20);
    fft_1d_array_typed_into(&signal16, &mut spectrum16);
    for (expected, actual) in expected16.iter().zip(spectrum16.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let recovered16 = ifft_1d_array_typed::<f16>(&spectrum16);
    let mut actual_recovered16 = Array1::<f16>::from_elem(20, f16::from_f32(0.0));
    let mut scratch16 = Array1::<Complex32>::zeros(20);
    ifft_1d_array_typed_into(&spectrum16, &mut actual_recovered16, &mut scratch16);
    for (expected, actual) in recovered16.iter().zip(actual_recovered16.iter()) {
        assert!((expected.to_f32() - actual.to_f32()).abs() < 1e-3);
    }
}

#[test]
fn typed_3d_into_supports_f64_f32_and_f16_profiles() {
    let (nx, ny, nz) = (4, 4, 4);
    let field64 = Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| {
        ((i as f64 * 0.17) + (j as f64 * 0.31) - (k as f64 * 0.11)).sin()
    });

    let expected64 = fft_3d_array_typed(&field64);
    let mut spectrum64 = Array3::<Complex64>::zeros([nx, ny, nz]);
    fft_3d_array_typed_into(&field64, &mut spectrum64);
    for (expected, actual) in expected64.iter().zip(spectrum64.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let mut recovered64 = Array3::<f64>::zeros([nx, ny, nz]);
    let mut scratch64 = Array3::<Complex64>::zeros([nx, ny, nz]);
    ifft_3d_array_typed_into(&spectrum64, &mut recovered64, &mut scratch64);
    for (expected, actual) in field64.iter().zip(recovered64.iter()) {
        assert!((expected - actual).abs() < 1e-12);
    }

    let field32 = field64.mapv(|value| value as f32);
    let expected32 = fft_3d_array_typed(&field32);
    let mut spectrum32 = Array3::<Complex32>::zeros([nx, ny, nz]);
    fft_3d_array_typed_into(&field32, &mut spectrum32);
    for (expected, actual) in expected32.iter().zip(spectrum32.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let mut recovered32 = Array3::<f32>::zeros([nx, ny, nz]);
    let mut scratch32 = Array3::<Complex32>::zeros([nx, ny, nz]);
    ifft_3d_array_typed_into(&spectrum32, &mut recovered32, &mut scratch32);
    for (expected, actual) in field32.iter().zip(recovered32.iter()) {
        assert!((expected - actual).abs() < 1e-5);
    }

    let field16 = field64.mapv(|value| f16::from_f32(value as f32));
    let expected16 = fft_3d_array_typed(&field16);
    let mut spectrum16 = Array3::<Complex32>::zeros([nx, ny, nz]);
    fft_3d_array_typed_into(&field16, &mut spectrum16);
    for (expected, actual) in expected16.iter().zip(spectrum16.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let mut recovered16 = Array3::<f16>::from_elem([nx, ny, nz], f16::from_f32(0.0));
    let mut scratch16 = Array3::<Complex32>::zeros([nx, ny, nz]);
    ifft_3d_array_typed_into(&spectrum16, &mut recovered16, &mut scratch16);
    for (expected, actual) in field16.iter().zip(recovered16.iter()) {
        let stage_count = 6.0_f32;
        let unit_roundoff = 2.0_f32.powi(-11);
        let bound = 2.0 * stage_count * unit_roundoff;
        assert!(
            (expected.to_f32() - actual.to_f32()).abs() < bound,
            "f16 round-trip error: got {}, expected {}",
            actual.to_f32(),
            expected.to_f32()
        );
    }
}
