use crate::{
    fft_3d_array, fft_3d_array_into, fft_3d_array_typed, fft_3d_array_typed_into,
    ifft_3d_array_typed_into, Complex32, Complex64,
};
use half::f16;
use ndarray::Array3;

#[test]
fn fft_3d_array_into_matches_allocating_path() {
    let (nx, ny, nz) = (8, 8, 8);
    let field = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| ((i + j + k) as f64 * 0.3).sin());
    let expected = fft_3d_array(&field);
    let mut actual = Array3::<Complex64>::zeros((nx, ny, nz));
    fft_3d_array_into(&field, &mut actual);
    for (lhs, rhs) in expected.iter().zip(actual.iter()) {
        assert!((lhs - rhs).norm() < 1e-13);
    }
}

#[test]
fn typed_3d_into_supports_f64_f32_and_f16_profiles() {
    let (nx, ny, nz) = (4, 4, 4);
    let field64 = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        ((i as f64 * 0.17) + (j as f64 * 0.31) - (k as f64 * 0.11)).sin()
    });

    let expected64 = fft_3d_array_typed(&field64);
    let mut spectrum64 = Array3::<Complex64>::zeros((nx, ny, nz));
    fft_3d_array_typed_into(&field64, &mut spectrum64);
    for (expected, actual) in expected64.iter().zip(spectrum64.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let mut recovered64 = Array3::<f64>::zeros((nx, ny, nz));
    let mut scratch64 = Array3::<Complex64>::zeros((nx, ny, nz));
    ifft_3d_array_typed_into(&spectrum64, &mut recovered64, &mut scratch64);
    for (expected, actual) in field64.iter().zip(recovered64.iter()) {
        assert!((expected - actual).abs() < 1e-12);
    }

    let field32 = field64.mapv(|value| value as f32);
    let expected32 = fft_3d_array_typed(&field32);
    let mut spectrum32 = Array3::<Complex32>::zeros((nx, ny, nz));
    fft_3d_array_typed_into(&field32, &mut spectrum32);
    for (expected, actual) in expected32.iter().zip(spectrum32.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let mut recovered32 = Array3::<f32>::zeros((nx, ny, nz));
    let mut scratch32 = Array3::<Complex32>::zeros((nx, ny, nz));
    ifft_3d_array_typed_into(&spectrum32, &mut recovered32, &mut scratch32);
    for (expected, actual) in field32.iter().zip(recovered32.iter()) {
        assert!((expected - actual).abs() < 1e-5);
    }

    let field16 = field64.mapv(|value| f16::from_f32(value as f32));
    let expected16 = fft_3d_array_typed(&field16);
    let mut spectrum16 = Array3::<Complex32>::zeros((nx, ny, nz));
    fft_3d_array_typed_into(&field16, &mut spectrum16);
    for (expected, actual) in expected16.iter().zip(spectrum16.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let mut recovered16 = Array3::<f16>::from_elem((nx, ny, nz), f16::from_f32(0.0));
    let mut scratch16 = Array3::<Complex32>::zeros((nx, ny, nz));
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
