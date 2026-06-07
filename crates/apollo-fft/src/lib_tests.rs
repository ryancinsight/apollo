use crate::{
    fft_1d_array, fft_1d_array_into, fft_1d_array_static_into, fft_1d_array_static_typed_into,
    fft_1d_array_typed, fft_1d_array_typed_into, fft_1d_complex, fft_1d_complex_into,
    fft_1d_complex_owned, fft_1d_complex_static, fft_1d_complex_static_inplace,
    fft_1d_complex_static_into, fft_1d_complex_static_typed, fft_1d_complex_static_typed_inplace,
    fft_1d_complex_static_typed_into, fft_1d_complex_typed, fft_1d_complex_typed_inplace,
    fft_1d_complex_typed_into, fft_1d_complex_typed_owned, fft_2d_array, fft_2d_array_into,
    fft_2d_array_static_into, fft_2d_array_static_typed_into, fft_2d_array_typed,
    fft_2d_array_typed_into, fft_2d_complex, fft_2d_complex_into, fft_2d_complex_owned,
    fft_2d_complex_static, fft_2d_complex_static_inplace, fft_2d_complex_static_into,
    fft_2d_complex_static_typed, fft_2d_complex_static_typed_inplace,
    fft_2d_complex_static_typed_into, fft_2d_complex_typed, fft_2d_complex_typed_inplace,
    fft_2d_complex_typed_into, fft_2d_complex_typed_owned, fft_3d_array, fft_3d_array_into,
    fft_3d_array_static_into, fft_3d_array_static_typed_into, fft_3d_array_typed,
    fft_3d_array_typed_into, fft_3d_complex, fft_3d_complex_into, fft_3d_complex_owned,
    fft_3d_complex_static, fft_3d_complex_static_inplace, fft_3d_complex_static_into,
    fft_3d_complex_static_typed, fft_3d_complex_static_typed_inplace,
    fft_3d_complex_static_typed_into, fft_3d_complex_typed, fft_3d_complex_typed_inplace,
    fft_3d_complex_typed_into, fft_3d_complex_typed_owned, ifft_1d_array, ifft_1d_array_into,
    ifft_1d_array_into_spectrum_scratch, ifft_1d_array_static_into,
    ifft_1d_array_static_typed_into, ifft_1d_array_typed, ifft_1d_array_typed_into,
    ifft_1d_array_typed_into_spectrum_scratch, ifft_1d_complex, ifft_1d_complex_into,
    ifft_1d_complex_owned, ifft_1d_complex_static, ifft_1d_complex_static_inplace,
    ifft_1d_complex_static_into, ifft_1d_complex_static_typed,
    ifft_1d_complex_static_typed_inplace, ifft_1d_complex_static_typed_into, ifft_1d_complex_typed,
    ifft_1d_complex_typed_inplace, ifft_1d_complex_typed_into, ifft_1d_complex_typed_owned,
    ifft_2d_array, ifft_2d_array_into, ifft_2d_array_into_spectrum_scratch,
    ifft_2d_array_static_into, ifft_2d_array_static_typed_into, ifft_2d_array_typed,
    ifft_2d_array_typed_into, ifft_2d_array_typed_into_spectrum_scratch, ifft_2d_complex,
    ifft_2d_complex_into, ifft_2d_complex_owned, ifft_2d_complex_static,
    ifft_2d_complex_static_inplace, ifft_2d_complex_static_into, ifft_2d_complex_static_typed,
    ifft_2d_complex_static_typed_inplace, ifft_2d_complex_static_typed_into, ifft_2d_complex_typed,
    ifft_2d_complex_typed_inplace, ifft_2d_complex_typed_into, ifft_2d_complex_typed_owned,
    ifft_3d_array, ifft_3d_array_into_scratch, ifft_3d_array_static_into,
    ifft_3d_array_static_typed_into, ifft_3d_array_typed, ifft_3d_array_typed_into,
    ifft_3d_array_typed_into_spectrum_scratch, ifft_3d_complex, ifft_3d_complex_into,
    ifft_3d_complex_owned, ifft_3d_complex_static, ifft_3d_complex_static_inplace,
    ifft_3d_complex_static_into, ifft_3d_complex_static_typed,
    ifft_3d_complex_static_typed_inplace, ifft_3d_complex_static_typed_into, ifft_3d_complex_typed,
    ifft_3d_complex_typed_inplace, ifft_3d_complex_typed_into, ifft_3d_complex_typed_owned,
    Complex32, Complex64, FftPlan1D, FftPlan2D, FftPlan3D, Shape1D, Shape2D, Shape3D,
};
use half::f16;
use ndarray::{Array1, Array2, Array3};

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

    let preserved_spectrum = expected.clone();
    let expected_recovered = ifft_3d_array(&expected);
    let mut actual_recovered = Array3::<f64>::zeros((nx, ny, nz));
    let mut scratch = Array3::<Complex64>::zeros((nx, ny, nz));
    ifft_3d_array_into_scratch(&expected, &mut actual_recovered, &mut scratch);
    for (lhs, rhs) in expected_recovered.iter().zip(actual_recovered.iter()) {
        assert!((lhs - rhs).abs() < 1e-12);
    }
    assert_eq!(expected, preserved_spectrum);
}

#[test]
fn complex_into_wrappers_match_allocating_paths() {
    let signal1 = Array1::from_shape_fn(16, |i| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let mut actual1 = Array1::<Complex64>::zeros(16);
    fft_1d_complex_into(&signal1, &mut actual1);
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered1 = ifft_1d_complex(&expected1);
    let mut actual_recovered1 = Array1::<Complex64>::zeros(16);
    ifft_1d_complex_into(&expected1, &mut actual_recovered1);
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn((4, 8), |(i, j)| {
        let x = (i * 8 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let mut actual2 = Array2::<Complex64>::zeros((4, 8));
    fft_2d_complex_into(&signal2, &mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered2 = ifft_2d_complex(&expected2);
    let mut actual_recovered2 = Array2::<Complex64>::zeros((4, 8));
    ifft_2d_complex_into(&expected2, &mut actual_recovered2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| {
        let x = (i * 16 + j * 4 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let mut actual3 = Array3::<Complex64>::zeros((4, 4, 4));
    fft_3d_complex_into(&signal3, &mut actual3);
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered3 = ifft_3d_complex(&expected3);
    let mut actual_recovered3 = Array3::<Complex64>::zeros((4, 4, 4));
    ifft_3d_complex_into(&expected3, &mut actual_recovered3);
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
}

#[test]
fn owned_complex_wrappers_reuse_input_allocation() {
    let signal1 = Array1::from_shape_fn(16, |i| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let ptr1 = signal1.as_ptr();
    let actual1 = fft_1d_complex_owned(signal1);
    assert_eq!(ptr1, actual1.as_ptr());
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered1 = ifft_1d_complex(&expected1);
    let ptr_recovered1 = expected1.as_ptr();
    let actual_recovered1 = ifft_1d_complex_owned(expected1);
    assert_eq!(ptr_recovered1, actual_recovered1.as_ptr());
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn((4, 8), |(i, j)| {
        let x = (i * 8 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let ptr2 = signal2.as_ptr();
    let actual2 = fft_2d_complex_owned(signal2);
    assert_eq!(ptr2, actual2.as_ptr());
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered2 = ifft_2d_complex(&expected2);
    let ptr_recovered2 = expected2.as_ptr();
    let actual_recovered2 = ifft_2d_complex_owned(expected2);
    assert_eq!(ptr_recovered2, actual_recovered2.as_ptr());
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| {
        let x = (i * 16 + j * 4 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let ptr3 = signal3.as_ptr();
    let actual3 = fft_3d_complex_owned(signal3);
    assert_eq!(ptr3, actual3.as_ptr());
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered3 = ifft_3d_complex(&expected3);
    let ptr_recovered3 = expected3.as_ptr();
    let actual_recovered3 = ifft_3d_complex_owned(expected3);
    assert_eq!(ptr_recovered3, actual_recovered3.as_ptr());
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
}

#[test]
fn static_complex_wrappers_match_dynamic_paths() {
    let signal1 = Array1::from_shape_fn(16, |i| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let mut actual1 = signal1.clone();
    fft_1d_complex_static_inplace::<16>(&mut actual1);
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    ifft_1d_complex_static_inplace::<16>(&mut actual1);
    for (expected, actual) in signal1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn((4, 5), |(i, j)| {
        let x = (i * 5 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let mut actual2 = signal2.clone();
    fft_2d_complex_static_inplace::<4, 5>(&mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    ifft_2d_complex_static_inplace::<4, 5>(&mut actual2);
    for (expected, actual) in signal2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| {
        let x = ((i * 4 + j) * 5 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let mut actual3 = signal3.clone();
    fft_3d_complex_static_inplace::<3, 4, 5>(&mut actual3);
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    ifft_3d_complex_static_inplace::<3, 4, 5>(&mut actual3);
    for (expected, actual) in signal3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
}

#[test]
fn static_complex_into_wrappers_match_allocating_paths() {
    let signal1 = Array1::from_shape_fn(16, |i| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let mut actual1 = Array1::<Complex64>::zeros(16);
    fft_1d_complex_static_into::<16>(&signal1, &mut actual1);
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered1 = ifft_1d_complex(&expected1);
    let mut actual_recovered1 = Array1::<Complex64>::zeros(16);
    ifft_1d_complex_static_into::<16>(&expected1, &mut actual_recovered1);
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn((4, 5), |(i, j)| {
        let x = (i * 5 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let mut actual2 = Array2::<Complex64>::zeros((4, 5));
    fft_2d_complex_static_into::<4, 5>(&signal2, &mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered2 = ifft_2d_complex(&expected2);
    let mut actual_recovered2 = Array2::<Complex64>::zeros((4, 5));
    ifft_2d_complex_static_into::<4, 5>(&expected2, &mut actual_recovered2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| {
        let x = ((i * 4 + j) * 5 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let mut actual3 = Array3::<Complex64>::zeros((3, 4, 5));
    fft_3d_complex_static_into::<3, 4, 5>(&signal3, &mut actual3);
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered3 = ifft_3d_complex(&expected3);
    let mut actual_recovered3 = Array3::<Complex64>::zeros((3, 4, 5));
    ifft_3d_complex_static_into::<3, 4, 5>(&expected3, &mut actual_recovered3);
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
}

#[test]
fn owned_static_complex_wrappers_reuse_input_allocation() {
    let signal1 = Array1::from_shape_fn(16, |i| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let ptr1 = signal1.as_ptr();
    let actual1 = fft_1d_complex_static::<16>(signal1);
    assert_eq!(ptr1, actual1.as_ptr());
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered1 = ifft_1d_complex(&expected1);
    let ptr_recovered1 = expected1.as_ptr();
    let actual_recovered1 = ifft_1d_complex_static::<16>(expected1);
    assert_eq!(ptr_recovered1, actual_recovered1.as_ptr());
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn((4, 5), |(i, j)| {
        let x = (i * 5 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let ptr2 = signal2.as_ptr();
    let actual2 = fft_2d_complex_static::<4, 5>(signal2);
    assert_eq!(ptr2, actual2.as_ptr());
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered2 = ifft_2d_complex(&expected2);
    let ptr_recovered2 = expected2.as_ptr();
    let actual_recovered2 = ifft_2d_complex_static::<4, 5>(expected2);
    assert_eq!(ptr_recovered2, actual_recovered2.as_ptr());
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| {
        let x = ((i * 4 + j) * 5 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let ptr3 = signal3.as_ptr();
    let actual3 = fft_3d_complex_static::<3, 4, 5>(signal3);
    assert_eq!(ptr3, actual3.as_ptr());
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered3 = ifft_3d_complex(&expected3);
    let ptr_recovered3 = expected3.as_ptr();
    let actual_recovered3 = ifft_3d_complex_static::<3, 4, 5>(expected3);
    assert_eq!(ptr_recovered3, actual_recovered3.as_ptr());
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal32 = Array1::from_shape_fn(16, |i| {
        let x = i as f32;
        Complex32::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let plan32 = FftPlan1D::<f32>::new(Shape1D::new(16).expect("valid 1D shape"));
    let mut expected32 = signal32.clone();
    plan32.forward_complex_inplace(&mut expected32);
    let ptr32 = signal32.as_ptr();
    let actual32 = fft_1d_complex_static_typed::<f32, 16>(signal32);
    assert_eq!(ptr32, actual32.as_ptr());
    for (expected, actual) in expected32.iter().zip(actual32.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let mut recovered32 = expected32.clone();
    plan32.inverse_complex_inplace(&mut recovered32);
    let ptr_recovered32 = expected32.as_ptr();
    let actual_recovered32 = ifft_1d_complex_static_typed::<f32, 16>(expected32);
    assert_eq!(ptr_recovered32, actual_recovered32.as_ptr());
    for (expected, actual) in recovered32.iter().zip(actual_recovered32.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }

    let signal32_2d = Array2::from_shape_fn((4, 5), |(i, j)| {
        let x = (i * 5 + j) as f32;
        Complex32::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let plan32_2d = FftPlan2D::<f32>::new(Shape2D::new(4, 5).expect("valid 2D shape"));
    let mut expected32_2d = signal32_2d.clone();
    plan32_2d.forward_complex_inplace(&mut expected32_2d);
    let ptr32_2d = signal32_2d.as_ptr();
    let actual32_2d = fft_2d_complex_static_typed::<f32, 4, 5>(signal32_2d);
    assert_eq!(ptr32_2d, actual32_2d.as_ptr());
    for (expected, actual) in expected32_2d.iter().zip(actual32_2d.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let mut recovered32_2d = expected32_2d.clone();
    plan32_2d.inverse_complex_inplace(&mut recovered32_2d);
    let ptr_recovered32_2d = expected32_2d.as_ptr();
    let actual_recovered32_2d = ifft_2d_complex_static_typed::<f32, 4, 5>(expected32_2d);
    assert_eq!(ptr_recovered32_2d, actual_recovered32_2d.as_ptr());
    for (expected, actual) in recovered32_2d.iter().zip(actual_recovered32_2d.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }

    let signal32_3d = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| {
        let x = ((i * 4 + j) * 5 + k) as f32;
        Complex32::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let plan32_3d = FftPlan3D::<f32>::new(Shape3D::new(3, 4, 5).expect("valid 3D shape"));
    let mut expected32_3d = signal32_3d.clone();
    plan32_3d.forward_complex_inplace(&mut expected32_3d);
    let ptr32_3d = signal32_3d.as_ptr();
    let actual32_3d = fft_3d_complex_static_typed::<f32, 3, 4, 5>(signal32_3d);
    assert_eq!(ptr32_3d, actual32_3d.as_ptr());
    for (expected, actual) in expected32_3d.iter().zip(actual32_3d.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let mut recovered32_3d = expected32_3d.clone();
    plan32_3d.inverse_complex_inplace(&mut recovered32_3d);
    let ptr_recovered32_3d = expected32_3d.as_ptr();
    let actual_recovered32_3d = ifft_3d_complex_static_typed::<f32, 3, 4, 5>(expected32_3d);
    assert_eq!(ptr_recovered32_3d, actual_recovered32_3d.as_ptr());
    for (expected, actual) in recovered32_3d.iter().zip(actual_recovered32_3d.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
}

#[test]
fn typed_static_complex_wrappers_match_dynamic_f32_paths() {
    let signal1 = Array1::from_shape_fn(16, |i| {
        let x = i as f32;
        Complex32::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let plan1 = FftPlan1D::<f32>::new(Shape1D::new(16).expect("valid 1D shape"));
    let mut expected1 = signal1.clone();
    plan1.forward_complex_inplace(&mut expected1);
    let mut actual1 = Array1::<Complex32>::zeros(16);
    fft_1d_complex_static_typed_into::<f32, 16>(&signal1, &mut actual1);
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let mut recovered1 = expected1.clone();
    plan1.inverse_complex_inplace(&mut recovered1);
    let mut actual_recovered1 = Array1::<Complex32>::zeros(16);
    ifft_1d_complex_static_typed_into::<f32, 16>(&expected1, &mut actual_recovered1);
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    fft_1d_complex_static_typed_inplace::<f32, 16>(&mut actual_recovered1);
    ifft_1d_complex_static_typed_inplace::<f32, 16>(&mut actual_recovered1);
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }

    let signal2 = Array2::from_shape_fn((4, 5), |(i, j)| {
        let x = (i * 5 + j) as f32;
        Complex32::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let plan2 = FftPlan2D::<f32>::new(Shape2D::new(4, 5).expect("valid 2D shape"));
    let mut expected2 = signal2.clone();
    plan2.forward_complex_inplace(&mut expected2);
    let mut actual2 = Array2::<Complex32>::zeros((4, 5));
    fft_2d_complex_static_typed_into::<f32, 4, 5>(&signal2, &mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let mut recovered2 = expected2.clone();
    plan2.inverse_complex_inplace(&mut recovered2);
    let mut actual_recovered2 = Array2::<Complex32>::zeros((4, 5));
    ifft_2d_complex_static_typed_into::<f32, 4, 5>(&expected2, &mut actual_recovered2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    fft_2d_complex_static_typed_inplace::<f32, 4, 5>(&mut actual_recovered2);
    ifft_2d_complex_static_typed_inplace::<f32, 4, 5>(&mut actual_recovered2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }

    let signal3 = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| {
        let x = ((i * 4 + j) * 5 + k) as f32;
        Complex32::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let plan3 = FftPlan3D::<f32>::new(Shape3D::new(3, 4, 5).expect("valid 3D shape"));
    let mut expected3 = signal3.clone();
    plan3.forward_complex_inplace(&mut expected3);
    let mut actual3 = Array3::<Complex32>::zeros((3, 4, 5));
    fft_3d_complex_static_typed_into::<f32, 3, 4, 5>(&signal3, &mut actual3);
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let mut recovered3 = expected3.clone();
    plan3.inverse_complex_inplace(&mut recovered3);
    let mut actual_recovered3 = Array3::<Complex32>::zeros((3, 4, 5));
    ifft_3d_complex_static_typed_into::<f32, 3, 4, 5>(&expected3, &mut actual_recovered3);
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    fft_3d_complex_static_typed_inplace::<f32, 3, 4, 5>(&mut actual_recovered3);
    ifft_3d_complex_static_typed_inplace::<f32, 3, 4, 5>(&mut actual_recovered3);
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
}

#[test]
fn typed_dynamic_complex_wrappers_match_f32_plans_and_reuse_owned_allocation() {
    let signal1 = Array1::from_shape_fn(16, |i| {
        let x = i as f32;
        Complex32::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let plan1 = FftPlan1D::<f32>::new(Shape1D::new(16).expect("valid 1D shape"));
    let mut expected1 = signal1.clone();
    plan1.forward_complex_inplace(&mut expected1);

    let actual1 = fft_1d_complex_typed(&signal1);
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let mut actual1_into = Array1::<Complex32>::zeros(16);
    fft_1d_complex_typed_into(&signal1, &mut actual1_into);
    for (expected, actual) in expected1.iter().zip(actual1_into.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let mut actual1_inplace = signal1.clone();
    fft_1d_complex_typed_inplace(&mut actual1_inplace);
    for (expected, actual) in expected1.iter().zip(actual1_inplace.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let owned1 = signal1.clone();
    let ptr1 = owned1.as_ptr();
    let actual1_owned = fft_1d_complex_typed_owned(owned1);
    assert_eq!(ptr1, actual1_owned.as_ptr());
    for (expected, actual) in expected1.iter().zip(actual1_owned.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }

    let mut recovered1 = expected1.clone();
    plan1.inverse_complex_inplace(&mut recovered1);
    let inverse1 = ifft_1d_complex_typed(&expected1);
    for (expected, actual) in recovered1.iter().zip(inverse1.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let mut inverse1_into = Array1::<Complex32>::zeros(16);
    ifft_1d_complex_typed_into(&expected1, &mut inverse1_into);
    for (expected, actual) in recovered1.iter().zip(inverse1_into.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let mut inverse1_inplace = expected1.clone();
    ifft_1d_complex_typed_inplace(&mut inverse1_inplace);
    for (expected, actual) in recovered1.iter().zip(inverse1_inplace.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let owned_inverse1 = expected1.clone();
    let inverse_ptr1 = owned_inverse1.as_ptr();
    let inverse1_owned = ifft_1d_complex_typed_owned(owned_inverse1);
    assert_eq!(inverse_ptr1, inverse1_owned.as_ptr());
    for (expected, actual) in recovered1.iter().zip(inverse1_owned.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }

    let signal2 = Array2::from_shape_fn((4, 5), |(i, j)| {
        let x = (i * 5 + j) as f32;
        Complex32::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let plan2 = FftPlan2D::<f32>::new(Shape2D::new(4, 5).expect("valid 2D shape"));
    let mut expected2 = signal2.clone();
    plan2.forward_complex_inplace(&mut expected2);

    let actual2 = fft_2d_complex_typed(&signal2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let mut actual2_into = Array2::<Complex32>::zeros((4, 5));
    fft_2d_complex_typed_into(&signal2, &mut actual2_into);
    for (expected, actual) in expected2.iter().zip(actual2_into.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let mut actual2_inplace = signal2.clone();
    fft_2d_complex_typed_inplace(&mut actual2_inplace);
    for (expected, actual) in expected2.iter().zip(actual2_inplace.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let owned2 = signal2.clone();
    let ptr2 = owned2.as_ptr();
    let actual2_owned = fft_2d_complex_typed_owned(owned2);
    assert_eq!(ptr2, actual2_owned.as_ptr());
    for (expected, actual) in expected2.iter().zip(actual2_owned.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }

    let mut recovered2 = expected2.clone();
    plan2.inverse_complex_inplace(&mut recovered2);
    let inverse2 = ifft_2d_complex_typed(&expected2);
    for (expected, actual) in recovered2.iter().zip(inverse2.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let mut inverse2_into = Array2::<Complex32>::zeros((4, 5));
    ifft_2d_complex_typed_into(&expected2, &mut inverse2_into);
    for (expected, actual) in recovered2.iter().zip(inverse2_into.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let mut inverse2_inplace = expected2.clone();
    ifft_2d_complex_typed_inplace(&mut inverse2_inplace);
    for (expected, actual) in recovered2.iter().zip(inverse2_inplace.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let owned_inverse2 = expected2.clone();
    let inverse_ptr2 = owned_inverse2.as_ptr();
    let inverse2_owned = ifft_2d_complex_typed_owned(owned_inverse2);
    assert_eq!(inverse_ptr2, inverse2_owned.as_ptr());
    for (expected, actual) in recovered2.iter().zip(inverse2_owned.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }

    let signal3 = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| {
        let x = ((i * 4 + j) * 5 + k) as f32;
        Complex32::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let plan3 = FftPlan3D::<f32>::new(Shape3D::new(3, 4, 5).expect("valid 3D shape"));
    let mut expected3 = signal3.clone();
    plan3.forward_complex_inplace(&mut expected3);

    let actual3 = fft_3d_complex_typed(&signal3);
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let mut actual3_into = Array3::<Complex32>::zeros((3, 4, 5));
    fft_3d_complex_typed_into(&signal3, &mut actual3_into);
    for (expected, actual) in expected3.iter().zip(actual3_into.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let mut actual3_inplace = signal3.clone();
    fft_3d_complex_typed_inplace(&mut actual3_inplace);
    for (expected, actual) in expected3.iter().zip(actual3_inplace.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let owned3 = signal3.clone();
    let ptr3 = owned3.as_ptr();
    let actual3_owned = fft_3d_complex_typed_owned(owned3);
    assert_eq!(ptr3, actual3_owned.as_ptr());
    for (expected, actual) in expected3.iter().zip(actual3_owned.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }

    let mut recovered3 = expected3.clone();
    plan3.inverse_complex_inplace(&mut recovered3);
    let inverse3 = ifft_3d_complex_typed(&expected3);
    for (expected, actual) in recovered3.iter().zip(inverse3.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let mut inverse3_into = Array3::<Complex32>::zeros((3, 4, 5));
    ifft_3d_complex_typed_into(&expected3, &mut inverse3_into);
    for (expected, actual) in recovered3.iter().zip(inverse3_into.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let mut inverse3_inplace = expected3.clone();
    ifft_3d_complex_typed_inplace(&mut inverse3_inplace);
    for (expected, actual) in recovered3.iter().zip(inverse3_inplace.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let owned_inverse3 = expected3.clone();
    let inverse_ptr3 = owned_inverse3.as_ptr();
    let inverse3_owned = ifft_3d_complex_typed_owned(owned_inverse3);
    assert_eq!(inverse_ptr3, inverse3_owned.as_ptr());
    for (expected, actual) in recovered3.iter().zip(inverse3_owned.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
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

    let signal2 = Array2::from_shape_fn((6, 10), |(i, j)| {
        ((i as f64 * 0.13) - (j as f64 * 0.23)).cos()
    });
    let expected2 = fft_2d_array(&signal2);
    let mut actual2 = Array2::<Complex64>::zeros((6, 10));
    fft_2d_array_into(&signal2, &mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered2 = ifft_2d_array(&expected2);
    let mut actual_recovered2 = Array2::<f64>::zeros((6, 10));
    let mut scratch2 = Array2::<Complex64>::zeros((6, 10));
    ifft_2d_array_into(&expected2, &mut actual_recovered2, &mut scratch2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).abs() < 1e-13);
    }
}

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
    let field64 = Array2::from_shape_fn((4, 5), |(i, j)| {
        ((i as f64 * 0.11) - (j as f64 * 0.07)).sin()
    });
    let expected64 = fft_2d_array(&field64);
    let mut spectrum64 = Array2::<Complex64>::zeros((4, 5));
    fft_2d_array_static_into::<4, 5>(&field64, &mut spectrum64);
    for (expected, actual) in expected64.iter().zip(spectrum64.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered64 = ifft_2d_array(&expected64);
    let mut actual_recovered64 = Array2::<f64>::zeros((4, 5));
    let mut scratch64 = Array2::<Complex64>::zeros((4, 5));
    ifft_2d_array_static_into::<4, 5>(&expected64, &mut actual_recovered64, &mut scratch64);
    for (expected, actual) in recovered64.iter().zip(actual_recovered64.iter()) {
        assert!((expected - actual).abs() < 1e-13);
    }

    let field32 = Array2::from_shape_fn((4, 5), |(i, j)| {
        ((i as f32 * 0.13) + (j as f32 * 0.17)).cos()
    });
    let expected32 = fft_2d_array_typed(&field32);
    let mut spectrum32 = Array2::<Complex32>::zeros((4, 5));
    fft_2d_array_static_typed_into::<f32, 4, 5>(&field32, &mut spectrum32);
    for (expected, actual) in expected32.iter().zip(spectrum32.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let recovered32 = ifft_2d_array_typed::<f32>(&expected32);
    let mut actual_recovered32 = Array2::<f32>::zeros((4, 5));
    let mut scratch32 = Array2::<Complex32>::zeros((4, 5));
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
    let mut spectrum16 = Array2::<Complex32>::zeros((4, 5));
    fft_2d_array_static_typed_into::<f16, 4, 5>(&field16, &mut spectrum16);
    for (expected, actual) in expected16.iter().zip(spectrum16.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let recovered16 = ifft_2d_array_typed::<f16>(&expected16);
    let mut actual_recovered16 = Array2::<f16>::from_elem((4, 5), f16::from_f32(0.0));
    let mut scratch16 = Array2::<Complex32>::zeros((4, 5));
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
    let field64 = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| {
        ((i as f64 * 0.11) + (j as f64 * 0.07) - (k as f64 * 0.05)).sin()
    });
    let expected64 = fft_3d_array(&field64);
    let mut spectrum64 = Array3::<Complex64>::zeros((3, 4, 5));
    fft_3d_array_static_into::<3, 4, 5>(&field64, &mut spectrum64);
    for (expected, actual) in expected64.iter().zip(spectrum64.iter()) {
        assert!((expected - actual).norm() < 1e-12);
    }
    let recovered64 = ifft_3d_array(&expected64);
    let mut actual_recovered64 = Array3::<f64>::zeros((3, 4, 5));
    let mut scratch64 = Array3::<Complex64>::zeros((3, 4, 5));
    ifft_3d_array_static_into::<3, 4, 5>(&expected64, &mut actual_recovered64, &mut scratch64);
    for (expected, actual) in recovered64.iter().zip(actual_recovered64.iter()) {
        assert!((expected - actual).abs() < 1e-12);
    }

    let field32 = field64.mapv(|value| value as f32);
    let expected32 = fft_3d_array_typed(&field32);
    let mut spectrum32 = Array3::<Complex32>::zeros((3, 4, 5));
    fft_3d_array_static_typed_into::<f32, 3, 4, 5>(&field32, &mut spectrum32);
    for (expected, actual) in expected32.iter().zip(spectrum32.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let recovered32 = ifft_3d_array_typed::<f32>(&expected32);
    let mut actual_recovered32 = Array3::<f32>::zeros((3, 4, 5));
    let mut scratch32 = Array3::<Complex32>::zeros((3, 4, 5));
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
    let mut spectrum16 = Array3::<Complex32>::zeros((3, 4, 5));
    fft_3d_array_static_typed_into::<f16, 3, 4, 5>(&field16, &mut spectrum16);
    for (expected, actual) in expected16.iter().zip(spectrum16.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let recovered16 = ifft_3d_array_typed::<f16>(&expected16);
    let mut actual_recovered16 = Array3::<f16>::from_elem((3, 4, 5), f16::from_f32(0.0));
    let mut scratch16 = Array3::<Complex32>::zeros((3, 4, 5));
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

    let field32 = Array2::from_shape_fn((4, 6), |(i, j)| {
        ((i as f32 * 0.19) + (j as f32 * 0.07)).sin()
    });
    let expected32 = fft_2d_array_typed(&field32);
    let mut spectrum32 = Array2::<Complex32>::zeros((4, 6));
    fft_2d_array_typed_into(&field32, &mut spectrum32);
    for (expected, actual) in expected32.iter().zip(spectrum32.iter()) {
        assert!((expected - actual).norm() < 1e-5);
    }
    let recovered32 = ifft_2d_array_typed::<f32>(&spectrum32);
    let mut actual_recovered32 = Array2::<f32>::zeros((4, 6));
    let mut scratch32 = Array2::<Complex32>::zeros((4, 6));
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

#[test]
fn test_bench_pot_sizes() {
    use crate::application::execution::kernel::FftPrecision;
    use num_complex::Complex32;
    use std::time::Instant;

    let sizes = [2, 4, 8, 16, 32, 64];
    for &n in &sizes {
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.17).sin(), (k as f32 * 0.29).cos()))
            .collect();
        let mut got = input.clone();
        let start = Instant::now();
        #[cfg(debug_assertions)]
        let iters = 1_000;
        #[cfg(not(debug_assertions))]
        let iters = 1_000_000;
        for _ in 0..iters {
            got.copy_from_slice(&input);
            Complex32::fft_forward(&mut got);
        }
        let elapsed = start.elapsed();
        println!(
            "Size {}: {:.2} ns per iteration (including copy)",
            n,
            (elapsed.as_secs_f64() * 1e9) / iters as f64
        );
    }
}

#[test]
fn test_f64_pot_dfts_correctness() {
    use num_complex::Complex64;
    let sizes = [2, 4, 8, 16, 32, 64];
    for &n in &sizes {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.29).cos()))
            .collect();

        let mut got_fwd = ndarray::Array1::from_vec(input.clone());
        crate::fft_1d_complex_inplace(&mut got_fwd);

        // Compare with naive DFT
        let mut expected_fwd = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * j as f64 * k as f64 / n as f64;
                let w = Complex64::new(angle.cos(), angle.sin());
                sum += input[j] * w;
            }
            expected_fwd[k] = sum;
        }
        for k in 0..n {
            let err = (got_fwd[k] - expected_fwd[k]).norm();
            assert!(
                err < 1e-12,
                "forward: n = {}, k = {}, got_fwd = {:?}, expected = {:?}, err = {}",
                n,
                k,
                got_fwd[k],
                expected_fwd[k],
                err
            );
        }

        let mut got_inv = ndarray::Array1::from_vec(input.clone());
        crate::ifft_1d_complex_inplace(&mut got_inv);
        // Naive IDFT (with normalization)
        let mut expected_inv = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..n {
                let angle = 2.0 * std::f64::consts::PI * j as f64 * k as f64 / n as f64;
                let w = Complex64::new(angle.cos(), angle.sin());
                sum += input[j] * w;
            }
            expected_inv[k] = sum / n as f64;
        }
        for k in 0..n {
            let err = (got_inv[k] - expected_inv[k]).norm();
            assert!(
                err < 1e-12,
                "inverse: n = {}, k = {}, got_inv = {:?}, expected = {:?}, err = {}",
                n,
                k,
                got_inv[k],
                expected_inv[k],
                err
            );
        }
    }
}

#[test]
fn test_debug_twiddles() {
    use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
    use num_complex::Complex64;
    let tw_table = <f64 as MixedRadixScalar>::small_pot_twiddles::<false>(32);
    println!("tw_table len: {}", tw_table.len());

    let get_twiddle = |idx: usize| {
        let half = 16;
        if idx < half {
            tw_table[half - 1 + idx]
        } else {
            -tw_table[half - 1 + idx - half]
        }
    };

    for k in 0..32 {
        let tw = get_twiddle(k);
        let angle = -2.0 * std::f64::consts::PI * k as f64 / 32.0;
        let expected = Complex64::new(angle.cos(), angle.sin());
        let diff = (tw - expected).norm();
        println!(
            "k = {}: got = {:?}, expected = {:?}, diff = {}",
            k, tw, expected, diff
        );
        assert!(diff < 1e-12);
    }
}

#[test]
fn test_f32_pot_plans_correctness() {
    use crate::PlanCacheProvider;
    use num_complex::Complex32;
    let sizes = [2, 4, 8, 16, 32, 64];
    for &n in &sizes {
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.17).sin(), (k as f32 * 0.29).cos()))
            .collect();

        let plan = f32::get_1d_plan(crate::Shape1D::new(n).unwrap());
        let mut got_fwd = input.clone();
        plan.forward_complex_slice_inplace(&mut got_fwd);

        // Compare with naive DFT
        let mut expected_fwd = vec![Complex32::new(0.0, 0.0); n];
        for k in 0..n {
            let mut sum = num_complex::Complex64::new(0.0, 0.0);
            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * j as f64 * k as f64 / n as f64;
                let w = num_complex::Complex64::new(angle.cos(), angle.sin());
                sum += num_complex::Complex64::new(input[j].re as f64, input[j].im as f64) * w;
            }
            expected_fwd[k] = Complex32::new(sum.re as f32, sum.im as f32);
        }
        for k in 0..n {
            let err = (got_fwd[k] - expected_fwd[k]).norm();
            assert!(
                err < 1e-5,
                "forward: n = {}, k = {}, got_fwd = {:?}, expected = {:?}, err = {}",
                n,
                k,
                got_fwd[k],
                expected_fwd[k],
                err
            );
        }

        let mut got_inv = input.clone();
        plan.inverse_complex_slice_inplace(&mut got_inv);
        // Naive IDFT (with normalization)
        let mut expected_inv = vec![Complex32::new(0.0, 0.0); n];
        for k in 0..n {
            let mut sum = num_complex::Complex64::new(0.0, 0.0);
            for j in 0..n {
                let angle = 2.0 * std::f64::consts::PI * j as f64 * k as f64 / n as f64;
                let w = num_complex::Complex64::new(angle.cos(), angle.sin());
                sum += num_complex::Complex64::new(input[j].re as f64, input[j].im as f64) * w;
            }
            expected_inv[k] =
                Complex32::new((sum.re / n as f64) as f32, (sum.im / n as f64) as f32);
        }
        for k in 0..n {
            let err = (got_inv[k] - expected_inv[k]).norm();
            assert!(
                err < 1e-5,
                "inverse: n = {}, k = {}, got_inv = {:?}, expected = {:?}, err = {}",
                n,
                k,
                got_inv[k],
                expected_inv[k],
                err
            );
        }
    }
}
