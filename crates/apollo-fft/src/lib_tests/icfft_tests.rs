//! Tests for forward/inverse complex FFT API (part 2).

use crate::*;
use leto::{Array1, Array2, Array3};
use eunomia::{Complex32, Complex64};

#[test]
fn owned_static_complex_wrappers_reuse_input_allocation() {
    let signal1 = Array1::from_shape_fn([16], |[i]| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let ptr1 = signal1.as_slice().unwrap().as_ptr();
    let actual1 = fft_1d_complex_static::<16>(signal1);
    assert_eq!(ptr1, actual1.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered1 = ifft_1d_complex(&expected1);
    let ptr_recovered1 = expected1.as_slice().unwrap().as_ptr();
    let actual_recovered1 = ifft_1d_complex_static::<16>(expected1);
    assert_eq!(ptr_recovered1, actual_recovered1.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn([4, 5], |[i, j]| {
        let x = (i * 5 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let ptr2 = signal2.as_slice().unwrap().as_ptr();
    let actual2 = fft_2d_complex_static::<4, 5>(signal2);
    assert_eq!(ptr2, actual2.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered2 = ifft_2d_complex(&expected2);
    let ptr_recovered2 = expected2.as_slice().unwrap().as_ptr();
    let actual_recovered2 = ifft_2d_complex_static::<4, 5>(expected2);
    assert_eq!(ptr_recovered2, actual_recovered2.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn([3, 4, 5], |[i, j, k]| {
        let x = ((i * 4 + j) * 5 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let ptr3 = signal3.as_slice().unwrap().as_ptr();
    let actual3 = fft_3d_complex_static::<3, 4, 5>(signal3);
    assert_eq!(ptr3, actual3.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    let recovered3 = ifft_3d_complex(&expected3);
    let ptr_recovered3 = expected3.as_slice().unwrap().as_ptr();
    let actual_recovered3 = ifft_3d_complex_static::<3, 4, 5>(expected3);
    assert_eq!(ptr_recovered3, actual_recovered3.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal32 = Array1::from_shape_fn([16], |[i]| {
        let x = i as f32;
        Complex32::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let plan32 = FftPlan1D::<f32>::new(Shape1D::new(16).expect("valid 1D shape"));
    let mut expected32 = signal32.clone();
    plan32.forward_complex_inplace(&mut expected32);
    let ptr32 = signal32.as_slice().unwrap().as_ptr();
    let actual32 = fft_1d_complex_static_typed::<f32, 16>(signal32);
    assert_eq!(ptr32, actual32.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected32.iter().zip(actual32.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }
    let mut recovered32 = expected32.clone();
    plan32.inverse_complex_inplace(&mut recovered32);
    let ptr_recovered32 = expected32.as_slice().unwrap().as_ptr();
    let actual_recovered32 = ifft_1d_complex_static_typed::<f32, 16>(expected32);
    assert_eq!(ptr_recovered32, actual_recovered32.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered32.iter().zip(actual_recovered32.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }

    let signal32_2d = Array2::from_shape_fn([4, 5], |[i, j]| {
        let x = (i * 5 + j) as f32;
        Complex32::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let plan32_2d = FftPlan2D::<f32>::new(Shape2D::new(4, 5).expect("valid 2D shape"));
    let mut expected32_2d = signal32_2d.clone();
    plan32_2d.forward_complex_inplace(&mut expected32_2d);
    let ptr32_2d = signal32_2d.as_slice().unwrap().as_ptr();
    let actual32_2d = fft_2d_complex_static_typed::<f32, 4, 5>(signal32_2d);
    assert_eq!(ptr32_2d, actual32_2d.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected32_2d.iter().zip(actual32_2d.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let mut recovered32_2d = expected32_2d.clone();
    plan32_2d.inverse_complex_inplace(&mut recovered32_2d);
    let ptr_recovered32_2d = expected32_2d.as_slice().unwrap().as_ptr();
    let actual_recovered32_2d = ifft_2d_complex_static_typed::<f32, 4, 5>(expected32_2d);
    assert_eq!(ptr_recovered32_2d, actual_recovered32_2d.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered32_2d.iter().zip(actual_recovered32_2d.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }

    let signal32_3d = Array3::from_shape_fn([3, 4, 5], |[i, j, k]| {
        let x = ((i * 4 + j) * 5 + k) as f32;
        Complex32::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let plan32_3d = FftPlan3D::<f32>::new(Shape3D::new(3, 4, 5).expect("valid 3D shape"));
    let mut expected32_3d = signal32_3d.clone();
    plan32_3d.forward_complex_inplace(&mut expected32_3d);
    let ptr32_3d = signal32_3d.as_slice().unwrap().as_ptr();
    let actual32_3d = fft_3d_complex_static_typed::<f32, 3, 4, 5>(signal32_3d);
    assert_eq!(ptr32_3d, actual32_3d.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected32_3d.iter().zip(actual32_3d.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let mut recovered32_3d = expected32_3d.clone();
    plan32_3d.inverse_complex_inplace(&mut recovered32_3d);
    let ptr_recovered32_3d = expected32_3d.as_slice().unwrap().as_ptr();
    let actual_recovered32_3d = ifft_3d_complex_static_typed::<f32, 3, 4, 5>(expected32_3d);
    assert_eq!(ptr_recovered32_3d, actual_recovered32_3d.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered32_3d.iter().zip(actual_recovered32_3d.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
}

#[test]
fn typed_static_complex_wrappers_match_dynamic_f32_paths() {
    let signal1 = Array1::from_shape_fn([16], |[i]| {
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

    let signal2 = Array2::from_shape_fn([4, 5], |[i, j]| {
        let x = (i * 5 + j) as f32;
        Complex32::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let plan2 = FftPlan2D::<f32>::new(Shape2D::new(4, 5).expect("valid 2D shape"));
    let mut expected2 = signal2.clone();
    plan2.forward_complex_inplace(&mut expected2);
    let mut actual2 = Array2::<Complex32>::zeros([4, 5]);
    fft_2d_complex_static_typed_into::<f32, 4, 5>(&signal2, &mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let mut recovered2 = expected2.clone();
    plan2.inverse_complex_inplace(&mut recovered2);
    let mut actual_recovered2 = Array2::<Complex32>::zeros([4, 5]);
    ifft_2d_complex_static_typed_into::<f32, 4, 5>(&expected2, &mut actual_recovered2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    fft_2d_complex_static_typed_inplace::<f32, 4, 5>(&mut actual_recovered2);
    ifft_2d_complex_static_typed_inplace::<f32, 4, 5>(&mut actual_recovered2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }

    let signal3 = Array3::from_shape_fn([3, 4, 5], |[i, j, k]| {
        let x = ((i * 4 + j) * 5 + k) as f32;
        Complex32::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let plan3 = FftPlan3D::<f32>::new(Shape3D::new(3, 4, 5).expect("valid 3D shape"));
    let mut expected3 = signal3.clone();
    plan3.forward_complex_inplace(&mut expected3);
    let mut actual3 = Array3::<Complex32>::zeros([3, 4, 5]);
    fft_3d_complex_static_typed_into::<f32, 3, 4, 5>(&signal3, &mut actual3);
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let mut recovered3 = expected3.clone();
    plan3.inverse_complex_inplace(&mut recovered3);
    let mut actual_recovered3 = Array3::<Complex32>::zeros([3, 4, 5]);
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
    let signal1 = Array1::from_shape_fn([16], |[i]| {
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
    let ptr1 = owned1.as_slice().unwrap().as_ptr();
    let actual1_owned = fft_1d_complex_typed_owned(owned1);
    assert_eq!(ptr1, actual1_owned.as_slice().unwrap().as_ptr());
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
    let inverse_ptr1 = owned_inverse1.as_slice().unwrap().as_ptr();
    let inverse1_owned = ifft_1d_complex_typed_owned(owned_inverse1);
    assert_eq!(inverse_ptr1, inverse1_owned.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered1.iter().zip(inverse1_owned.iter()) {
        assert!((expected - actual).norm() < 2e-5);
    }

    let signal2 = Array2::from_shape_fn([4, 5], |[i, j]| {
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
    let mut actual2_into = Array2::<Complex32>::zeros([4, 5]);
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
    let ptr2 = owned2.as_slice().unwrap().as_ptr();
    let actual2_owned = fft_2d_complex_typed_owned(owned2);
    assert_eq!(ptr2, actual2_owned.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected2.iter().zip(actual2_owned.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }

    let mut recovered2 = expected2.clone();
    plan2.inverse_complex_inplace(&mut recovered2);
    let inverse2 = ifft_2d_complex_typed(&expected2);
    for (expected, actual) in recovered2.iter().zip(inverse2.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }
    let mut inverse2_into = Array2::<Complex32>::zeros([4, 5]);
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
    let inverse_ptr2 = owned_inverse2.as_slice().unwrap().as_ptr();
    let inverse2_owned = ifft_2d_complex_typed_owned(owned_inverse2);
    assert_eq!(inverse_ptr2, inverse2_owned.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered2.iter().zip(inverse2_owned.iter()) {
        assert!((expected - actual).norm() < 3e-5);
    }

    let signal3 = Array3::from_shape_fn([3, 4, 5], |[i, j, k]| {
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
    let mut actual3_into = Array3::<Complex32>::zeros([3, 4, 5]);
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
    let ptr3 = owned3.as_slice().unwrap().as_ptr();
    let actual3_owned = fft_3d_complex_typed_owned(owned3);
    assert_eq!(ptr3, actual3_owned.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected3.iter().zip(actual3_owned.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }

    let mut recovered3 = expected3.clone();
    plan3.inverse_complex_inplace(&mut recovered3);
    let inverse3 = ifft_3d_complex_typed(&expected3);
    for (expected, actual) in recovered3.iter().zip(inverse3.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
    let mut inverse3_into = Array3::<Complex32>::zeros([3, 4, 5]);
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
    let inverse_ptr3 = owned_inverse3.as_slice().unwrap().as_ptr();
    let inverse2_owned = ifft_3d_complex_typed_owned(owned_inverse3);
    assert_eq!(inverse_ptr3, inverse2_owned.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered3.iter().zip(inverse2_owned.iter()) {
        assert!((expected - actual).norm() < 4e-5);
    }
}
