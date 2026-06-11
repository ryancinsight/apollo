//! Unit tests for 1D Fractional Fourier Transform.

use super::plan::{FrftPlan, frft, frft_leto};
use crate::domain::contracts::error::FrftError;
use apollo_fft::{f16, PrecisionProfile};
use ndarray::Array1;
use num_complex::{Complex32, Complex64};

#[test]
fn integer_order_zero_is_identity() {
    let input = Array1::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]);
    assert_eq!(frft(&input, 0.0).expect("frft"), input);
}

#[test]
fn leto_forward_and_inverse_match_ndarray_path() {
    use leto::Storage;

    let n = 8;
    let plan = FrftPlan::new(n, 0.75).expect("valid plan");
    let signal = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.17).cos(), (i as f64 * 0.23).sin()))
        .collect::<Vec<_>>();
    let ndarray_input = Array1::from_vec(signal.clone());
    let leto_input = leto::Array1::from_shape_vec([n], signal).expect("leto input");

    let leto_forward = plan.forward_leto(leto_input.view()).expect("leto forward");
    let ndarray_forward = plan.forward(&ndarray_input).expect("ndarray forward");
    assert_eq!(leto_forward.shape(), [ndarray_forward.len()]);
    for (actual, expected) in leto_forward
        .storage()
        .as_slice()
        .iter()
        .zip(ndarray_forward.iter())
    {
        assert!((actual - expected).norm() < 1.0e-12);
    }

    let leto_inverse = plan
        .inverse_leto(leto_forward.view())
        .expect("leto inverse");
    let ndarray_inverse = plan.inverse(&ndarray_forward).expect("ndarray inverse");
    for (actual, expected) in leto_inverse
        .storage()
        .as_slice()
        .iter()
        .zip(ndarray_inverse.iter())
    {
        assert!((actual - expected).norm() < 1.0e-12);
    }
}

#[test]
fn leto_forward_accepts_strided_logical_view() {
    use leto::{SliceArg, Storage};

    let n = 8;
    let logical = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.19).sin(), (i as f64 * 0.29).cos()))
        .collect::<Vec<_>>();
    let interleaved = logical
        .iter()
        .flat_map(|&value| [value, Complex64::new(99.0, -99.0)])
        .collect::<Vec<_>>();
    let leto_input =
        leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto input");
    let strided = leto_input
        .view()
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided view");

    let via_leto = frft_leto(strided, 0.5).expect("leto frft");
    let via_ndarray = frft(&Array1::from_vec(logical), 0.5).expect("ndarray frft");
    for (actual, expected) in via_leto.storage().as_slice().iter().zip(via_ndarray.iter()) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
}

#[test]
fn leto_typed_complex32_matches_ndarray_typed_path() {
    use leto::Storage;

    let n = 8;
    let plan = FrftPlan::new(n, 0.75).expect("valid plan");
    let input = Array1::from_shape_fn(n, |i| {
        Complex32::new((i as f32 * 0.17).cos(), (i as f32 * 0.23).sin())
    });
    let leto_input =
        leto::Array1::from_shape_vec([n], input.iter().copied().collect()).expect("leto input");

    let mut expected = Array1::<Complex32>::zeros(n);
    plan.forward_typed_into(&input, &mut expected, PrecisionProfile::LOW_PRECISION_F32)
        .expect("ndarray typed forward");
    let actual = plan
        .forward_leto_typed(leto_input.view(), PrecisionProfile::LOW_PRECISION_F32)
        .expect("leto typed forward");

    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert!((actual.re - expected.re).abs() < 1.0e-7);
        assert!((actual.im - expected.im).abs() < 1.0e-7);
    }
}

#[test]
fn leto_typed_strided_f16_matches_ndarray_typed_path() {
    use leto::{SliceArg, Storage};

    let n = 8;
    let plan = FrftPlan::new(n, 0.5).expect("valid plan");
    let logical = Array1::from_shape_fn(n, |i| {
        [
            f16::from_f32((i as f32 * 0.13).sin()),
            f16::from_f32((i as f32 * 0.31).cos()),
        ]
    });
    let interleaved = logical
        .iter()
        .copied()
        .flat_map(|value| [value, [f16::from_f32(99.0), f16::from_f32(-99.0)]])
        .collect::<Vec<_>>();
    let leto_input =
        leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto input");
    let strided = leto_input
        .view()
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided view");

    let mut expected = Array1::from_elem(n, [f16::from_f32(0.0); 2]);
    plan.forward_typed_into(
        &logical,
        &mut expected,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("ndarray typed forward");
    let actual = plan
        .forward_leto_typed(strided, PrecisionProfile::MIXED_PRECISION_F16_F32)
        .expect("leto typed forward");

    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_eq!(actual[0].to_f32(), expected[0].to_f32());
        assert_eq!(actual[1].to_f32(), expected[1].to_f32());
    }
}

#[test]
fn exact_centered_continuity_at_boundary() {
    let n: usize = 16;
    let input = Array1::from_shape_fn(n, |i| Complex64::new((i as f64 * 0.1).sin(), 0.0));
    let boundary = frft(&input, 1.0).unwrap();
    let near_limit = frft(&input, 0.9999999999).unwrap();

    for (a, b) in boundary.iter().zip(near_limit.iter()) {
        assert!((a.re - b.re).abs() < 1.0e-6);
        assert!((a.im - b.im).abs() < 1.0e-6);
    }
}

#[test]
fn integer_order_one_inverse_recovers_input() {
    let n: usize = 8;
    let plan = FrftPlan::new(n, 1.0).expect("valid plan");
    let input = Array1::from_shape_fn(n, |i| {
        Complex64::new((i as f64 * 0.17).cos(), (i as f64 * 0.23).sin())
    });
    let recovered = plan
        .inverse(&plan.forward(&input).expect("forward"))
        .expect("inverse");

    for (actual, expected) in recovered.iter().zip(input.iter()) {
        assert!((actual.re - expected.re).abs() < 1.0e-12);
        assert!((actual.im - expected.im).abs() < 1.0e-12);
    }
}

#[test]
fn inverse_into_matches_allocating_inverse() {
    let n: usize = 8;
    let plan = FrftPlan::new(n, 3.0).expect("valid plan");
    let input = Array1::from_shape_fn(n, |i| Complex64::new(i as f64 - 2.0, i as f64 * 0.5));
    let expected = plan.inverse(&input).expect("inverse");
    let mut actual = Array1::<Complex64>::zeros(n);
    plan.inverse_into(&input, &mut actual)
        .expect("inverse_into");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual.re - expected.re).abs() < 1.0e-12);
        assert!((actual.im - expected.im).abs() < 1.0e-12);
    }
}

#[test]
fn typed_paths_support_complex64_complex32_and_mixed_f16_storage() {
    let n: usize = 8;
    let plan = FrftPlan::new(n, 0.75).expect("valid plan");
    let input64 = Array1::from_shape_fn(n, |i| {
        Complex64::new((i as f64 * 0.17).cos(), (i as f64 * 0.23).sin())
    });
    let expected = plan.forward(&input64).expect("forward");

    let mut out64 = Array1::<Complex64>::zeros(n);
    plan.forward_typed_into(&input64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
        .expect("complex64 forward");
    for (actual, expected) in out64.iter().zip(expected.iter()) {
        assert!((actual - expected).norm() < 1.0e-12);
    }

    let input32 = input64.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
    let mut out32 = Array1::<Complex32>::zeros(n);
    plan.forward_typed_into(&input32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
        .expect("complex32 forward");
    for (actual, expected) in out32.iter().zip(expected.iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
        assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
    }

    let input16 = input64.mapv(|value| {
        [
            f16::from_f32(value.re as f32),
            f16::from_f32(value.im as f32),
        ]
    });
    let mut out16 = Array1::from_elem(n, [f16::from_f32(0.0); 2]);
    let input16_reference = input16.mapv(|value| {
        Complex64::new(f64::from(value[0].to_f32()), f64::from(value[1].to_f32()))
    });
    let expected16 = plan.forward(&input16_reference).expect("mixed reference");
    plan.forward_typed_into(
        &input16,
        &mut out16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("mixed f16 forward");
    for (actual, expected) in out16.iter().zip(expected16.iter()) {
        let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
        assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
    }

    let mut recovered32 = Array1::<Complex32>::zeros(n);
    plan.inverse_typed_into(
        &out32,
        &mut recovered32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("complex32 inverse");
    let out32_reference =
        out32.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
    let expected_recovered32 = plan.inverse(&out32_reference).expect("inverse reference");
    for (actual, expected) in recovered32.iter().zip(expected_recovered32.iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
        assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
    }
}

#[test]
fn typed_complex32_path_reuses_complex64_workspaces() {
    let n: usize = 8;
    let plan = FrftPlan::new(n, 0.5).expect("valid plan");
    let input = Array1::from_shape_fn(n, |i| {
        Complex32::new((i as f32 * 0.17).cos(), (i as f32 * 0.23).sin())
    });
    let mut first = Array1::<Complex32>::zeros(n);
    let mut second = Array1::<Complex32>::zeros(n);

    plan.forward_typed_into(&input, &mut first, PrecisionProfile::LOW_PRECISION_F32)
        .expect("first typed forward");
    let first_caps =
        crate::application::execution::plan::frft::storage::typed_scratch_capacities();
    plan.forward_typed_into(&input, &mut second, PrecisionProfile::LOW_PRECISION_F32)
        .expect("second typed forward");
    let second_caps =
        crate::application::execution::plan::frft::storage::typed_scratch_capacities();

    assert_eq!(first_caps, second_caps);
    assert!(first_caps.0 >= n);
    assert!(first_caps.1 >= n);
    for (actual, expected) in second.iter().zip(first.iter()) {
        assert!((actual.re - expected.re).abs() < 1.0e-7);
        assert!((actual.im - expected.im).abs() < 1.0e-7);
    }
}

#[test]
fn typed_path_rejects_profile_storage_mismatch() {
    let n: usize = 4;
    let plan = FrftPlan::new(n, 1.0).expect("valid plan");
    let input = Array1::from_elem(n, Complex32::new(1.0, 0.0));
    let mut output = Array1::<Complex32>::zeros(n);
    assert!(matches!(
        plan.forward_typed_into(&input, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
        Err(FrftError::PrecisionMismatch)
    ));
}

#[test]
fn rejects_invalid_plan() {
    assert_eq!(FrftPlan::new(0, 1.0), Err(FrftError::EmptySignal));
    assert_eq!(FrftPlan::new(4, f64::NAN), Err(FrftError::NonFiniteOrder));
}

#[test]
fn frft_order_1_matches_dft() {
    use std::f64::consts::PI;
    let n = 8usize;
    let input: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.31).sin(), 0.0))
        .collect();
    let input_arr = Array1::from_vec(input.clone());
    let result = frft(&input_arr, 1.0).unwrap();
    let center = (n as f64 - 1.0) * 0.5;
    let scale = 1.0 / (n as f64).sqrt();
    for k in 0..n {
        let u = k as f64 - center;
        let expected: Complex64 = input
            .iter()
            .enumerate()
            .map(|(j, &val)| {
                let x = j as f64 - center;
                let angle = -2.0 * PI * x * u / n as f64;
                val * Complex64::new(angle.cos(), angle.sin())
            })
            .sum::<Complex64>()
            * scale;
        assert!(
            (result[k] - expected).norm() < 1e-12,
            "FrFT(1) != centered DFT reference at k={}",
            k
        );
    }
}

#[test]
fn frft_order_4_is_identity() {
    let n = 8usize;
    let input: Array1<Complex64> = Array1::from_shape_fn(n, |i| {
        Complex64::new((i as f64 * 0.5).cos(), (i as f64 * 0.3).sin())
    });
    let plan = FrftPlan::new(n, 4.0).unwrap();
    let result = plan.forward(&input).unwrap();
    for (orig, out) in input.iter().zip(result.iter()) {
        assert!(
            (orig - out).norm() < 1e-12,
            "FrFT(4) is not identity: diff = {}",
            (orig - out).norm()
        );
    }
}

#[test]
fn frft_order_2_is_reversal() {
    let n = 8usize;
    let input: Array1<Complex64> =
        Array1::from_shape_fn(n, |i| Complex64::new(i as f64 + 1.0, 0.0));
    let plan = FrftPlan::new(n, 2.0).unwrap();
    let result = plan.forward(&input).unwrap();
    for k in 0..n {
        let expected = input[n - 1 - k];
        assert!(
            (result[k] - expected).norm() < 1e-12,
            "FrFT(2) reversal failed at k={}: got {:?}, expected {:?}",
            k,
            result[k],
            expected
        );
    }
}
