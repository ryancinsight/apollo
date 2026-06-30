use super::*;
use apollo_fft::f16;
use approx::assert_relative_eq;
use proptest::prelude::*;

#[test]
fn two_point_transform_matches_reference() {
    let plan = FwhtPlan::new(2).expect("valid plan");
    let input = Array1::from(vec![1.0, 3.0]);
    let output = plan.forward(&input).expect("forward");
    assert_relative_eq!(output[0], 4.0, epsilon = 1.0e-12);
    assert_relative_eq!(output[1], -2.0, epsilon = 1.0e-12);
}

#[test]
fn roundtrip_recovers_input() {
    let plan = FwhtPlan::new(8).expect("valid plan");
    let input = Array1::from(vec![1.0, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0]);
    let fwd = plan.forward(&input).expect("forward");
    let recovered = plan.inverse(&fwd).expect("inverse");
    for (actual, expected) in recovered.iter().zip(input.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn caller_owned_real_paths_match_allocating_paths() {
    let plan = FwhtPlan::new(8).expect("valid plan");
    let input = Array1::from(vec![1.0, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0]);
    let expected_forward = plan.forward(&input).expect("forward");
    let mut forward = Array1::zeros([8]);
    plan.forward_into(&input, &mut forward)
        .expect("forward_into");
    assert_eq!(forward, expected_forward);

    let expected_inverse = plan.inverse(&expected_forward).expect("inverse");
    let mut inverse = Array1::zeros([8]);
    plan.inverse_into(&forward, &mut inverse)
        .expect("inverse_into");
    for (actual, expected) in inverse.iter().zip(expected_inverse.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_real_forward_and_inverse_match_ndarray_path() {
    use leto::Storage;

    let plan = FwhtPlan::new(8).expect("valid plan");
    let signal = vec![1.0, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0];
    let ndarray_input = Array1::from(signal.clone());
    let leto_input = leto::Array1::from_shape_vec([8], signal).expect("leto input");

    let leto_forward = plan.forward_leto(leto_input.view()).expect("leto forward");
    let ndarray_forward = plan.forward(&ndarray_input).expect("ndarray forward");
    for (actual, expected) in leto_forward
        .storage()
        .as_slice()
        .iter()
        .zip(ndarray_forward.iter())
    {
        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
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
        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_real_caller_owned_outputs_match_allocating_path() {
    use leto::Storage;

    let plan = FwhtPlan::new(8).expect("valid plan");
    let signal = vec![1.0, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0];
    let leto_input = leto::Array1::from_shape_vec([8], signal).expect("leto input");
    let mut forward = leto::Array::<f64, leto::MnemosyneStorage<f64>, 1>::zeros_mnemosyne([8]);

    plan.forward_leto_into(leto_input.view(), forward.view_mut())
        .expect("leto forward into");
    let expected_forward = plan
        .forward_leto(leto_input.view())
        .expect("allocating leto forward");
    for (actual, expected) in forward
        .storage()
        .as_slice()
        .iter()
        .zip(expected_forward.storage().as_slice().iter())
    {
        assert_relative_eq!(actual, expected, epsilon = 0.0);
    }

    let mut inverse = leto::Array::<f64, leto::MnemosyneStorage<f64>, 1>::zeros_mnemosyne([8]);
    plan.inverse_leto_into(forward.view(), inverse.view_mut())
        .expect("leto inverse into");
    for (actual, expected) in inverse
        .storage()
        .as_slice()
        .iter()
        .zip(leto_input.storage().as_slice())
    {
        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_real_forward_accepts_strided_logical_view() {
    use leto::{SliceArg, Storage};

    let plan = FwhtPlan::new(8).expect("valid plan");
    let logical = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let interleaved = logical
        .iter()
        .copied()
        .flat_map(|value| [value, 99.0])
        .collect::<Vec<_>>();
    let leto_input = leto::Array1::from_shape_vec([interleaved.len()], interleaved).unwrap();
    let strided = leto_input
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .unwrap();

    let actual = plan.forward_leto(strided).expect("leto forward");
    let expected = plan
        .forward(&Array1::from(logical))
        .expect("ndarray forward");
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_real_forward_scatters_into_strided_output() {
    use leto::{SliceArg, Storage};

    let plan = FwhtPlan::new(8).expect("valid plan");
    let signal = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let leto_input = leto::Array1::from_shape_vec([8], signal.clone()).unwrap();
    let mut output = leto::Array1::from_shape_vec([16], vec![-1.0; 16]).unwrap();
    let strided_output = output
        .view_mut()
        .slice_with_mut::<1>(&[SliceArg::range(Some(0), None, 2)])
        .unwrap();

    plan.forward_leto_into(leto_input.view(), strided_output)
        .expect("leto strided output forward");
    let expected = plan
        .forward(&Array1::from(signal))
        .expect("ndarray forward");
    for index in 0..8 {
        assert_relative_eq!(
            output.storage().as_slice()[index * 2],
            expected[index],
            epsilon = 0.0
        );
        assert_relative_eq!(
            output.storage().as_slice()[index * 2 + 1],
            -1.0,
            epsilon = 0.0
        );
    }
}

#[test]
fn leto_typed_f32_matches_ndarray_typed_path() {
    use leto::Storage;

    let plan = FwhtPlan::new(8).expect("valid plan");
    let signal = Array1::from(vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75]);
    let leto_input = leto::Array1::from_shape_vec([8], signal.iter().copied().collect()).unwrap();
    let mut expected = Array1::<f32>::zeros([8]);
    plan.forward_typed_into(&signal, &mut expected, PrecisionProfile::LOW_PRECISION_F32)
        .expect("ndarray typed forward");

    let actual = plan
        .forward_leto_typed(leto_input.view(), PrecisionProfile::LOW_PRECISION_F32)
        .expect("leto typed forward");
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 0.0);
    }
}

#[test]
fn leto_typed_f32_caller_owned_output_matches_ndarray_typed_path() {
    use leto::Storage;

    let plan = FwhtPlan::new(8).expect("valid plan");
    let signal = Array1::from(vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75]);
    let leto_input = leto::Array1::from_shape_vec([8], signal.iter().copied().collect()).unwrap();
    let mut expected = Array1::<f32>::zeros([8]);
    plan.forward_typed_into(&signal, &mut expected, PrecisionProfile::LOW_PRECISION_F32)
        .expect("ndarray typed forward");

    let mut actual = leto::Array1::from_shape_vec([8], vec![0.0_f32; 8]).unwrap();
    plan.forward_leto_typed_into(
        leto_input.view(),
        actual.view_mut(),
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("leto typed forward into");
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 0.0);
    }
}

#[test]
fn leto_typed_strided_f16_matches_ndarray_typed_path() {
    use leto::{SliceArg, Storage};

    let plan = FwhtPlan::new(8).expect("valid plan");
    let signal = Array1::from(vec![
        f16::from_f32(1.0),
        f16::from_f32(-2.0),
        f16::from_f32(0.5),
        f16::from_f32(2.25),
        f16::from_f32(-4.0),
        f16::from_f32(1.5),
        f16::from_f32(0.0),
        f16::from_f32(-0.75),
    ]);
    let interleaved = signal
        .iter()
        .copied()
        .flat_map(|value| [value, f16::from_f32(99.0)])
        .collect::<Vec<_>>();
    let leto_input = leto::Array1::from_shape_vec([interleaved.len()], interleaved).unwrap();
    let strided = leto_input
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .unwrap();
    let mut expected = Array1::from_elem([8], f16::from_f32(0.0));
    plan.forward_typed_into(
        &signal,
        &mut expected,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("ndarray typed forward");

    let actual = plan
        .forward_leto_typed(strided, PrecisionProfile::MIXED_PRECISION_F16_F32)
        .expect("leto typed forward");
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_relative_eq!(actual.to_f32(), expected.to_f32(), epsilon = 0.0);
    }
}

#[test]
fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
    let plan = FwhtPlan::new(8).expect("valid plan");
    let signal64 = Array1::from(vec![1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75]);
    let expected = plan.forward(&signal64).expect("forward");

    let mut out64 = Array1::zeros([8]);
    plan.forward_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
        .expect("typed f64 forward");
    for (actual, expected) in out64.iter().zip(expected.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
    }

    let signal32 = signal64.mapv(|value| value as f32);
    let mut out32 = Array1::zeros([8]);
    plan.forward_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
        .expect("typed f32 forward");
    for (actual, expected) in out32.iter().zip(expected.iter()) {
        assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
    }

    let signal16 = signal64.mapv(|value| f16::from_f32(value as f32));
    let mut out16 = Array1::from_elem([8], f16::from_f32(0.0));
    plan.forward_typed_into(
        &signal16,
        &mut out16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed mixed f16 forward");
    for (actual, expected) in out16.iter().zip(expected.iter()) {
        let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
    }

    let mut recovered32 = Array1::zeros([8]);
    plan.inverse_typed_into(
        &out32,
        &mut recovered32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed f32 inverse");
    for (actual, expected) in recovered32.iter().zip(signal32.iter()) {
        assert!((f64::from(*actual) - f64::from(*expected)).abs() < 1.0e-5);
    }

    let mut recovered16 = Array1::from_elem([8], f16::from_f32(0.0));
    plan.inverse_typed_into(
        &out16,
        &mut recovered16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed mixed f16 inverse");
    for (actual, expected) in recovered16.iter().zip(signal64.iter()) {
        let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
    }
}

#[test]
fn mixed_f16_typed_paths_reuse_f32_workspace() {
    let plan = FwhtPlan::new(8).expect("valid plan");
    let signal = Array1::from(vec![
        f16::from_f32(1.0),
        f16::from_f32(-2.0),
        f16::from_f32(0.5),
        f16::from_f32(2.25),
        f16::from_f32(-4.0),
        f16::from_f32(1.5),
        f16::from_f32(0.0),
        f16::from_f32(-0.75),
    ]);
    let mut first = Array1::from_elem([8], f16::from_f32(0.0));
    let mut second = Array1::from_elem([8], f16::from_f32(0.0));

    plan.forward_typed_into(
        &signal,
        &mut first,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("first mixed f16 forward");
    let forward_caps =
        crate::application::execution::plan::fwht::storage::typed_scratch_capacities();
    plan.forward_typed_into(
        &signal,
        &mut second,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("second mixed f16 forward");

    assert_eq!(
        crate::application::execution::plan::fwht::storage::typed_scratch_capacities(),
        forward_caps
    );
    assert!(forward_caps.2 >= plan.len());
    for (actual, expected) in second.iter().zip(first.iter()) {
        assert_relative_eq!(actual.to_f32(), expected.to_f32(), epsilon = 0.0);
    }

    let mut recovered_first = Array1::from_elem([8], f16::from_f32(0.0));
    let mut recovered_second = Array1::from_elem([8], f16::from_f32(0.0));
    plan.inverse_typed_into(
        &first,
        &mut recovered_first,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("first mixed f16 inverse");
    let inverse_caps =
        crate::application::execution::plan::fwht::storage::typed_scratch_capacities();
    plan.inverse_typed_into(
        &first,
        &mut recovered_second,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("second mixed f16 inverse");

    assert_eq!(
        crate::application::execution::plan::fwht::storage::typed_scratch_capacities(),
        inverse_caps
    );
    assert!(inverse_caps.2 >= plan.len());
    for (actual, expected) in recovered_second.iter().zip(recovered_first.iter()) {
        assert_relative_eq!(actual.to_f32(), expected.to_f32(), epsilon = 0.0);
    }
}

#[test]
fn typed_path_rejects_profile_storage_mismatch() {
    let plan = FwhtPlan::new(4).expect("valid plan");
    let signal = Array1::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
    let mut output = Array1::zeros([4]);
    assert!(matches!(
        plan.forward_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
        Err(FwhtError::PrecisionMismatch)
    ));
}

#[test]
fn complex_roundtrip_recovers_input() {
    let plan = FwhtPlan::new(4).expect("valid plan");
    let input = Array1::from(vec![
        Complex64::new(1.0, -1.0),
        Complex64::new(2.0, 0.5),
        Complex64::new(-0.75, 0.25),
        Complex64::new(0.125, -0.625),
    ]);
    let fwd = plan.forward_complex(&input).expect("forward_complex");
    let recovered = plan.inverse_complex(&fwd).expect("inverse_complex");
    for (actual, expected) in recovered.iter().zip(input.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn caller_owned_complex_paths_match_allocating_paths() {
    let plan = FwhtPlan::new(4).expect("valid plan");
    let input = Array1::from(vec![
        Complex64::new(1.0, -1.0),
        Complex64::new(2.0, 0.5),
        Complex64::new(-0.75, 0.25),
        Complex64::new(0.125, -0.625),
    ]);
    let expected_forward = plan.forward_complex(&input).expect("forward_complex");
    let mut forward = Array1::from_elem([4], Complex64::new(0.0, 0.0));
    plan.forward_complex_into(&input, &mut forward)
        .expect("forward_complex_into");
    assert_eq!(forward, expected_forward);

    let expected_inverse = plan
        .inverse_complex(&expected_forward)
        .expect("inverse_complex");
    let mut inverse = Array1::from_elem([4], Complex64::new(0.0, 0.0));
    plan.inverse_complex_into(&forward, &mut inverse)
        .expect("inverse_complex_into");
    for (actual, expected) in inverse.iter().zip(expected_inverse.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn rejects_invalid_lengths() {
    assert!(matches!(FwhtPlan::new(0), Err(FwhtError::EmptyInput)));
    assert!(matches!(FwhtPlan::new(3), Err(FwhtError::NonPowerOfTwo)));
}

#[test]
fn length_mismatch_returns_error() {
    let plan = FwhtPlan::new(4).expect("valid plan");
    let wrong = Array1::from(vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        plan.forward(&wrong),
        Err(FwhtError::LengthMismatch)
    ));
    assert!(matches!(
        plan.inverse(&wrong),
        Err(FwhtError::LengthMismatch)
    ));
    let wrong_c = Array1::from(vec![Complex64::new(1.0, 0.0); 3]);
    assert!(matches!(
        plan.forward_complex(&wrong_c),
        Err(FwhtError::LengthMismatch)
    ));
    assert!(matches!(
        plan.inverse_complex(&wrong_c),
        Err(FwhtError::LengthMismatch)
    ));
    let mut output_c = Array1::from(vec![Complex64::new(0.0, 0.0); 4]);
    assert!(matches!(
        plan.forward_complex_into(&wrong_c, &mut output_c),
        Err(FwhtError::LengthMismatch)
    ));
    assert!(matches!(
        plan.inverse_complex_into(&wrong_c, &mut output_c),
        Err(FwhtError::LengthMismatch)
    ));
    let leto_wrong = leto::Array1::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();
    let mut leto_output = leto::Array1::from_shape_vec([4], vec![0.0; 4]).unwrap();
    assert!(matches!(
        plan.forward_leto_into(leto_wrong.view(), leto_output.view_mut()),
        Err(FwhtError::LengthMismatch)
    ));
    let leto_input = leto::Array1::from_shape_vec([4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut leto_wrong_output = leto::Array1::from_shape_vec([3], vec![0.0; 3]).unwrap();
    assert!(matches!(
        plan.inverse_leto_into(leto_input.view(), leto_wrong_output.view_mut()),
        Err(FwhtError::LengthMismatch)
    ));
}

#[test]
fn single_element_is_identity() {
    let plan = FwhtPlan::new(1).expect("valid plan");
    let input = Array1::from(vec![42.0f64]);
    let fwd = plan.forward(&input).expect("forward");
    assert_relative_eq!(fwd[0], 42.0, epsilon = 1.0e-12);
    let inv = plan.inverse(&fwd).expect("inverse");
    assert_relative_eq!(inv[0], 42.0, epsilon = 1.0e-12);
}

#[test]
fn involution_property() {
    let plan = FwhtPlan::new(8).expect("valid plan");
    let input = Array1::from(vec![1.0, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0]);
    let fwd1 = plan.forward(&input).expect("fwd1");
    let fwd2 = plan.forward(&fwd1).expect("fwd2");
    for (actual, expected) in fwd2.iter().zip(input.iter()) {
        assert_relative_eq!(*actual, *expected * 8.0, epsilon = 1.0e-10);
    }
}

proptest::proptest! {
    #[test]
    fn roundtrip_holds_for_random_power_of_two_lengths(
        power in 1usize..12,
        samples in prop::collection::vec(-10.0f64..10.0f64, 1usize..4096)
    ) {
        let n = 1usize << power;
        let input = Array1::from(
            samples.into_iter().cycle().take(n).collect::<Vec<_>>()
        );
        let plan = FwhtPlan::new(n).expect("valid plan");
        let fwd = plan.forward(&input).expect("forward");
        let recovered = plan.inverse(&fwd).expect("inverse");
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            prop_assert!((actual - expected).abs() < 1.0e-10);
        }
    }
}
