use leto::{SliceArg, Storage};

use crate::{
    infrastructure::transport::gpu::{DctDstWgpuPlan, WgpuError},
    RealTransformKind,
};

use super::support::{
    assert_cpu_differential, assert_roundtrip, backend, cpu_forward, ONE_DIMENSIONAL_TOLERANCE,
};

#[test]
fn forward_matches_cpu_for_each_supported_kind() {
    let Some(backend) = backend() else {
        return;
    };
    let cases: [(RealTransformKind, &[f32]); 8] = [
        (
            RealTransformKind::DctII,
            &[1.0, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75],
        ),
        (
            RealTransformKind::DctIII,
            &[0.75, -1.0, 2.5, -0.25, 3.0, 0.5],
        ),
        (RealTransformKind::DstII, &[1.0, -0.5, 2.0, -1.5, 0.25, 3.0]),
        (
            RealTransformKind::DstIII,
            &[0.25, -1.0, 2.0, 0.5, -0.75, 1.25],
        ),
        (RealTransformKind::DctI, &[1.0, -0.5, 2.0, 0.75]),
        (
            RealTransformKind::DctIV,
            &[0.5, -1.0, 2.5, -0.25, 1.5, 0.75],
        ),
        (RealTransformKind::DstI, &[1.0, -2.0, 0.5, 3.0, -1.5]),
        (RealTransformKind::DstIV, &[0.75, -1.25, 2.0, -0.5, 1.0]),
    ];

    for (kind, input) in cases {
        let plan = backend.plan(input.len(), kind);
        let actual = backend.execute_forward(&plan, input).expect("GPU forward");
        assert_cpu_differential(
            &actual,
            &cpu_forward(input, kind),
            ONE_DIMENSIONAL_TOLERANCE,
        );
    }
}

#[test]
fn inverse_recovers_each_existing_roundtrip_case() {
    let Some(backend) = backend() else {
        return;
    };
    let cases: [(RealTransformKind, &[f32]); 6] = [
        (
            RealTransformKind::DctII,
            &[0.25, -1.25, 2.0, -0.5, 3.0, 1.5],
        ),
        (
            RealTransformKind::DstII,
            &[0.75, -1.25, 0.5, 2.0, -0.5, 1.0],
        ),
        (RealTransformKind::DctI, &[1.0, -0.5, 2.0, 0.75]),
        (
            RealTransformKind::DctIV,
            &[0.5, -1.0, 2.5, -0.25, 1.5, 0.75],
        ),
        (RealTransformKind::DstI, &[1.0, -2.0, 0.5, 3.0, -1.5]),
        (RealTransformKind::DstIV, &[0.75, -1.25, 2.0, -0.5, 1.0]),
    ];

    for (kind, input) in cases {
        let plan = backend.plan(input.len(), kind);
        let spectrum = backend.execute_forward(&plan, input).expect("GPU forward");
        let actual = backend
            .execute_inverse(&plan, &spectrum)
            .expect("GPU inverse");
        assert_roundtrip(&actual, input, ONE_DIMENSIONAL_TOLERANCE as f32);
    }
}

#[test]
fn caller_owned_execution_matches_allocating_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let input = [0.5_f32, -1.25, 2.0, -0.75, 0.25, 1.5];
    let plan = backend.plan(input.len(), RealTransformKind::DctII);
    let expected_forward = backend.execute_forward(&plan, &input).expect("forward");
    let mut forward = [0.0_f32; 6];
    backend
        .execute_forward_into(&plan, &input, &mut forward)
        .expect("caller-owned forward");
    assert_eq!(forward.as_slice(), expected_forward.as_slice());

    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward)
        .expect("inverse");
    let mut inverse = [0.0_f32; 6];
    backend
        .execute_inverse_into(&plan, &expected_forward, &mut inverse)
        .expect("caller-owned inverse");
    assert_eq!(inverse.as_slice(), expected_inverse.as_slice());
}

#[test]
fn leto_boundaries_match_slice_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![0.25_f32, -1.25, 2.0, -0.5, 3.0, 1.5];
    let plan = backend.plan(input.len(), RealTransformKind::DctII);
    let expected_forward = backend.execute_forward(&plan, &input).expect("forward");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto input");
    let actual_forward = backend
        .execute_forward_leto(&plan, leto_input.view())
        .expect("Leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward)
        .expect("inverse");
    let leto_spectrum = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("Leto spectrum");
    let actual_inverse = backend
        .execute_inverse_leto(&plan, leto_spectrum.view())
        .expect("Leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn strided_leto_input_matches_its_logical_slice() {
    let Some(backend) = backend() else {
        return;
    };
    let logical = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5];
    let backing = logical
        .iter()
        .copied()
        .flat_map(|value| [value, 99.0])
        .collect::<Vec<_>>();
    let plan = backend.plan(logical.len(), RealTransformKind::DctII);
    let expected = backend
        .execute_forward(&plan, &logical)
        .expect("slice forward");
    let leto_input = leto::Array1::from_shape_vec([backing.len()], backing).expect("Leto input");
    let strided = leto_input
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided Leto view");
    let actual = backend
        .execute_forward_leto(&plan, strided)
        .expect("strided Leto forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}

#[test]
fn rejects_invalid_plan_and_length_contracts() {
    let Some(backend) = backend() else {
        return;
    };
    let empty_error = backend
        .execute_forward(&DctDstWgpuPlan::new(0, RealTransformKind::DctII), &[])
        .expect_err("empty plan must fail");
    assert!(matches!(empty_error, WgpuError::InvalidPlan { .. }));

    let mismatch_error = backend
        .execute_forward(&DctDstWgpuPlan::new(8, RealTransformKind::DctII), &[0.0; 4])
        .expect_err("length mismatch must fail");
    assert!(matches!(
        mismatch_error,
        WgpuError::LengthMismatch {
            expected: 8,
            actual: 4,
        }
    ));

    let dct_one_error = backend
        .execute_forward(&DctDstWgpuPlan::new(1, RealTransformKind::DctI), &[0.5])
        .expect_err("DCT-I length one must fail");
    assert!(matches!(dct_one_error, WgpuError::InvalidPlan { .. }));
}
