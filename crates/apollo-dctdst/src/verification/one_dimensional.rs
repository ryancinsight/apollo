use crate::infrastructure::kernel::direct::{dct2, dst2};
use crate::infrastructure::kernel::fast::{dct2_dst2_fast, dct2_fast, dst2_fast};
use crate::{DctDstError, DctDstPlan, RealTransformKind};
use approx::assert_abs_diff_eq;

#[test]
fn plan_preserves_validated_configuration() {
    let plan = DctDstPlan::new(16, RealTransformKind::DctII).expect("plan");
    assert_eq!(plan.len(), 16);
    assert_eq!(plan.kind(), RealTransformKind::DctII);
    assert!(!plan.is_empty());
}

#[test]
fn rejects_zero_length() {
    assert_eq!(
        DctDstPlan::new(0, RealTransformKind::DstIII).unwrap_err(),
        DctDstError::EmptyLength
    );
}

#[test]
fn dct2_matches_known_two_point_projection() {
    let plan = DctDstPlan::new(2, RealTransformKind::DctII).expect("plan");
    let mut output = [0.0; 2];
    plan.forward_into(&[1.0, 3.0], &mut output)
        .expect("forward");
    assert_abs_diff_eq!(output[0], 4.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1], -std::f64::consts::SQRT_2, epsilon = 1.0e-12);
}

#[test]
fn dct3_inverts_dct2_with_standard_scale() {
    let signal = [1.0, -2.0, 0.5, 4.0];
    let forward = DctDstPlan::new(signal.len(), RealTransformKind::DctII).expect("forward");
    let inverse = DctDstPlan::new(signal.len(), RealTransformKind::DctIII).expect("inverse");
    let mut coefficients = [0.0; 4];
    let mut recovered = [0.0; 4];
    forward
        .forward_into(&signal, &mut coefficients)
        .expect("dct2");
    inverse
        .forward_into(&coefficients, &mut recovered)
        .expect("dct3");
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(
            *actual * 2.0 / signal.len() as f64,
            *expected,
            epsilon = 1.0e-12
        );
    }
}

#[test]
fn dct3_fast_boundary_impulse_matches_direct() {
    let mut signal = vec![0.0_f64; 32];
    signal[31] = 0.4074011305980829;
    let mut coefficients = vec![0.0_f64; 32];
    DctDstPlan::new(signal.len(), RealTransformKind::DctII)
        .expect("dct2 plan")
        .forward_into(&signal, &mut coefficients)
        .expect("planned dct2");
    let mut direct = vec![0.0_f64; 32];
    let mut fast = vec![0.0_f64; 32];
    crate::infrastructure::kernel::direct::dct3(&coefficients, &mut direct);
    crate::infrastructure::kernel::fast::dct3_fast(&coefficients, &mut fast);
    for (actual, expected) in fast.iter().zip(direct.iter()) {
        assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-11);
    }
    for (actual, expected) in fast.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(
            *actual * 2.0 / signal.len() as f64,
            *expected,
            epsilon = 1.0e-11
        );
    }
}

#[test]
fn dst2_matches_known_two_point_projection() {
    let plan = DctDstPlan::new(2, RealTransformKind::DstII).expect("plan");
    let mut output = [0.0; 2];
    plan.forward_into(&[1.0, 3.0], &mut output)
        .expect("forward");
    assert_abs_diff_eq!(
        output[0],
        1.0 * (std::f64::consts::PI / 4.0).sin() + 3.0 * (3.0 * std::f64::consts::PI / 4.0).sin(),
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(output[1], -2.0, epsilon = 1.0e-12);
}

#[test]
fn dst3_inverts_dst2_with_standard_scale() {
    let signal = [1.0, -2.0, 0.5, 4.0];
    let forward = DctDstPlan::new(signal.len(), RealTransformKind::DstII).expect("forward");
    let inverse = DctDstPlan::new(signal.len(), RealTransformKind::DstIII).expect("inverse");
    let mut coefficients = [0.0; 4];
    let mut recovered = [0.0; 4];
    forward
        .forward_into(&signal, &mut coefficients)
        .expect("dst2");
    inverse
        .forward_into(&coefficients, &mut recovered)
        .expect("dst3");
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(
            *actual * 2.0 / signal.len() as f64,
            *expected,
            epsilon = 1.0e-12
        );
    }
}

#[test]
fn rejects_execution_length_mismatch() {
    let plan = DctDstPlan::new(4, RealTransformKind::DctII).expect("plan");
    let mut output = [0.0; 4];
    assert_eq!(
        plan.forward_into(&[1.0, 2.0], &mut output).unwrap_err(),
        DctDstError::LengthMismatch
    );
}

/// DCT-II n=32: plan dispatch and direct kernel produce identical output.
/// Guards against future dispatch regressions (serial path, below PAR_THRESHOLD=256).
#[test]
fn large_plan_path_matches_direct_kernels() {
    let signal: Vec<f64> = (0..32)
        .map(|index| {
            let x = index as f64;
            (0.011 * x).sin() + 0.25 * (0.037 * x).cos()
        })
        .collect();
    let plan = DctDstPlan::new(32, RealTransformKind::DctII).expect("plan");
    let actual = plan.forward(&signal).expect("plan forward");
    let mut expected = vec![0.0_f64; 32];
    dct2(&signal, &mut expected);
    let error = actual
        .iter()
        .zip(expected.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        error < 1.0e-12,
        "plan dispatch diverges from direct kernel: err={error}"
    );
}

#[test]
fn fast_single_projection_paths_match_dual_projection_kernel() {
    let signal: Vec<f64> = (0..32)
        .map(|index| {
            let x = index as f64;
            (0.021 * x).sin() - 0.5 * (0.047 * x).cos() + 0.125 * (0.13 * x).sin()
        })
        .collect();

    let mut dct_single = vec![0.0_f64; signal.len()];
    let mut dst_single = vec![0.0_f64; signal.len()];
    let mut dct_dual = vec![0.0_f64; signal.len()];
    let mut dst_dual = vec![0.0_f64; signal.len()];
    let mut dct_reference = vec![0.0_f64; signal.len()];
    let mut dst_reference = vec![0.0_f64; signal.len()];

    dct2_fast(&signal, &mut dct_single);
    dst2_fast(&signal, &mut dst_single);
    dct2_dst2_fast(&signal, &mut dct_dual, &mut dst_dual);
    dct2(&signal, &mut dct_reference);
    dst2(&signal, &mut dst_reference);

    for ((single, dual), reference) in dct_single
        .iter()
        .zip(dct_dual.iter())
        .zip(dct_reference.iter())
    {
        assert_abs_diff_eq!(single, dual, epsilon = 1.0e-12);
        assert_abs_diff_eq!(single, reference, epsilon = 1.0e-12);
    }
    for ((single, dual), reference) in dst_single
        .iter()
        .zip(dst_dual.iter())
        .zip(dst_reference.iter())
    {
        assert_abs_diff_eq!(single, dual, epsilon = 1.0e-12);
        assert_abs_diff_eq!(single, reference, epsilon = 1.0e-12);
    }
}

/// DCT-II of a single element equals the element itself.
/// Formula: X[0] = x[0] * cos(pi/N * (0 + 0.5) * 0) = x[0] * cos(0) = x[0].
#[test]
fn dct2_single_element_equals_input() {
    let plan = DctDstPlan::new(1, RealTransformKind::DctII).unwrap();
    let result = plan.forward(&[3.7_f64]).unwrap();
    assert!(
        (result[0] - 3.7).abs() < 1e-14,
        "DCT-II n=1: expected 3.7, got {}",
        result[0]
    );
}

/// Allocating forward returns LengthMismatch when signal length differs from the plan.
#[test]
fn length_mismatch_returns_correct_error() {
    let plan = DctDstPlan::new(4, RealTransformKind::DctII).unwrap();
    let result = plan.forward(&[1.0, 2.0, 3.0]);
    assert!(matches!(result, Err(DctDstError::LengthMismatch)));
}

/// Caller-owned inverse output matches the allocating inverse path.
#[test]
fn inverse_into_matches_allocating_inverse() {
    let signal = [1.0, -2.0, 0.5, 4.0];
    for kind in [RealTransformKind::DctII, RealTransformKind::DstII] {
        let plan = DctDstPlan::new(signal.len(), kind).unwrap();
        let coeffs = plan.forward(&signal).unwrap();
        let expected = plan.inverse(&coeffs).unwrap();
        let mut actual = vec![0.0_f64; signal.len()];
        plan.inverse_into(&coeffs, &mut actual).unwrap();
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-12);
        }
    }
}
