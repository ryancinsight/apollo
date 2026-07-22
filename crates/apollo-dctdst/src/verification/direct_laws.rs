use crate::infrastructure::kernel::direct::{dct1, dct4, dst1, dst4};
use crate::{DctDstError, DctDstPlan, RealTransformKind};
use eunomia::assert_abs_diff_eq;

// -------------------------------------------------------------------------
// DCT-I
// -------------------------------------------------------------------------

/// DCT-I([1,2,3]) = [8, −2, 0] (Rao & Yip 1990, verified by hand).
///
/// Formula for N=3: X_k = x_0 + (−1)^k x_2 + 2·x_1·cos(πk/2)
///   X_0 = 1 + 3 + 2·2·cos(0)   = 8
///   X_1 = 1 − 3 + 2·2·cos(π/2) = −2
///   X_2 = 1 + 3 + 2·2·cos(π)   = 0
#[test]
fn dct1_known_three_point_value() {
    let mut output = [0.0_f64; 3];
    dct1(&[1.0, 2.0, 3.0], &mut output);
    assert_abs_diff_eq!(output[0], 8.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1], -2.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[2], 0.0, epsilon = 1.0e-12);
}

/// DCT-I is self-inverse: DCT-I(DCT-I(x)) = 2(N−1)·x.
/// Verified for N=3: DCT-I([8,−2,0]) = [4,8,12] = 4·[1,2,3] = 2·2·[1,2,3].
#[test]
fn dct1_self_inverse_direct_n3() {
    let signal = [1.0_f64, 2.0, 3.0];
    let mut first = [0.0_f64; 3];
    let mut second = [0.0_f64; 3];
    dct1(&signal, &mut first);
    dct1(&first, &mut second);
    let scale = 2.0 * (signal.len() - 1) as f64; // 2*(3-1) = 4
    for (actual, expected) in second.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(*actual, *expected * scale, epsilon = 1.0e-10);
    }
}

/// DCT-I with N=2: sum over interior is empty, boundary formula only.
/// X_0 = x_0 + x_1, X_1 = x_0 − x_1.
/// Self-inverse scale = 2*(N−1) = 2.
#[test]
fn dct1_two_point_boundary_only() {
    let signal = [3.0_f64, 7.0];
    let mut first = [0.0_f64; 2];
    let mut second = [0.0_f64; 2];
    dct1(&signal, &mut first);
    assert_abs_diff_eq!(first[0], 10.0, epsilon = 1.0e-12); // 3+7
    assert_abs_diff_eq!(first[1], -4.0, epsilon = 1.0e-12); // 3-7
    dct1(&first, &mut second);
    // 2*(N-1)*x = 2*[3,7] = [6,14]
    assert_abs_diff_eq!(second[0], 6.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(second[1], 14.0, epsilon = 1.0e-12);
}

/// DctI rejects N=1 at plan construction: UnsupportedLength.
#[test]
fn dct1_rejects_length_one() {
    assert_eq!(
        DctDstPlan::new(1, RealTransformKind::DctI).unwrap_err(),
        DctDstError::UnsupportedLength
    );
}

/// DctI also rejects N=0 (EmptyLength takes priority over UnsupportedLength).
#[test]
fn dct1_rejects_length_zero() {
    assert_eq!(
        DctDstPlan::new(0, RealTransformKind::DctI).unwrap_err(),
        DctDstError::EmptyLength
    );
}

/// Plan inverse for DctI recovers original signal: inverse scale = 1/(2(N−1)).
#[test]
fn plan_inverse_roundtrip_dct1() {
    let signal = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let plan = DctDstPlan::new(signal.len(), RealTransformKind::DctI).unwrap();
    let forward = plan.forward(&signal).unwrap();
    let recovered = plan.inverse(&forward).unwrap();
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-11);
    }
}

// -------------------------------------------------------------------------
// DCT-IV
// -------------------------------------------------------------------------

/// DCT-IV([1,3]) matches the analytical formula for N=2.
/// X_0 = cos(π/8) + 3·cos(3π/8), X_1 = cos(3π/8) + 3·cos(9π/8).
#[test]
fn dct4_two_point_known_value() {
    let signal = [1.0_f64, 3.0];
    let mut output = [0.0_f64; 2];
    dct4(&signal, &mut output);
    let pi = std::f64::consts::PI;
    let expected0 = (pi / 8.0).cos() + 3.0 * (3.0 * pi / 8.0).cos();
    let expected1 = (3.0 * pi / 8.0).cos() + 3.0 * (9.0 * pi / 8.0).cos();
    assert_abs_diff_eq!(output[0], expected0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1], expected1, epsilon = 1.0e-12);
}

/// DCT-IV is self-inverse: DCT-IV(DCT-IV(x)) = (N/2)·x.
/// Verified for N=2: scale = 1.
#[test]
fn dct4_self_inverse_direct_n2() {
    let signal = [1.0_f64, 3.0];
    let mut first = [0.0_f64; 2];
    let mut second = [0.0_f64; 2];
    dct4(&signal, &mut first);
    dct4(&first, &mut second);
    let scale = signal.len() as f64 / 2.0; // N/2 = 1
    for (actual, expected) in second.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(*actual, *expected * scale, epsilon = 1.0e-12);
    }
}

/// Plan inverse for DctIV recovers original signal: inverse scale = 2/N.
#[test]
fn plan_inverse_roundtrip_dct4() {
    let signal = [1.0_f64, -2.0, 0.5, 4.0, -1.5, 3.0];
    let plan = DctDstPlan::new(signal.len(), RealTransformKind::DctIV).unwrap();
    let forward = plan.forward(&signal).unwrap();
    let recovered = plan.inverse(&forward).unwrap();
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-11);
    }
}

// -------------------------------------------------------------------------
// DST-I
// -------------------------------------------------------------------------

/// DST-I([1,3]) = [4√3, −2√3] (verified by hand, N=2, N+1=3).
///
/// X_0 = 2(1·sin(π/3) + 3·sin(2π/3)) = 2·(√3/2 + 3√3/2) = 4√3
/// X_1 = 2(1·sin(2π/3) + 3·sin(4π/3)) = 2·(√3/2 − 3√3/2) = −2√3
#[test]
fn dst1_two_point_known_value() {
    let signal = [1.0_f64, 3.0];
    let mut output = [0.0_f64; 2];
    dst1(&signal, &mut output);
    let sqrt3 = 3.0_f64.sqrt();
    assert_abs_diff_eq!(output[0], 4.0 * sqrt3, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1], -2.0 * sqrt3, epsilon = 1.0e-12);
}

/// DST-I is self-inverse: DST-I(DST-I(x)) = 2(N+1)·x.
/// For N=2: DST-I([4√3,−2√3]) = [6,18] = 6·[1,3] = 2·3·[1,3] ✓.
#[test]
fn dst1_self_inverse_direct_n2() {
    let signal = [1.0_f64, 3.0];
    let mut first = [0.0_f64; 2];
    let mut second = [0.0_f64; 2];
    dst1(&signal, &mut first);
    dst1(&first, &mut second);
    let scale = 2.0 * (signal.len() + 1) as f64; // 2*(2+1) = 6
    for (actual, expected) in second.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(*actual, *expected * scale, epsilon = 1.0e-10);
    }
}

/// Plan inverse for DstI recovers original signal: inverse scale = 1/(2(N+1)).
#[test]
fn plan_inverse_roundtrip_dst1() {
    let signal = [1.0_f64, -2.0, 0.5, 4.0];
    let plan = DctDstPlan::new(signal.len(), RealTransformKind::DstI).unwrap();
    let forward = plan.forward(&signal).unwrap();
    let recovered = plan.inverse(&forward).unwrap();
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-11);
    }
}

// -------------------------------------------------------------------------
// DST-IV
// -------------------------------------------------------------------------

/// DST-IV([1,3]) matches the analytical formula for N=2.
/// X_0 = sin(π/8) + 3·sin(3π/8), X_1 = sin(3π/8) + 3·sin(9π/8).
#[test]
fn dst4_two_point_known_value() {
    let signal = [1.0_f64, 3.0];
    let mut output = [0.0_f64; 2];
    dst4(&signal, &mut output);
    let pi = std::f64::consts::PI;
    let expected0 = (pi / 8.0).sin() + 3.0 * (3.0 * pi / 8.0).sin();
    let expected1 = (3.0 * pi / 8.0).sin() + 3.0 * (9.0 * pi / 8.0).sin();
    assert_abs_diff_eq!(output[0], expected0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1], expected1, epsilon = 1.0e-12);
}

/// DST-IV is self-inverse: DST-IV(DST-IV(x)) = (N/2)·x.
/// For N=2: scale = 1, so DST-IV(DST-IV(x)) = x.
#[test]
fn dst4_self_inverse_direct_n2() {
    let signal = [1.0_f64, 3.0];
    let mut first = [0.0_f64; 2];
    let mut second = [0.0_f64; 2];
    dst4(&signal, &mut first);
    dst4(&first, &mut second);
    let scale = signal.len() as f64 / 2.0; // N/2 = 1
    for (actual, expected) in second.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(*actual, *expected * scale, epsilon = 1.0e-12);
    }
}

/// Plan inverse for DstIV recovers original signal: inverse scale = 2/N.
#[test]
fn plan_inverse_roundtrip_dst4() {
    let signal = [1.0_f64, -2.0, 0.5, 4.0, -1.5, 3.0];
    let plan = DctDstPlan::new(signal.len(), RealTransformKind::DstIV).unwrap();
    let forward = plan.forward(&signal).unwrap();
    let recovered = plan.inverse(&forward).unwrap();
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-11);
    }
}
