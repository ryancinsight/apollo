//! Verification modules for DCT/DST.

#[cfg(test)]
mod tests {
    use crate::infrastructure::kernel::direct::{dct1, dct2, dct4, dst1, dst2, dst4};
    use crate::infrastructure::kernel::fast::{dct2_dst2_fast, dct2_fast, dst2_fast};
    use crate::{DctDstError, DctDstPlan, RealTransformKind};
    use approx::assert_abs_diff_eq;
    use leto::{Array2, Array3};
    use proptest::prelude::*;

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
            1.0 * (std::f64::consts::PI / 4.0).sin()
                + 3.0 * (3.0 * std::f64::consts::PI / 4.0).sin(),
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

    #[test]
    fn forward_2d_matches_separable_manual_application() {
        let plan = DctDstPlan::new(4, RealTransformKind::DctII).unwrap();
        let input = Array2::from_shape_vec(
            [4, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, //
                0.5, -1.0, 2.0, 1.5, //
                -2.0, 0.0, 1.0, 3.0, //
                4.0, -3.0, 2.5, -0.5,
            ],
        )
        .unwrap();

        let actual = plan.forward_2d(&input).unwrap();

        let mut row_stage = Array2::<f64>::zeros([4, 4]);
        let mut expected = Array2::<f64>::zeros([4, 4]);
        for i in 0..4 {
            let row: Vec<f64> = (0..4).map(|j| input[[i, j]]).collect();
            let transformed = plan.forward(&row).unwrap();
            for j in 0..4 {
                row_stage[[i, j]] = transformed[j];
            }
        }
        for j in 0..4 {
            let col: Vec<f64> = (0..4).map(|i| row_stage[[i, j]]).collect();
            let transformed = plan.forward(&col).unwrap();
            for i in 0..4 {
                expected[[i, j]] = transformed[i];
            }
        }

        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn inverse_2d_roundtrip_recovers_signal() {
        let plan = DctDstPlan::new(4, RealTransformKind::DstII).unwrap();
        let input = Array2::from_shape_vec(
            [4, 4],
            vec![
                0.25, -1.0, 3.0, 2.0, //
                1.0, 0.0, -2.5, 4.0, //
                -3.0, 2.0, 1.5, -0.5, //
                2.25, -4.0, 0.75, 1.0,
            ],
        )
        .unwrap();
        let coeff = plan.forward_2d(&input).unwrap();
        let recovered = plan.inverse_2d(&coeff).unwrap();
        for (lhs, rhs) in recovered.iter().zip(input.iter()) {
            assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn inverse_3d_roundtrip_recovers_signal() {
        let plan = DctDstPlan::new(3, RealTransformKind::DctIII).unwrap();
        let input = Array3::from_shape_fn([3, 3, 3], |[i, j, k]| {
            (0.7 * i as f64).sin() + 0.5 * (1.3 * j as f64).cos() - 0.25 * k as f64
        });
        let coeff = plan.forward_3d(&input).unwrap();
        let recovered = plan.inverse_3d(&coeff).unwrap();
        for (lhs, rhs) in recovered.iter().zip(input.iter()) {
            assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn rejects_non_square_or_non_cubic_shapes() {
        let plan2 = DctDstPlan::new(4, RealTransformKind::DctII).unwrap();
        let nonsquare = Array2::<f64>::zeros([4, 3]);
        assert!(matches!(
            plan2.forward_2d(&nonsquare),
            Err(DctDstError::LengthMismatch)
        ));

        let plan3 = DctDstPlan::new(3, RealTransformKind::DstIII).unwrap();
        let noncubic = Array3::<f64>::zeros([3, 3, 2]);
        assert!(matches!(
            plan3.inverse_3d(&noncubic),
            Err(DctDstError::LengthMismatch)
        ));
    }

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

    // -------------------------------------------------------------------------
    // Property tests
    // -------------------------------------------------------------------------

    proptest! {
        /// Property: (2/N) * DCT-III(DCT-II(x)) = x, L-inf err < 1e-9, for n in [2,32].
        #[test]
        fn dct2_dct3_inverse_pair(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 2..33usize),
        ) {
            let n = signal.len();
            let plan_forward = DctDstPlan::new(n, RealTransformKind::DctII).unwrap();
            let plan_inverse = DctDstPlan::new(n, RealTransformKind::DctIII).unwrap();
            let forward = plan_forward.forward(&signal).unwrap();
            let recovered_raw = plan_inverse.forward(&forward).unwrap();
            let scale = 2.0 / n as f64;
            let recovered: Vec<f64> = recovered_raw.into_iter().map(|v| v * scale).collect();
            let err: f64 = signal
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DCT-II/III inverse pair failed: err={}", err);
        }

        /// Property: (2/N) * DST-III(DST-II(x)) = x, L-inf err < 1e-9, for n in [2,32].
        #[test]
        fn dst2_dst3_inverse_pair(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 2..33usize),
        ) {
            let n = signal.len();
            let plan_forward = DctDstPlan::new(n, RealTransformKind::DstII).unwrap();
            let plan_inverse = DctDstPlan::new(n, RealTransformKind::DstIII).unwrap();
            let forward = plan_forward.forward(&signal).unwrap();
            let recovered_raw = plan_inverse.forward(&forward).unwrap();
            let scale = 2.0 / n as f64;
            let recovered: Vec<f64> = recovered_raw.into_iter().map(|v| v * scale).collect();
            let err: f64 = signal
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DST-II/III inverse pair failed: err={}", err);
        }

        /// Property: DCT-I(DCT-I(x)) = 2(N−1)·x, L-inf err < 1e-9, for n in [2,32].
        #[test]
        fn dct1_self_inverse_property(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 2..33usize),
        ) {
            let n = signal.len();
            let mut first = vec![0.0_f64; n];
            let mut second = vec![0.0_f64; n];
            dct1(&signal, &mut first);
            dct1(&first, &mut second);
            let scale = 2.0 * (n - 1) as f64;
            let err: f64 = signal
                .iter()
                .zip(second.iter())
                .map(|(x, y)| (y - x * scale).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DCT-I self-inverse failed: err={}", err);
        }

        /// Property: plan.inverse(plan.forward(x)) = x for DctI, L-inf err < 1e-9, n in [2,32].
        #[test]
        fn plan_dct1_roundtrip(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 2..33usize),
        ) {
            let n = signal.len();
            let plan = DctDstPlan::new(n, RealTransformKind::DctI).unwrap();
            let forward = plan.forward(&signal).unwrap();
            let recovered = plan.inverse(&forward).unwrap();
            let err: f64 = signal
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DctI plan roundtrip failed: err={}", err);
        }

        /// Property: DCT-IV(DCT-IV(x)) = (N/2)·x, L-inf err < 1e-9, for n in [1,32].
        #[test]
        fn dct4_self_inverse_property(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
        ) {
            let n = signal.len();
            let mut first = vec![0.0_f64; n];
            let mut second = vec![0.0_f64; n];
            dct4(&signal, &mut first);
            dct4(&first, &mut second);
            let scale = n as f64 / 2.0;
            let err: f64 = signal
                .iter()
                .zip(second.iter())
                .map(|(x, y)| (y - x * scale).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DCT-IV self-inverse failed: err={}", err);
        }

        /// Property: plan.inverse(plan.forward(x)) = x for DctIV, L-inf err < 1e-9, n in [1,32].
        #[test]
        fn plan_dct4_roundtrip(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
        ) {
            let n = signal.len();
            let plan = DctDstPlan::new(n, RealTransformKind::DctIV).unwrap();
            let forward = plan.forward(&signal).unwrap();
            let recovered = plan.inverse(&forward).unwrap();
            let err: f64 = signal
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DctIV plan roundtrip failed: err={}", err);
        }

        /// Property: DST-I(DST-I(x)) = 2(N+1)·x, L-inf err < 1e-9, for n in [1,32].
        #[test]
        fn dst1_self_inverse_property(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
        ) {
            let n = signal.len();
            let mut first = vec![0.0_f64; n];
            let mut second = vec![0.0_f64; n];
            dst1(&signal, &mut first);
            dst1(&first, &mut second);
            let scale = 2.0 * (n + 1) as f64;
            let err: f64 = signal
                .iter()
                .zip(second.iter())
                .map(|(x, y)| (y - x * scale).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DST-I self-inverse failed: err={}", err);
        }

        /// Property: plan.inverse(plan.forward(x)) = x for DstI, L-inf err < 1e-9, n in [1,32].
        #[test]
        fn plan_dst1_roundtrip(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
        ) {
            let n = signal.len();
            let plan = DctDstPlan::new(n, RealTransformKind::DstI).unwrap();
            let forward = plan.forward(&signal).unwrap();
            let recovered = plan.inverse(&forward).unwrap();
            let err: f64 = signal
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DstI plan roundtrip failed: err={}", err);
        }

        /// Property: DST-IV(DST-IV(x)) = (N/2)·x, L-inf err < 1e-9, for n in [1,32].
        #[test]
        fn dst4_self_inverse_property(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
        ) {
            let n = signal.len();
            let mut first = vec![0.0_f64; n];
            let mut second = vec![0.0_f64; n];
            dst4(&signal, &mut first);
            dst4(&first, &mut second);
            let scale = n as f64 / 2.0;
            let err: f64 = signal
                .iter()
                .zip(second.iter())
                .map(|(x, y)| (y - x * scale).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DST-IV self-inverse failed: err={}", err);
        }

        /// Property: plan.inverse(plan.forward(x)) = x for DstIV, L-inf err < 1e-9, n in [1,32].
        #[test]
        fn plan_dst4_roundtrip(
            signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
        ) {
            let n = signal.len();
            let plan = DctDstPlan::new(n, RealTransformKind::DstIV).unwrap();
            let forward = plan.forward(&signal).unwrap();
            let recovered = plan.inverse(&forward).unwrap();
            let err: f64 = signal
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-9, "DstIV plan roundtrip failed: err={}", err);
        }
    }
}
