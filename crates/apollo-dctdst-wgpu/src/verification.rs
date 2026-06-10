//! WGPU value-semantic verification.

#![allow(clippy::manual_memcpy)]

#[cfg(test)]
mod tests {
    use apollo_dctdst::{DctDstPlan, RealTransformKind};
    use leto::{SliceArg, Storage};

    use crate::{DctDstWgpuBackend, DctDstWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_full_kernel_surface() {
        let capabilities = WgpuCapabilities::full(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
        assert!(capabilities.supports_dct);
        assert!(capabilities.supports_dst);
        assert!(capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = DctDstWgpuPlan::new(64, RealTransformKind::DctII);
        assert_eq!(plan.len(), 64);
        assert_eq!(plan.kind(), RealTransformKind::DctII);
        assert!(!DctDstWgpuPlan::new(64, RealTransformKind::DctIII).is_empty());
        assert!(DctDstWgpuPlan::new(0, RealTransformKind::DctII).is_empty());
    }

    #[test]
    fn unsupported_execution_error_identifies_operation() {
        let err = WgpuError::UnsupportedExecution {
            operation: "forward",
        };
        assert_eq!(
            err.to_string(),
            "forward is unsupported by the current WGPU capability set"
        );
    }

    #[test]
    fn backend_reports_dct_and_dst_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
        assert!(capabilities.supports_dct);
        assert!(capabilities.supports_dst);
    }

    #[test]
    fn dct2_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DctII);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = DctDstPlan::new(input.len(), RealTransformKind::DctII).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dct3_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.75_f32, -1.0, 2.5, -0.25, 3.0, 0.5];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DctIII);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = DctDstPlan::new(input.len(), RealTransformKind::DctIII).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dct2_inverse_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.25_f32, -1.25, 2.0, -0.5, 3.0, 1.5];
        let plan = backend.plan(input.len(), RealTransformKind::DctII);
        let spectrum = backend
            .execute_forward(&plan, &input)
            .expect("wgpu forward execution");
        let recovered = backend
            .execute_inverse(&plan, &spectrum)
            .expect("wgpu inverse execution");

        assert_eq!(recovered.len(), input.len());
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dst2_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -0.5, 2.0, -1.5, 0.25, 3.0];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DstII);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = DctDstPlan::new(input.len(), RealTransformKind::DstII).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dst3_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.25_f32, -1.0, 2.0, 0.5, -0.75, 1.25];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DstIII);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu forward execution");

        let cpu_plan = DctDstPlan::new(input.len(), RealTransformKind::DstIII).expect("cpu plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .expect("cpu forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dst2_inverse_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.75_f32, -1.25, 0.5, 2.0, -0.5, 1.0];
        let plan = backend.plan(input.len(), RealTransformKind::DstII);
        let spectrum = backend
            .execute_forward(&plan, &input)
            .expect("wgpu forward execution");
        let recovered = backend
            .execute_inverse(&plan, &spectrum)
            .expect("wgpu inverse execution");

        assert_eq!(recovered.len(), input.len());
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn typed_mixed_storage_dct2_matches_represented_f32_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use apollo_fft::{f16, PrecisionProfile};

        let represented = [
            0.75_f32, -1.25_f32, 2.0_f32, -0.5_f32, 3.0_f32, 1.5_f32, 0.25_f32, -0.875_f32,
        ];
        let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
        let represented_input: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
        let plan = backend.plan(input.len(), RealTransformKind::DctII);
        let expected_forward = backend
            .execute_forward(&plan, &represented_input)
            .expect("represented f32 forward");
        let mut typed_output = vec![f16::from_f32(0.0); input.len()];
        backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &input,
                &mut typed_output,
            )
            .expect("typed mixed forward");
        for (actual, expected) in typed_output.iter().zip(expected_forward.iter()) {
            let expected_quantized = f16::from_f32(*expected);
            assert_eq!(
                actual.to_bits(),
                expected_quantized.to_bits(),
                "f16 forward bit mismatch: actual={}, expected={}",
                actual.to_f32(),
                expected_quantized.to_f32()
            );
        }
    }

    #[test]
    fn leto_forward_and_inverse_match_slice_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.25_f32, -1.25, 2.0, -0.5, 3.0, 1.5];
        let plan = backend.plan(input.len(), RealTransformKind::DctII);
        let expected_forward = backend.execute_forward(&plan, &input).expect("forward");
        let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("leto input");
        let actual_forward = backend
            .execute_forward_leto(&plan, leto_input.view())
            .expect("leto forward");
        assert_eq!(
            actual_forward.storage().as_slice(),
            expected_forward.as_slice()
        );

        let expected_inverse = backend
            .execute_inverse(&plan, &expected_forward)
            .expect("inverse");
        let leto_spectrum =
            leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                .expect("leto spectrum");
        let actual_inverse = backend
            .execute_inverse_leto(&plan, leto_spectrum.view())
            .expect("leto inverse");
        assert_eq!(
            actual_inverse.storage().as_slice(),
            expected_inverse.as_slice()
        );
    }

    #[test]
    fn leto_strided_forward_matches_logical_slice_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let logical = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5];
        let mut backing = Vec::with_capacity(logical.len() * 2);
        for value in logical.iter().copied() {
            backing.push(value);
            backing.push(99.0);
        }
        let plan = backend.plan(logical.len(), RealTransformKind::DctII);
        let expected = backend
            .execute_forward(&plan, &logical)
            .expect("slice forward");
        let leto_input = leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
        let strided = leto_input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");
        let actual = backend
            .execute_forward_leto(&plan, strided)
            .expect("strided leto forward");
        assert_eq!(actual.storage().as_slice(), expected.as_slice());
    }

    #[test]
    fn typed_leto_forward_and_inverse_match_typed_slice_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use apollo_fft::{f16, PrecisionProfile};

        let represented = [0.75_f32, -1.25_f32, 2.0_f32, -0.5_f32, 3.0_f32, 1.5_f32];
        let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
        let plan = backend.plan(input.len(), RealTransformKind::DctII);
        let mut expected_forward = vec![f16::from_f32(0.0); input.len()];
        backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &input,
                &mut expected_forward,
            )
            .expect("typed forward");
        let leto_input =
            leto::Array1::from_shape_vec([input.len()], input).expect("leto typed input");
        let actual_forward = backend
            .execute_forward_leto_typed::<f16>(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                leto_input.view(),
            )
            .expect("leto typed forward");
        assert_eq!(
            actual_forward.storage().as_slice(),
            expected_forward.as_slice()
        );

        let mut expected_inverse = vec![f16::from_f32(0.0); expected_forward.len()];
        backend
            .execute_inverse_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &expected_forward,
                &mut expected_inverse,
            )
            .expect("typed inverse");
        let leto_spectrum =
            leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                .expect("leto typed spectrum");
        let actual_inverse = backend
            .execute_inverse_leto_typed::<f16>(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                leto_spectrum.view(),
            )
            .expect("leto typed inverse");
        assert_eq!(
            actual_inverse.storage().as_slice(),
            expected_inverse.as_slice()
        );
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use apollo_fft::PrecisionProfile;

        let plan = backend.plan(4, RealTransformKind::DctII);
        let input = [1.0_f32, -1.0_f32, 0.5_f32, -0.5_f32];
        let mut output = [0.0_f32; 4];
        let error = backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::HIGH_ACCURACY_F64,
                &input,
                &mut output,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(error, WgpuError::InvalidPrecisionProfile);
    }

    #[test]
    fn rejects_invalid_lengths() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let empty_err = backend
            .execute_forward(&DctDstWgpuPlan::new(0, RealTransformKind::DctII), &[])
            .expect_err("empty plan must fail");
        assert!(matches!(empty_err, WgpuError::InvalidPlan { .. }));

        let mismatch_err = backend
            .execute_forward(&DctDstWgpuPlan::new(8, RealTransformKind::DctII), &[0.0; 4])
            .expect_err("length mismatch must fail");
        assert_eq!(
            mismatch_err,
            WgpuError::LengthMismatch {
                expected: 8,
                actual: 4,
            }
        );
    }

    #[test]
    fn dct1_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -0.5, 2.0, 0.75];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DctI);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu dct1 forward execution");

        let cpu_plan =
            DctDstPlan::new(input.len(), RealTransformKind::DctI).expect("cpu dct1 plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&v| v as f64).collect::<Vec<_>>())
            .expect("cpu dct1 forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dct4_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.5_f32, -1.0, 2.5, -0.25, 1.5, 0.75];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DctIV);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu dct4 forward execution");

        let cpu_plan =
            DctDstPlan::new(input.len(), RealTransformKind::DctIV).expect("cpu dct4 plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&v| v as f64).collect::<Vec<_>>())
            .expect("cpu dct4 forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dst1_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -2.0, 0.5, 3.0, -1.5];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DstI);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu dst1 forward execution");

        let cpu_plan =
            DctDstPlan::new(input.len(), RealTransformKind::DstI).expect("cpu dst1 plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&v| v as f64).collect::<Vec<_>>())
            .expect("cpu dst1 forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dst4_forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.75_f32, -1.25, 2.0, -0.5, 1.0];
        let gpu_plan = backend.plan(input.len(), RealTransformKind::DstIV);
        let gpu = backend
            .execute_forward(&gpu_plan, &input)
            .expect("wgpu dst4 forward execution");

        let cpu_plan =
            DctDstPlan::new(input.len(), RealTransformKind::DstIV).expect("cpu dst4 plan");
        let cpu = cpu_plan
            .forward(&input.iter().map(|&v| v as f64).collect::<Vec<_>>())
            .expect("cpu dst4 forward");

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dct1_inverse_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -0.5, 2.0, 0.75];
        let plan = backend.plan(input.len(), RealTransformKind::DctI);
        let spectrum = backend
            .execute_forward(&plan, &input)
            .expect("wgpu dct1 forward");
        let recovered = backend
            .execute_inverse(&plan, &spectrum)
            .expect("wgpu dct1 inverse");

        assert_eq!(recovered.len(), input.len());
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dct4_inverse_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.5_f32, -1.0, 2.5, -0.25, 1.5, 0.75];
        let plan = backend.plan(input.len(), RealTransformKind::DctIV);
        let spectrum = backend
            .execute_forward(&plan, &input)
            .expect("wgpu dct4 forward");
        let recovered = backend
            .execute_inverse(&plan, &spectrum)
            .expect("wgpu dct4 inverse");

        assert_eq!(recovered.len(), input.len());
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dst1_inverse_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![1.0_f32, -2.0, 0.5, 3.0, -1.5];
        let plan = backend.plan(input.len(), RealTransformKind::DstI);
        let spectrum = backend
            .execute_forward(&plan, &input)
            .expect("wgpu dst1 forward");
        let recovered = backend
            .execute_inverse(&plan, &spectrum)
            .expect("wgpu dst1 inverse");

        assert_eq!(recovered.len(), input.len());
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dst4_inverse_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let input = vec![0.75_f32, -1.25, 2.0, -0.5, 1.0];
        let plan = backend.plan(input.len(), RealTransformKind::DstIV);
        let spectrum = backend
            .execute_forward(&plan, &input)
            .expect("wgpu dst4 forward");
        let recovered = backend
            .execute_inverse(&plan, &spectrum)
            .expect("wgpu dst4 inverse");

        assert_eq!(recovered.len(), input.len());
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn dct1_rejects_length_less_than_two() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        let plan = DctDstWgpuPlan::new(1, RealTransformKind::DctI);
        let err = backend
            .execute_forward(&plan, &[0.5_f32])
            .expect_err("DCT-I length 1 must fail");
        assert!(matches!(err, WgpuError::InvalidPlan { .. }));
    }

    #[test]
    fn dct2_forward_2d_matches_cpu_parity_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use ndarray::array;
        // 3×3 analytically-separable: each row and column independently verified by
        // the 1D DCT-II; the 2D separable result equals applying 1D twice.
        let input = array![[1.0_f32, -2.0, 0.5], [0.25, 3.0, -1.5], [-0.75, 0.5, 2.0]];
        let plan = DctDstWgpuPlan::new(3, RealTransformKind::DctII);
        let gpu_2d = backend
            .execute_forward_2d(&plan, &input)
            .expect("gpu 2d forward");

        // CPU reference: apply 1D forward row-then-column using apollo-dctdst.
        let cpu_plan = DctDstPlan::new(3, RealTransformKind::DctII).expect("cpu plan");
        let mut tmp = [[0.0_f64; 3]; 3];
        for r in 0..3 {
            let row: Vec<f64> = (0..3).map(|c| f64::from(input[[r, c]])).collect();
            let out = cpu_plan.forward(&row).expect("cpu row forward");
            for c in 0..3 {
                tmp[r][c] = out[c];
            }
        }
        let mut cpu_2d = [[0.0_f64; 3]; 3];
        for c in 0..3 {
            let col: Vec<f64> = (0..3).map(|r| tmp[r][c]).collect();
            let out = cpu_plan.forward(&col).expect("cpu col forward");
            for r in 0..3 {
                cpu_2d[r][c] = out[r];
            }
        }
        assert_eq!(gpu_2d.dim(), (3, 3));
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (f64::from(gpu_2d[[r, c]]) - cpu_2d[r][c]).abs() < 1.0e-3,
                    "mismatch at [{r},{c}]: gpu={}, cpu={}",
                    gpu_2d[[r, c]],
                    cpu_2d[r][c]
                );
            }
        }
    }

    #[test]
    fn dct2_inverse_2d_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use ndarray::array;
        let input = array![[1.0_f32, -2.0, 0.5], [0.25, 3.0, -1.5], [-0.75, 0.5, 2.0]];
        let plan = DctDstWgpuPlan::new(3, RealTransformKind::DctII);
        let spectrum = backend
            .execute_forward_2d(&plan, &input)
            .expect("gpu 2d forward");
        let recovered = backend
            .execute_inverse_2d(&plan, &spectrum)
            .expect("gpu 2d inverse");

        assert_eq!(recovered.dim(), (3, 3));
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (recovered[[r, c]] - input[[r, c]]).abs() < 1.0e-3,
                    "roundtrip mismatch at [{r},{c}]: recovered={}, original={}",
                    recovered[[r, c]],
                    input[[r, c]]
                );
            }
        }
    }

    #[test]
    fn leto_2d_forward_and_inverse_match_ndarray_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use ndarray::array;
        let input = array![[1.0_f32, -2.0, 0.5], [0.25, 3.0, -1.5], [-0.75, 0.5, 2.0]];
        let plan = DctDstWgpuPlan::new(3, RealTransformKind::DctII);
        let expected_forward = backend
            .execute_forward_2d(&plan, &input)
            .expect("2d forward");
        let leto_input = leto::Array2::from_shape_vec([3, 3], input.iter().copied().collect())
            .expect("leto 2d input");
        let actual_forward = backend
            .execute_forward_2d_leto(&plan, leto_input.view())
            .expect("leto 2d forward");
        assert_eq!(
            actual_forward.storage().as_slice(),
            expected_forward
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .as_slice()
        );

        let expected_inverse = backend
            .execute_inverse_2d(&plan, &expected_forward)
            .expect("2d inverse");
        let leto_spectrum = leto::Array2::from_shape_vec(
            [3, 3],
            expected_forward.iter().copied().collect::<Vec<_>>(),
        )
        .expect("leto 2d spectrum");
        let actual_inverse = backend
            .execute_inverse_2d_leto(&plan, leto_spectrum.view())
            .expect("leto 2d inverse");
        assert_eq!(
            actual_inverse.storage().as_slice(),
            expected_inverse
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    #[test]
    fn dct2_forward_3d_matches_cpu_parity_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use ndarray::Array3;
        // 3×3×3 analytically-separable: same separable kernel applied along each axis.
        let mut input = Array3::<f32>::zeros((3, 3, 3));
        // Fill with a known non-trivial signal.
        let flat: [f32; 27] = [
            1.0, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0, 0.1, -0.3, 1.2, -0.5, 2.1, -1.1, 0.7,
            -0.9, 0.3, 1.5, -0.2, 0.8, -1.4, 0.6, -0.1, 0.9, -2.5, 1.3,
        ];
        let mut idx = 0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    input[[i, j, k]] = flat[idx];
                    idx += 1;
                }
            }
        }
        let plan = DctDstWgpuPlan::new(3, RealTransformKind::DctII);
        let gpu_3d = backend
            .execute_forward_3d(&plan, &input)
            .expect("gpu 3d forward");

        // CPU reference: separable axis-0, axis-1, axis-2 using apollo-dctdst 1D.
        let cpu_plan = DctDstPlan::new(3, RealTransformKind::DctII).expect("cpu plan");
        let mut tmp0 = [[[0.0_f64; 3]; 3]; 3];
        for j in 0..3 {
            for k in 0..3 {
                let fiber: Vec<f64> = (0..3).map(|i| f64::from(input[[i, j, k]])).collect();
                let out = cpu_plan.forward(&fiber).expect("cpu axis0");
                for i in 0..3 {
                    tmp0[i][j][k] = out[i];
                }
            }
        }
        let mut tmp1 = [[[0.0_f64; 3]; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                let fiber: Vec<f64> = (0..3).map(|j| tmp0[i][j][k]).collect();
                let out = cpu_plan.forward(&fiber).expect("cpu axis1");
                for j in 0..3 {
                    tmp1[i][j][k] = out[j];
                }
            }
        }
        let mut cpu_3d = [[[0.0_f64; 3]; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let fiber: Vec<f64> = (0..3).map(|k| tmp1[i][j][k]).collect();
                let out = cpu_plan.forward(&fiber).expect("cpu axis2");
                for k in 0..3 {
                    cpu_3d[i][j][k] = out[k];
                }
            }
        }
        assert_eq!(gpu_3d.dim(), (3, 3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    assert!(
                        (f64::from(gpu_3d[[i, j, k]]) - cpu_3d[i][j][k]).abs() < 1.0e-3,
                        "mismatch at [{i},{j},{k}]: gpu={}, cpu={}",
                        gpu_3d[[i, j, k]],
                        cpu_3d[i][j][k]
                    );
                }
            }
        }
    }

    #[test]
    fn dct2_inverse_3d_recovers_input_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use ndarray::Array3;
        let mut input = Array3::<f32>::zeros((3, 3, 3));
        let flat: [f32; 27] = [
            1.0, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0, 0.1, -0.3, 1.2, -0.5, 2.1, -1.1, 0.7,
            -0.9, 0.3, 1.5, -0.2, 0.8, -1.4, 0.6, -0.1, 0.9, -2.5, 1.3,
        ];
        let mut idx = 0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    input[[i, j, k]] = flat[idx];
                    idx += 1;
                }
            }
        }
        let plan = DctDstWgpuPlan::new(3, RealTransformKind::DctII);
        let spectrum = backend
            .execute_forward_3d(&plan, &input)
            .expect("gpu 3d forward");
        let recovered = backend
            .execute_inverse_3d(&plan, &spectrum)
            .expect("gpu 3d inverse");

        assert_eq!(recovered.dim(), (3, 3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    assert!(
                        (recovered[[i, j, k]] - input[[i, j, k]]).abs() < 1.0e-3,
                        "roundtrip mismatch at [{i},{j},{k}]: recovered={}, original={}",
                        recovered[[i, j, k]],
                        input[[i, j, k]]
                    );
                }
            }
        }
    }

    #[test]
    fn leto_3d_forward_and_inverse_match_ndarray_when_device_exists() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use ndarray::Array3;
        let flat: [f32; 27] = [
            1.0, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0, 0.1, -0.3, 1.2, -0.5, 2.1, -1.1, 0.7,
            -0.9, 0.3, 1.5, -0.2, 0.8, -1.4, 0.6, -0.1, 0.9, -2.5, 1.3,
        ];
        let input = Array3::from_shape_vec((3, 3, 3), flat.to_vec()).expect("ndarray 3d input");
        let plan = DctDstWgpuPlan::new(3, RealTransformKind::DctII);
        let expected_forward = backend
            .execute_forward_3d(&plan, &input)
            .expect("3d forward");
        let leto_input =
            leto::Array3::from_shape_vec([3, 3, 3], flat.to_vec()).expect("leto 3d input");
        let actual_forward = backend
            .execute_forward_3d_leto(&plan, leto_input.view())
            .expect("leto 3d forward");
        assert_eq!(
            actual_forward.storage().as_slice(),
            expected_forward
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .as_slice()
        );

        let expected_inverse = backend
            .execute_inverse_3d(&plan, &expected_forward)
            .expect("3d inverse");
        let leto_spectrum = leto::Array3::from_shape_vec(
            [3, 3, 3],
            expected_forward.iter().copied().collect::<Vec<_>>(),
        )
        .expect("leto 3d spectrum");
        let actual_inverse = backend
            .execute_inverse_3d_leto(&plan, leto_spectrum.view())
            .expect("leto 3d inverse");
        assert_eq!(
            actual_inverse.storage().as_slice(),
            expected_inverse
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    #[test]
    fn execute_forward_2d_rejects_non_square_input() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use ndarray::Array2;
        let plan = DctDstWgpuPlan::new(3, RealTransformKind::DctII);
        let input = Array2::<f32>::zeros((2, 3));
        let err = backend
            .execute_forward_2d(&plan, &input)
            .expect_err("non-square 2D must fail");
        assert!(matches!(err, WgpuError::ShapeMismatch { .. }));
    }

    #[test]
    fn execute_forward_3d_rejects_non_cubic_input() {
        let Ok(backend) = DctDstWgpuBackend::try_default() else {
            return;
        };
        use ndarray::Array3;
        let plan = DctDstWgpuPlan::new(3, RealTransformKind::DctII);
        let input = Array3::<f32>::zeros((2, 3, 3));
        let err = backend
            .execute_forward_3d(&plan, &input)
            .expect_err("non-cubic 3D must fail");
        assert!(matches!(err, WgpuError::ShapeMismatch { .. }));
    }
}
