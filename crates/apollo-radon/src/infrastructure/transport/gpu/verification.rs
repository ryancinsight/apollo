//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_fft::{f16, PrecisionProfile};
    use crate::RadonPlan;
    use leto::{SliceArg, Storage};
    use ndarray::{array, Array2};

    use crate::infrastructure::transport::gpu::{RadonWgpuBackend, RadonWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_forward_only_kernel_surface() {
        let capabilities = WgpuCapabilities::forward_only(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(!capabilities.supports_inverse);
        assert!(capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn capabilities_reflect_forward_and_inverse_surface() {
        let caps = WgpuCapabilities::forward_and_inverse(true);
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
        assert!(caps.supports_mixed_precision);
        assert_eq!(
            caps.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
        let caps_off = WgpuCapabilities::forward_and_inverse(false);
        assert!(!caps_off.device_available);
        assert!(!caps_off.supports_forward);
        assert!(!caps_off.supports_inverse);
    }

    #[test]
    fn plan_preserves_geometry_configuration() {
        let plan = RadonWgpuPlan::new(8, 9, 3, 11, 0.5_f64.to_bits());
        assert_eq!(plan.rows(), 8);
        assert_eq!(plan.cols(), 9);
        assert_eq!(plan.angle_count(), 3);
        assert_eq!(plan.detector_count(), 11);
        assert_eq!(plan.detector_spacing(), 0.5);
        assert!(!plan.is_empty());
        assert!(RadonWgpuPlan::new(0, 9, 3, 11, 0.5_f64.to_bits()).is_empty());
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
    fn capabilities_include_filtered_backprojection() {
        let caps = WgpuCapabilities::forward_inverse_and_fbp(true);
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
        assert!(caps.supports_filtered_backprojection);
        assert!(caps.supports_mixed_precision);
        assert_eq!(
            caps.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
        let caps_off = WgpuCapabilities::forward_inverse_and_fbp(false);
        assert!(!caps_off.device_available);
        assert!(!caps_off.supports_forward);
        assert!(!caps_off.supports_inverse);
        assert!(!caps_off.supports_filtered_backprojection);
    }

    #[test]
    fn radon_wgpu_execution_suite_when_device_exists() {
        let Ok(backend) = RadonWgpuBackend::try_default() else {
            return;
        };

        // 1. backend_reports_forward_and_backproject
        {
            let capabilities = backend.capabilities();
            assert!(capabilities.device_available);
            assert!(capabilities.supports_forward);
            assert!(capabilities.supports_inverse);
        }

        // 2. forward_projection_matches_cpu_reference
        {
            let image = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
            let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
            let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
            let gpu = backend
                .execute_forward(&plan, &image, &angles)
                .expect("wgpu forward execution");

            let cpu_plan = RadonPlan::new(
                3,
                3,
                angles.iter().map(|&angle| f64::from(angle)).collect(),
                5,
                1.0,
            )
            .expect("cpu plan");
            let cpu = cpu_plan
                .forward(&image.mapv(f64::from))
                .expect("cpu forward");

            assert_eq!(gpu.dim(), cpu.values().dim());
            for (index, (actual, expected)) in gpu.iter().zip(cpu.values().iter()).enumerate() {
                let error = (f64::from(*actual) - *expected).abs();
                assert!(
                    error < 5.0e-4,
                    "mismatch at linear index {index}: actual={}, expected={}, error={error}",
                    actual,
                    expected
                );
            }
        }

        // 3. backproject_matches_cpu_reference
        {
            let image_f64 = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
            let angles_f64: Vec<f64> = vec![
                0.0,
                std::f64::consts::FRAC_PI_4,
                std::f64::consts::FRAC_PI_2,
            ];
            let angles_f32: Vec<f32> = angles_f64.iter().map(|&a| a as f32).collect();
            // CPU forward then backproject (f64 reference path).
            let cpu_plan = RadonPlan::new(3, 3, angles_f64.clone(), 7, 1.0).expect("cpu plan");
            let sinogram_cpu = cpu_plan.forward(&image_f64).expect("cpu forward");
            let cpu_bp = cpu_plan
                .backproject(&sinogram_cpu)
                .expect("cpu backproject");
            // GPU backproject using f32 sinogram derived from the f64 CPU result.
            let sinogram_f32 = sinogram_cpu.values().mapv(|v| v as f32);
            let gpu_plan = backend.plan(3, 3, angles_f32.len(), 7, 1.0);
            let gpu_bp = backend
                .execute_inverse(&gpu_plan, &sinogram_f32, &angles_f32)
                .expect("gpu backproject");
            assert_eq!(gpu_bp.dim(), (3, 3));
            for ((r, c), gpu_val) in gpu_bp.indexed_iter() {
                let cpu_val = cpu_bp[(r, c)] as f32;
                let err = (gpu_val - cpu_val).abs();
                assert!(
                    err < 5e-3,
                    "mismatch at ({r},{c}): gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
                );
            }
        }

        // 4. leto_forward_inverse_and_fbp_match_ndarray
        {
            let image = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
            let angles = vec![
                0.0_f32,
                std::f32::consts::FRAC_PI_4,
                std::f32::consts::FRAC_PI_2,
            ];
            let plan = backend.plan(3, 3, angles.len(), 7, 1.0);
            let image_leto = leto::Array::from_mnemosyne_slice(
                [3, 3],
                &image.iter().copied().collect::<Vec<_>>(),
            )
            .expect("leto image");
            let angles_leto =
                leto::Array::from_mnemosyne_slice([angles.len()], &angles).expect("leto angles");

            let expected_forward = backend
                .execute_forward(&plan, &image, &angles)
                .expect("ndarray forward");
            let actual_forward = backend
                .execute_forward_leto(&plan, image_leto.view(), angles_leto.view())
                .expect("leto forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward.as_slice().expect("contiguous forward")
            );

            let expected_inverse = backend
                .execute_inverse(&plan, &expected_forward, &angles)
                .expect("ndarray inverse");
            let actual_inverse = backend
                .execute_inverse_leto(&plan, actual_forward.view(), angles_leto.view())
                .expect("leto inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice().expect("contiguous inverse")
            );

            let expected_fbp = backend
                .execute_filtered_backproject(&plan, &expected_forward, &angles)
                .expect("ndarray fbp");
            let actual_fbp = backend
                .execute_filtered_backproject_leto(&plan, actual_forward.view(), angles_leto.view())
                .expect("leto fbp");
            assert_eq!(
                actual_fbp.storage().as_slice(),
                expected_fbp.as_slice().expect("contiguous fbp")
            );
        }

        // 5. leto_strided_forward_matches_logical_ndarray
        {
            let image = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
            let mut interleaved = Vec::with_capacity(3 * 6);
            for row in image.rows() {
                for value in row.iter().copied() {
                    interleaved.push(value);
                    interleaved.push(99.0);
                }
            }
            let image_leto =
                leto::Array::from_mnemosyne_slice([3, 6], &interleaved).expect("leto image");
            let strided = image_leto
                .slice_with::<2>(&[
                    SliceArg::range(Some(0), None, 1),
                    SliceArg::range(Some(0), None, 2),
                ])
                .expect("strided image");
            let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
            let angles_leto =
                leto::Array::from_mnemosyne_slice([angles.len()], &angles).expect("leto angles");
            let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
            let expected = backend
                .execute_forward(&plan, &image, &angles)
                .expect("ndarray forward");
            let actual = backend
                .execute_forward_leto(&plan, strided, angles_leto.view())
                .expect("strided leto forward");
            assert_eq!(
                actual.storage().as_slice(),
                expected.as_slice().expect("contiguous forward")
            );
        }

        // 6. execute_inverse_rejects_sinogram_shape_mismatch
        {
            let plan = backend.plan(3, 3, 2, 5, 1.0);
            let wrong_sinogram = Array2::<f32>::zeros((3, 5));
            let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
            let err = backend
                .execute_inverse(&plan, &wrong_sinogram, &angles)
                .expect_err("sinogram shape mismatch must fail");
            assert!(matches!(err, WgpuError::ShapeMismatch { .. }));
        }

        // 7. typed_flat_mixed_storage_matches_represented_f32_execution
        {
            let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
            let plan = backend.plan(3, 3, angles.len(), 5, 1.0);

            let flat_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

            let flat_f16: Vec<f16> = flat_f32.iter().copied().map(f16::from_f32).collect();
            let represented_f32: Vec<f32> = flat_f16.iter().map(|v| v.to_f32()).collect();
            let image_represented =
                Array2::from_shape_vec((3, 3), represented_f32).expect("reshape");

            let expected = backend
                .execute_forward(&plan, &image_represented, &angles)
                .expect("represented f32 forward");
            let actual = backend
                .execute_forward_flat_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &flat_f16,
                    &angles,
                )
                .expect("typed flat mixed forward");

            assert_eq!(actual.dim(), expected.dim());
            for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (f64::from(*a) - f64::from(*e)).abs() < 0.1,
                    "mismatch at index {index}: actual={a}, expected={e}"
                );
            }
        }

        // 8. typed_leto_forward_and_inverse_match_typed_flat
        {
            let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
            let angles_leto =
                leto::Array::from_mnemosyne_slice([angles.len()], &angles).expect("leto angles");
            let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
            let flat_f16: Vec<f16> = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                .into_iter()
                .map(f16::from_f32)
                .collect();
            let image_leto =
                leto::Array::from_mnemosyne_slice([3, 3], &flat_f16).expect("typed leto image");

            let expected_forward = backend
                .execute_forward_flat_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &flat_f16,
                    &angles,
                )
                .expect("typed flat forward");
            let actual_forward = backend
                .execute_forward_leto_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    image_leto.view(),
                    angles_leto.view(),
                )
                .expect("typed leto forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward.as_slice().expect("contiguous forward")
            );

            let sinogram_f16: Vec<f16> = expected_forward
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect();
            let sinogram_leto = leto::Array::from_mnemosyne_slice(
                [plan.angle_count(), plan.detector_count()],
                &sinogram_f16,
            )
            .expect("typed leto sinogram");
            let expected_inverse = backend
                .execute_inverse_flat_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &sinogram_f16,
                    &angles,
                )
                .expect("typed flat inverse");
            let actual_inverse = backend
                .execute_inverse_leto_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    sinogram_leto.view(),
                    angles_leto.view(),
                )
                .expect("typed leto inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice().expect("contiguous inverse")
            );
        }

        // 9. typed_flat_path_rejects_profile_mismatch
        {
            let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
            let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
            let flat_f16: Vec<f16> = vec![f16::from_f32(1.0); plan.rows() * plan.cols()];

            let err = backend
                .execute_forward_flat_typed::<f16>(
                    &plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    &flat_f16,
                    &angles,
                )
                .expect_err("profile mismatch must fail");
            assert_eq!(err, WgpuError::InvalidPrecisionProfile);
        }

        // 10. rejects_invalid_plan_and_input_shape_before_dispatch
        {
            let empty_plan_err = backend
                .execute_forward(
                    &RadonWgpuPlan::new(0, 3, 1, 3, 1.0_f64.to_bits()),
                    &array![[1.0_f32]],
                    &[0.0_f32],
                )
                .expect_err("empty plan must fail");
            assert!(matches!(empty_plan_err, WgpuError::InvalidPlan { .. }));

            let shape_err = backend
                .execute_forward(
                    &RadonWgpuPlan::new(3, 3, 1, 3, 1.0_f64.to_bits()),
                    &array![[1.0_f32, 2.0]],
                    &[0.0_f32],
                )
                .expect_err("image shape mismatch must fail");
            assert!(matches!(shape_err, WgpuError::ShapeMismatch { .. }));

            let angle_err = backend
                .execute_forward(
                    &RadonWgpuPlan::new(3, 3, 2, 3, 1.0_f64.to_bits()),
                    &array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    &[0.0_f32],
                )
                .expect_err("angle mismatch must fail");
            assert_eq!(
                angle_err,
                WgpuError::LengthMismatch {
                    expected: 2,
                    actual: 1,
                }
            );
        }

        // 11. backproject_satisfies_adjoint_identity
        {
            let f_f64 = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
            let angles_f64 = vec![
                0.0_f64,
                std::f64::consts::FRAC_PI_4,
                std::f64::consts::FRAC_PI_2,
            ];
            let angles_f32: Vec<f32> = angles_f64.iter().map(|&a| a as f32).collect();

            let cpu_plan = RadonPlan::new(3, 3, angles_f64, 5, 1.0).expect("cpu plan");
            let af = cpu_plan.forward(&f_f64).expect("cpu forward");

            let g = array![
                [1.0_f32, 0.5, -0.5, -1.0, 0.5],
                [0.0, 1.0, 0.0, -1.0, 0.0],
                [0.5, -0.5, 0.5, -0.5, 0.5]
            ];

            let lhs: f64 = af
                .values()
                .iter()
                .zip(g.iter())
                .map(|(a, b)| *a * f64::from(*b))
                .sum();

            let gpu_plan = backend.plan(3, 3, 3, 5, 1.0);
            let adj_g = backend
                .execute_inverse(&gpu_plan, &g, &angles_f32)
                .expect("gpu backproject");

            let rhs: f64 = f_f64
                .iter()
                .zip(adj_g.iter())
                .map(|(f, a)| *f * f64::from(*a))
                .sum();

            let magnitude = lhs.abs().max(rhs.abs());
            assert!(
                magnitude > 0.0,
                "inner products must be non-zero for a meaningful test"
            );
            let rel_err = (lhs - rhs).abs() / magnitude;
            assert!(
                rel_err < 5e-3,
                "adjoint identity violated: <Af,g>={lhs:.6}, <f,A†g>={rhs:.6}, rel_err={rel_err:.2e}"
            );
        }

        // 12. filtered_backproject_matches_cpu_reference
        {
            let image_f64 = array![[0.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
            let angles_f64: Vec<f64> = (0..4)
                .map(|i| i as f64 * std::f64::consts::FRAC_PI_4)
                .collect();
            let angles_f32: Vec<f32> = angles_f64.iter().map(|&a| a as f32).collect();

            let cpu_plan = RadonPlan::new(3, 3, angles_f64, 5, 1.0).expect("cpu plan");
            let cpu_sinogram = cpu_plan.forward(&image_f64).expect("cpu forward");
            let cpu_fbp = cpu_plan
                .filtered_backprojection(&cpu_sinogram)
                .expect("cpu fbp");

            let gpu_plan = backend.plan(3, 3, 4, 5, 1.0);
            let sinogram_f32 = cpu_sinogram.values().mapv(|v| v as f32);
            let gpu_fbp = backend
                .execute_filtered_backproject(&gpu_plan, &sinogram_f32, &angles_f32)
                .expect("gpu fbp");

            assert_eq!(gpu_fbp.dim(), (3, 3));
            const TOL: f32 = 5e-2;
            for ((r, c), gpu_val) in gpu_fbp.indexed_iter() {
                let cpu_val = cpu_fbp[(r, c)] as f32;
                let err = (gpu_val - cpu_val).abs();
                assert!(
                    err < TOL,
                    "FBP mismatch at ({r},{c}): gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
                );
            }
        }

        // 13. filtered_backproject_rejects_sinogram_shape_mismatch
        {
            let plan = backend.plan(3, 3, 2, 5, 1.0);
            let wrong_sinogram = Array2::<f32>::zeros((3, 5));
            let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
            let err = backend
                .execute_filtered_backproject(&plan, &wrong_sinogram, &angles)
                .expect_err("sinogram shape mismatch must fail");
            assert!(matches!(err, WgpuError::ShapeMismatch { .. }));
        }
    }
}
