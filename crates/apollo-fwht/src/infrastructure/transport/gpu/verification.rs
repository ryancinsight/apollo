//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use crate::infrastructure::transport::gpu::{
        FwhtWgpuBackend, FwhtWgpuPlan, WgpuCapabilities, WgpuError,
    };
    use crate::FwhtPlan;
    use apollo_fft::{f16, PrecisionProfile};
    use leto::Array1;
    use leto::{SliceArg, Storage};

    #[test]
    fn capabilities_reflect_implemented_kernel_surface() {
        let capabilities = WgpuCapabilities::implemented(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
        assert!(capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = FwhtWgpuPlan::new(64);
        assert_eq!(plan.len(), 64);
        assert!(!FwhtWgpuPlan::new(64).is_empty());
        assert!(FwhtWgpuPlan::new(0).is_empty());
    }

    #[test]
    fn fwht_wgpu_execution_suite_when_device_exists() {
        let device = match hephaestus_wgpu::WgpuDevice::try_default("apollo-fwht-wgpu") {
            Ok(device) => device,
            Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => return,
            Err(error) => panic!("FWHT GPU verification requires a working provider: {error}"),
        };
        let backend = FwhtWgpuBackend::new(device);

        // 1. rejects_invalid_plan_shape_before_dispatch
        {
            let empty_err = backend
                .execute_forward(&FwhtWgpuPlan::new(0), &[])
                .expect_err("empty plan must fail");
            assert!(matches!(empty_err, WgpuError::InvalidPlan { .. }));

            let non_power_err = backend
                .execute_forward(&FwhtWgpuPlan::new(6), &[0.0; 6])
                .expect_err("non-power-of-two plan must fail");
            assert!(matches!(non_power_err, WgpuError::InvalidPlan { .. }));

            let mismatch_err = backend
                .execute_forward(&FwhtWgpuPlan::new(8), &[0.0; 4])
                .expect_err("length mismatch must fail");
            assert!(matches!(
                mismatch_err,
                WgpuError::LengthMismatch {
                    expected: 8,
                    actual: 4,
                }
            ));
        }

        // 2. backend_reports_forward_and_inverse_when_device_exists
        {
            let capabilities = backend.capabilities();
            assert!(capabilities.device_available);
            assert!(capabilities.supports_forward);
            assert!(capabilities.supports_inverse);
        }

        // 3. forward_matches_cpu_reference_when_device_exists
        {
            let input = vec![1.0_f32, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0];
            let gpu_plan = backend.plan(input.len());
            let gpu = backend
                .execute_forward(&gpu_plan, &input)
                .expect("wgpu forward execution");

            let cpu_plan = FwhtPlan::new(input.len()).expect("cpu plan");
            let cpu_input =
                Array1::from(input.iter().map(|&value| value as f64).collect::<Vec<_>>());
            let cpu = cpu_plan.forward(&cpu_input).expect("cpu forward");

            assert_eq!(gpu.len(), cpu.size());
            for (actual, expected) in gpu.iter().zip(cpu.iter()) {
                assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
            }
        }

        // 4. inverse_recovers_input_when_device_exists
        {
            let input = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0, 0.25, -0.125];
            let plan = backend.plan(input.len());
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

        // 5. two_forward_passes_satisfy_hadamard_square_identity
        {
            let input = [1.0_f32, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0];
            let plan = backend.plan(input.len());
            let first = backend
                .execute_forward(&plan, &input)
                .expect("first provider forward");
            let second = backend
                .execute_forward(&plan, &first)
                .expect("second provider forward");
            let expected = [8.0_f32, -16.0, 24.0, 32.0, -40.0, 48.0, 56.0, -64.0];
            assert_eq!(second, expected, "H_n squared must equal n times I");
        }

        // 6. leto_forward_matches_slice_forward_when_device_exists
        {
            let input = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0, 0.25, -0.125];
            let leto_input =
                leto::Array1::from_shape_vec([input.len()], input.clone()).expect("input");
            let plan = backend.plan(input.len());
            let expected = backend
                .execute_forward(&plan, &input)
                .expect("slice forward");
            let actual = backend
                .execute_forward_leto(&plan, leto_input.view())
                .expect("leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 7. leto_strided_forward_matches_logical_slice_forward_when_device_exists
        {
            let logical = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0, 0.25, -0.125];
            let mut backing = Vec::with_capacity(logical.len() * 2);
            for value in &logical {
                backing.push(*value);
                backing.push(99.0);
            }
            let leto_input = leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
            let strided = leto_input
                .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
                .expect("strided view");
            let plan = backend.plan(logical.len());
            let expected = backend
                .execute_forward(&plan, &logical)
                .expect("slice forward");
            let actual = backend
                .execute_forward_leto(&plan, strided)
                .expect("strided leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 8. leto_inverse_matches_slice_inverse_when_device_exists
        {
            let input = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0, 0.25, -0.125];
            let plan = backend.plan(input.len());
            let spectrum = backend
                .execute_forward(&plan, &input)
                .expect("slice forward");
            let leto_spectrum =
                leto::Array1::from_shape_vec([spectrum.len()], spectrum.clone()).expect("spectrum");
            let expected = backend
                .execute_inverse(&plan, &spectrum)
                .expect("slice inverse");
            let actual = backend
                .execute_inverse_leto(&plan, leto_spectrum.view())
                .expect("leto inverse");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 9. typed_mixed_storage_matches_represented_f32_execution_when_device_exists
        {
            let represented = [0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0, 0.25, -0.125];
            let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
            let represented_input: Vec<f32> = input.iter().map(|value| value.to_f32()).collect();
            let plan = backend.plan(input.len());
            let expected_forward = backend
                .execute_forward(&plan, &represented_input)
                .expect("represented forward");
            let mut typed_forward = vec![f16::from_f32(0.0); input.len()];

            backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &input,
                    &mut typed_forward,
                )
                .expect("mixed forward");

            for (actual, expected) in typed_forward.iter().zip(expected_forward.iter()) {
                let expected = f16::from_f32(*expected);
                assert_eq!(actual.to_bits(), expected.to_bits());
            }

            let expected_inverse = backend
                .execute_inverse(&plan, &expected_forward)
                .expect("represented inverse");
            let mut typed_inverse = vec![f16::from_f32(0.0); input.len()];
            backend
                .execute_inverse_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &typed_forward,
                    &mut typed_inverse,
                )
                .expect("mixed inverse");

            for (actual, expected) in typed_inverse.iter().zip(expected_inverse.iter()) {
                assert_f16_quantized_close(*actual, *expected);
            }
        }

        // 10. typed_leto_forward_and_inverse_match_typed_slice_when_device_exists
        {
            let represented = [0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0, 0.25, -0.125];
            let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
            let leto_input =
                leto::Array1::from_shape_vec([input.len()], input.clone()).expect("input");
            let plan = backend.plan(input.len());

            let mut expected_forward = vec![f16::from_f32(0.0); input.len()];
            backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &input,
                    &mut expected_forward,
                )
                .expect("typed slice forward");
            let actual_forward = backend
                .execute_forward_leto_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    leto_input.view(),
                )
                .expect("typed leto forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward.as_slice()
            );

            let leto_spectrum =
                leto::Array1::from_shape_vec([expected_forward.len()], expected_forward.clone())
                    .expect("spectrum");
            let mut expected_inverse = vec![f16::from_f32(0.0); input.len()];
            backend
                .execute_inverse_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &expected_forward,
                    &mut expected_inverse,
                )
                .expect("typed slice inverse");
            let actual_inverse = backend
                .execute_inverse_leto_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    leto_spectrum.view(),
                )
                .expect("typed leto inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice()
            );
        }

        // 11. typed_path_rejects_profile_storage_mismatch_when_device_exists
        {
            let plan = backend.plan(2);
            let input = [f16::from_f32(1.0), f16::from_f32(-1.0)];
            let mut output = [f16::from_f32(0.0); 2];
            let error = backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    &input,
                    &mut output,
                )
                .expect_err("profile mismatch must fail");
            assert!(matches!(error, WgpuError::InvalidPrecisionProfile));
        }
    }

    fn assert_f16_quantized_close(actual: f16, expected: f32) {
        let actual = actual.to_f32();
        let quantum_bound = expected.abs() * 2.0_f32.powi(-10) + f32::from(f16::MIN_POSITIVE);
        assert!(
            (actual - expected).abs() <= quantum_bound,
            "f16 quantization mismatch: actual={actual}, expected={expected}, bound={quantum_bound}"
        );
    }
}
