//! Value-semantic verification for the Haar DWT WGPU backend.

#[cfg(test)]
mod tests {
    use crate::{DiscreteWavelet, DwtPlan};
    use leto::{SliceArg, Storage};

    use crate::infrastructure::transport::gpu::{
        WaveletWgpuBackend, WaveletWgpuPlan, WgpuCapabilities, WgpuError,
    };

    // Structural / plan tests (no GPU required)

    #[test]
    fn capabilities_reflect_forward_and_inverse() {
        let caps = WgpuCapabilities::implemented(true);
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
        assert!(caps.supports_mixed_precision);
        assert_eq!(
            caps.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
        let caps_off = WgpuCapabilities::implemented(false);
        assert!(!caps_off.device_available);
        assert!(!caps_off.supports_forward);
        assert!(!caps_off.supports_inverse);
        assert!(caps_off.supports_mixed_precision);
        assert_eq!(
            caps_off.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_len_and_levels() {
        let plan = WaveletWgpuPlan::new(64, 3);
        assert_eq!(plan.len(), 64);
        assert_eq!(plan.levels(), 3);
        assert!(!plan.is_empty());
        assert!(WaveletWgpuPlan::new(0, 3).is_empty());
        assert!(WaveletWgpuPlan::new(64, 0).is_empty());
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
    fn wavelet_wgpu_execution_suite_when_device_exists() {
        let Ok(backend) = WaveletWgpuBackend::try_default() else {
            return;
        };

        // 1. rejects_invalid_plan_before_dispatch
        {
            let r = backend.execute_forward(&WaveletWgpuPlan::new(6, 1), &[0.0f32; 6]);
            assert!(
                matches!(r, Err(WgpuError::InvalidPlan { .. })),
                "non-pow2: {r:?}"
            );
            let r = backend.execute_forward(&WaveletWgpuPlan::new(4, 0), &[0.0f32; 4]);
            assert!(
                matches!(r, Err(WgpuError::InvalidPlan { .. })),
                "zero levels: {r:?}"
            );
            let r = backend.execute_forward(&WaveletWgpuPlan::new(4, 3), &[0.0f32; 4]);
            assert!(
                matches!(r, Err(WgpuError::InvalidPlan { .. })),
                "levels too large: {r:?}"
            );
            let r = backend.execute_forward(&WaveletWgpuPlan::new(8, 1), &[0.0f32; 4]);
            assert!(
                matches!(r, Err(WgpuError::LengthMismatch { .. })),
                "len mismatch: {r:?}"
            );
        }

        // 2. backend_reports_forward_and_inverse_when_device_exists
        {
            let caps = backend.capabilities();
            assert!(caps.device_available);
            assert!(caps.supports_forward);
            assert!(caps.supports_inverse);
        }

        // 3. analytical_haar_two_sample_forward
        {
            let plan = WaveletWgpuPlan::new(2, 1);
            let signal = [2.0f32, 0.0f32];
            let out = backend.execute_forward(&plan, &signal).expect("forward");
            assert_eq!(out.len(), 2);
            let expected = std::f32::consts::SQRT_2;
            assert!(
                (out[0] - expected).abs() < 1e-5,
                "approx: got {} expected {}",
                out[0],
                expected
            );
            assert!(
                (out[1] - expected).abs() < 1e-5,
                "detail: got {} expected {}",
                out[1],
                expected
            );
        }

        // 4. analytical_haar_two_sample_inverse
        {
            let plan = WaveletWgpuPlan::new(2, 1);
            let sqrt2 = std::f32::consts::SQRT_2;
            let coeffs = [sqrt2, sqrt2];
            let out = backend.execute_inverse(&plan, &coeffs).expect("inverse");
            assert_eq!(out.len(), 2);
            assert!((out[0] - 2.0f32).abs() < 1e-5, "x[0]: got {}", out[0]);
            assert!((out[1] - 0.0f32).abs() < 1e-5, "x[1]: got {}", out[1]);
        }

        // 5. roundtrip_forward_inverse_single_level
        {
            let plan = WaveletWgpuPlan::new(8, 1);
            let signal: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).sin()).collect();
            let coeffs = backend.execute_forward(&plan, &signal).expect("forward");
            let recovered = backend.execute_inverse(&plan, &coeffs).expect("inverse");
            assert_eq!(recovered.len(), signal.len());
            for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < 1e-5,
                    "idx {i}: orig={orig:.8}, rec={rec:.8}"
                );
            }
        }

        // 6. roundtrip_forward_inverse_multi_level
        {
            let plan = WaveletWgpuPlan::new(8, 3);
            let signal = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let coeffs = backend.execute_forward(&plan, &signal).expect("forward");
            let recovered = backend.execute_inverse(&plan, &coeffs).expect("inverse");
            assert_eq!(recovered.len(), signal.len());
            for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < 1e-5,
                    "idx {i}: orig={orig:.8}, rec={rec:.8}"
                );
            }
        }

        // 7. forward_preserves_energy_parseval
        {
            let plan = WaveletWgpuPlan::new(16, 2);
            let signal: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();
            let coeffs = backend.execute_forward(&plan, &signal).expect("forward");
            let energy_in: f32 = signal.iter().map(|x| x * x).sum();
            let energy_out: f32 = coeffs.iter().map(|x| x * x).sum();
            assert!(
                (energy_in - energy_out).abs() < energy_in * 1e-5,
                "Parseval: in={energy_in:.8} out={energy_out:.8}"
            );
        }

        // 8. forward_matches_cpu_haar_coefficients_when_device_exists
        {
            let signal = vec![1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
            let plan = WaveletWgpuPlan::new(signal.len(), 3);
            let gpu = backend
                .execute_forward(&plan, &signal)
                .expect("wgpu forward Haar DWT");

            let cpu_plan =
                DwtPlan::new(signal.len(), 3, DiscreteWavelet::Haar).expect("cpu Haar DWT plan");
            let cpu_coeffs = cpu_plan
                .forward(
                    &signal
                        .iter()
                        .map(|&value| f64::from(value))
                        .collect::<Vec<_>>(),
                )
                .expect("cpu forward Haar DWT");
            let mut expected = cpu_coeffs
                .approximation()
                .iter()
                .map(|&value| value as f32)
                .collect::<Vec<_>>();
            for detail in cpu_coeffs.details().iter().rev() {
                expected.extend(detail.iter().map(|&value| value as f32));
            }

            assert_eq!(gpu.len(), expected.len());
            for (index, (actual, expected)) in gpu.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() < 1.0e-5,
                    "coefficient mismatch at index {index}: gpu={actual}, cpu={expected}"
                );
            }
        }

        // 9. leto_forward_and_inverse_match_slice_when_device_exists
        {
            let signal = vec![1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
            let plan = WaveletWgpuPlan::new(signal.len(), 3);
            let expected_forward = backend
                .execute_forward(&plan, &signal)
                .expect("slice forward");
            let signal_leto =
                leto::Array1::from_shape_vec([signal.len()], signal).expect("leto signal");
            let actual_forward = backend
                .execute_forward_leto(&plan, signal_leto.view())
                .expect("leto forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward.as_slice()
            );

            let expected_inverse = backend
                .execute_inverse(&plan, &expected_forward)
                .expect("slice inverse");
            let coeffs_leto =
                leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                    .expect("leto coeffs");
            let actual_inverse = backend
                .execute_inverse_leto(&plan, coeffs_leto.view())
                .expect("leto inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice()
            );
        }

        // 10. leto_strided_forward_matches_logical_slice_when_device_exists
        {
            let signal = vec![1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
            let backing = signal
                .iter()
                .flat_map(|value| [*value, 99.0_f32])
                .collect::<Vec<_>>();
            let signal_leto =
                leto::Array1::from_shape_vec([backing.len()], backing).expect("interleaved signal");
            let strided = signal_leto
                .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
                .expect("strided signal");
            let plan = WaveletWgpuPlan::new(signal.len(), 3);
            let expected = backend
                .execute_forward(&plan, &signal)
                .expect("slice forward");
            let actual = backend
                .execute_forward_leto(&plan, strided)
                .expect("strided leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 11. inverse_matches_cpu_haar_reconstruction_when_device_exists
        {
            let signal = vec![0.5_f32, 1.25, -0.75, 2.0, -1.0, 0.25, 1.5, -2.5];
            let plan = WaveletWgpuPlan::new(signal.len(), 3);
            let gpu_coeffs = backend
                .execute_forward(&plan, &signal)
                .expect("wgpu forward Haar DWT");
            let gpu_reconstructed = backend
                .execute_inverse(&plan, &gpu_coeffs)
                .expect("wgpu inverse Haar DWT");

            let cpu_plan =
                DwtPlan::new(signal.len(), 3, DiscreteWavelet::Haar).expect("cpu Haar DWT plan");
            let cpu_coeffs = cpu_plan
                .forward(
                    &signal
                        .iter()
                        .map(|&value| f64::from(value))
                        .collect::<Vec<_>>(),
                )
                .expect("cpu forward Haar DWT");
            let cpu_reconstructed = cpu_plan.inverse(&cpu_coeffs).expect("cpu inverse Haar DWT");

            assert_eq!(gpu_reconstructed.len(), cpu_reconstructed.len());
            for (index, (actual, expected)) in gpu_reconstructed
                .iter()
                .zip(cpu_reconstructed.iter())
                .enumerate()
            {
                assert!(
                    (f64::from(*actual) - *expected).abs() < 1.0e-5,
                    "reconstruction mismatch at index {index}: gpu={actual}, cpu={expected}"
                );
            }
        }

        // 12. typed_mixed_storage_matches_represented_f32_execution_when_device_exists
        {
            use apollo_fft::{f16, PrecisionProfile};
            let represented = [1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
            let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
            let represented_input: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
            let plan = WaveletWgpuPlan::new(input.len(), 3);
            let expected_fwd = backend
                .execute_forward(&plan, &represented_input)
                .expect("represented forward");
            let mut typed_fwd = vec![f16::from_f32(0.0); input.len()];
            backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &input,
                    &mut typed_fwd,
                )
                .expect("typed mixed forward");
            assert_eq!(typed_fwd.len(), expected_fwd.len());
            for (actual, expected) in typed_fwd.iter().zip(expected_fwd.iter()) {
                let expected_f16 = f16::from_f32(*expected);
                assert_eq!(actual.to_bits(), expected_f16.to_bits());
            }

            let expected_inv = backend
                .execute_inverse(&plan, &expected_fwd)
                .expect("represented inverse");
            let mut typed_inv = vec![f16::from_f32(0.0); input.len()];
            backend
                .execute_inverse_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &typed_fwd,
                    &mut typed_inv,
                )
                .expect("typed mixed inverse");
            for (actual, expected) in typed_inv.iter().zip(expected_inv.iter()) {
                let q = expected.abs() * 2.0_f32.powi(-10) + f32::from(f16::MIN_POSITIVE);
                assert!(
                    (actual.to_f32() - expected).abs() <= q,
                    "f16 quantization mismatch: actual={}, expected={}",
                    actual.to_f32(),
                    expected
                );
            }
        }

        // 13. typed_leto_forward_and_inverse_match_typed_slice_when_device_exists
        {
            use apollo_fft::{f16, PrecisionProfile};
            let represented = [1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
            let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
            let plan = WaveletWgpuPlan::new(input.len(), 3);

            let mut expected_forward = vec![f16::from_f32(0.0); input.len()];
            backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &input,
                    &mut expected_forward,
                )
                .expect("typed slice forward");
            let input_leto = leto::Array1::from_shape_vec([input.len()], input).expect("input");
            let actual_forward = backend
                .execute_forward_leto_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    input_leto.view(),
                )
                .expect("typed leto forward");
            assert_eq!(
                actual_forward
                    .storage()
                    .as_slice()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_forward
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );

            let mut expected_inverse = vec![f16::from_f32(0.0); expected_forward.len()];
            backend
                .execute_inverse_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &expected_forward,
                    &mut expected_inverse,
                )
                .expect("typed slice inverse");
            let coeffs_leto =
                leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                    .expect("coefficients");
            let actual_inverse = backend
                .execute_inverse_leto_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    coeffs_leto.view(),
                )
                .expect("typed leto inverse");
            assert_eq!(
                actual_inverse
                    .storage()
                    .as_slice()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_inverse
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }

        // 14. typed_path_rejects_profile_mismatch_when_device_exists
        {
            use apollo_fft::{f16, PrecisionProfile};
            let plan = WaveletWgpuPlan::new(8, 3);
            let input = vec![f16::from_f32(1.0); 8];
            let mut output = vec![f16::from_f32(0.0); 8];
            let err = backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    &input,
                    &mut output,
                )
                .expect_err("profile mismatch must fail");
            assert_eq!(err, WgpuError::InvalidPrecisionProfile);
        }
    }
}
