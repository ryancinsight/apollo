//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_fft::{f16, PrecisionProfile};
    use apollo_sft::{SparseFftPlan, SparseSpectrum};
    use num_complex::{Complex32, Complex64};

    use crate::{SftWgpuBackend, SftWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_advertise_direct_dense_sparse_execution() {
        let capabilities = WgpuCapabilities::direct_dense_spectrum(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
        assert!(capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_logical_length_and_sparsity() {
        let plan = SftWgpuPlan::new(64, 5);
        assert_eq!(plan.len(), 64);
        assert_eq!(plan.sparsity(), 5);
        assert!(!SftWgpuPlan::new(64, 5).is_empty());
        assert!(SftWgpuPlan::new(0, 5).is_empty());
        assert!(SftWgpuPlan::new(64, 0).is_empty());
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
    fn invalid_plan_rejects_zero_length_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let error = backend
            .execute_forward(&SftWgpuPlan::new(0, 1), &[])
            .expect_err("zero length must be invalid");
        assert!(matches!(error, WgpuError::InvalidPlan { .. }));
    }

    #[test]
    fn input_length_mismatch_reports_expected_and_actual_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let error = backend
            .execute_forward(&SftWgpuPlan::new(8, 2), &[Complex32::new(0.0, 0.0); 4])
            .expect_err("mismatched input length must be invalid");
        assert_eq!(
            error,
            WgpuError::LengthMismatch {
                expected: 8,
                actual: 4
            }
        );
    }

    #[test]
    fn forward_matches_cpu_sparse_support_and_coefficients_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let plan = SftWgpuPlan::new(8, 2);
        let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
        let signal_f32: Vec<Complex32> = signal
            .iter()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
            .collect();

        let cpu = SparseFftPlan::new(plan.len(), plan.sparsity())
            .expect("valid CPU plan")
            .forward(&signal)
            .expect("CPU SFT");
        let gpu = backend
            .execute_forward(&plan, &signal_f32)
            .expect("GPU SFT");

        assert_eq!(gpu.frequencies, cpu.frequencies);
        assert_eq!(gpu.values.len(), cpu.values.len());
        for (actual, expected) in gpu.values.iter().zip(cpu.values.iter()) {
            assert_complex64_close(*actual, *expected, 2.0e-4);
        }
    }

    #[test]
    fn inverse_matches_cpu_sparse_reconstruction_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let plan = SftWgpuPlan::new(8, 2);
        let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
        let cpu_plan = SparseFftPlan::new(plan.len(), plan.sparsity()).expect("valid CPU plan");
        let spectrum = cpu_plan.forward(&signal).expect("CPU SFT");
        let expected = cpu_plan.inverse(&spectrum).expect("CPU inverse");

        let actual = backend
            .execute_inverse(&plan, &spectrum)
            .expect("GPU inverse");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex32_close(
                *actual,
                Complex32::new(expected.re as f32, expected.im as f32),
                2.0e-4,
            );
        }
    }

    #[test]
    fn typed_mixed_storage_forward_matches_complex32_execution_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let plan = SftWgpuPlan::new(4, 2);
        let signal_c64 = two_tone_signal(4, &[(1, 3.0), (2, 1.25)]);
        let signal_c32: Vec<Complex32> = signal_c64
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();

        // Quantize to [f16; 2] and recover represented f32 for the reference path.
        let input_f16: Vec<[f16; 2]> = signal_c32
            .iter()
            .map(|v| [f16::from_f32(v.re), f16::from_f32(v.im)])
            .collect();
        let represented_c32: Vec<Complex32> = input_f16
            .iter()
            .map(|v| Complex32::new(v[0].to_f32(), v[1].to_f32()))
            .collect();

        let f32_result = backend
            .execute_forward(&plan, &represented_c32)
            .expect("represented f32 forward");
        let typed_result = backend
            .execute_forward_typed(&plan, PrecisionProfile::MIXED_PRECISION_F16_F32, &input_f16)
            .expect("typed mixed forward");

        assert_eq!(typed_result.frequencies, f32_result.frequencies);
        assert_eq!(typed_result.values.len(), f32_result.values.len());
        for (actual, expected) in typed_result.values.iter().zip(f32_result.values.iter()) {
            assert_complex64_close(*actual, *expected, 1.0e-3);
        }
    }

    #[test]
    fn typed_path_rejects_profile_mismatch_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let plan = SftWgpuPlan::new(4, 2);
        let input_f16: Vec<[f16; 2]> = vec![[f16::from_f32(0.0); 2]; 4];

        // Forward: [f16; 2] carries MIXED_PRECISION_F16_F32; passing LOW_PRECISION_F32 must fail.
        let fwd_err = backend
            .execute_forward_typed::<[f16; 2]>(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                &input_f16,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(fwd_err, WgpuError::InvalidPrecisionProfile);

        // Inverse: same mismatch on the output typed path.
        let spectrum = SparseSpectrum::new(4);
        let mut output_f16: Vec<[f16; 2]> = vec![[f16::from_f32(0.0); 2]; 4];
        let inv_err = backend
            .execute_inverse_typed_into::<[f16; 2]>(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                &spectrum,
                &mut output_f16,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(inv_err, WgpuError::InvalidPrecisionProfile);
    }

    fn backend_or_skip() -> Option<SftWgpuBackend> {
        match SftWgpuBackend::try_default() {
            Ok(backend) => Some(backend),
            Err(error) => {
                eprintln!("skipping WGPU-dependent SFT test: {error}");
                None
            }
        }
    }

    fn two_tone_signal(len: usize, tones: &[(usize, f64)]) -> Vec<Complex64> {
        (0..len)
            .map(|n| {
                tones
                    .iter()
                    .map(|(frequency, amplitude)| {
                        let angle = 2.0 * std::f64::consts::PI * (*frequency as f64) * (n as f64)
                            / (len as f64);
                        Complex64::new(amplitude * angle.cos(), amplitude * angle.sin())
                    })
                    .sum()
            })
            .collect()
    }

    fn assert_complex64_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        assert!(
            (actual.re - expected.re).abs() <= tolerance,
            "real mismatch: actual={actual:?}, expected={expected:?}"
        );
        assert!(
            (actual.im - expected.im).abs() <= tolerance,
            "imag mismatch: actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_complex32_close(actual: Complex32, expected: Complex32, tolerance: f32) {
        assert!(
            (actual.re - expected.re).abs() <= tolerance,
            "real mismatch: actual={actual:?}, expected={expected:?}"
        );
        assert!(
            (actual.im - expected.im).abs() <= tolerance,
            "imag mismatch: actual={actual:?}, expected={expected:?}"
        );
    }
}
