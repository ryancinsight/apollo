//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use crate::{SparseFftPlan, SparseSpectrum};
    use apollo_fft::{f16, PrecisionProfile};
    use eunomia::{Complex32, Complex64};
    use leto::{SliceArg, Storage};

    use crate::infrastructure::transport::gpu::{
        SftWgpuBackend, SftWgpuPlan, WgpuCapabilities, WgpuError,
    };

    // Four-term f32 inverse evaluation is bounded by twice gamma_12 times the
    // largest retained magnitude (1/3); gamma_12 = 12u / (1 - 12u).
    const INVERSE_N0_ERROR_BOUND: f32 = {
        let unit_roundoff = f32::EPSILON / 2.0;
        let gamma_12 = (12.0 * unit_roundoff) / (1.0 - 12.0 * unit_roundoff);
        2.0 * gamma_12 / 3.0
    };

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
    fn sft_wgpu_execution_suite_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };

        // 1. invalid_plan_rejects_zero_length
        {
            let error = backend
                .execute_forward(&SftWgpuPlan::new(0, 1), &[])
                .expect_err("zero length must be invalid");
            assert!(matches!(error, WgpuError::InvalidPlan { .. }));
        }

        // 2. input_length_mismatch_reports_expected_and_actual
        {
            let error = backend
                .execute_forward(&SftWgpuPlan::new(8, 2), &[Complex32::new(0.0, 0.0); 4])
                .expect_err("mismatched input length must be invalid");
            assert!(matches!(
                error,
                WgpuError::LengthMismatch {
                    expected: 8,
                    actual: 4
                }
            ));
        }

        // 3. forward_matches_cpu_sparse_support_and_coefficients
        {
            let plan = SftWgpuPlan::new(8, 2);
            let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
            let represented_signal: Vec<Complex32> = signal
                .iter()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect();

            let cpu = SparseFftPlan::new(plan.len(), plan.sparsity())
                .expect("valid CPU plan")
                .forward(&signal)
                .expect("CPU SFT");
            let gpu = backend
                .execute_forward(&plan, &represented_signal)
                .expect("GPU SFT");

            assert_eq!(gpu.frequencies, cpu.frequencies);
            assert_eq!(gpu.values.len(), cpu.values.len());
            for (actual, expected) in gpu.values.iter().zip(cpu.values.iter()) {
                assert_complex64_close(*actual, *expected, 2.0e-4);
            }
        }

        // 4. inverse_matches_cpu_sparse_reconstruction
        {
            let plan = SftWgpuPlan::new(8, 2);
            let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
            let cpu_plan = SparseFftPlan::new(plan.len(), plan.sparsity()).expect("valid CPU plan");
            let spectrum = backend
                .execute_forward(&plan, &represented_signal(&signal))
                .expect("GPU forward SFT");
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

        // 5. leto_forward_matches_slice_forward
        {
            let plan = SftWgpuPlan::new(8, 2);
            let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
            let input: Vec<Complex32> = signal
                .iter()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect();
            let leto_input =
                leto::Array1::from_shape_vec([input.len()], input.clone()).expect("input");

            let expected = backend
                .execute_forward(&plan, &input)
                .expect("slice forward");
            let actual = backend
                .execute_forward_leto(&plan, leto_input.view())
                .expect("leto forward");
            assert_eq!(actual.frequencies, expected.frequencies);
            assert_eq!(actual.values, expected.values);
        }

        // 6. leto_strided_forward_matches_logical_slice_forward
        {
            let plan = SftWgpuPlan::new(8, 2);
            let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
            let logical: Vec<Complex32> = signal
                .iter()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect();
            let mut backing = Vec::with_capacity(logical.len() * 2);
            for value in &logical {
                backing.push(*value);
                backing.push(Complex32::new(99.0, -99.0));
            }
            let leto_input = leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
            let strided = leto_input
                .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
                .expect("strided view");

            let expected = backend
                .execute_forward(&plan, &logical)
                .expect("slice forward");
            let actual = backend
                .execute_forward_leto(&plan, strided)
                .expect("strided leto forward");
            assert_eq!(actual.frequencies, expected.frequencies);
            assert_eq!(actual.values, expected.values);
        }

        // 7. leto_inverse_matches_slice_inverse
        {
            let plan = SftWgpuPlan::new(8, 2);
            let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
            let spectrum = backend
                .execute_forward(&plan, &represented_signal(&signal))
                .expect("GPU SFT");
            let expected = backend
                .execute_inverse(&plan, &spectrum)
                .expect("slice inverse");
            let actual = backend
                .execute_inverse_leto(&plan, &spectrum)
                .expect("leto inverse");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 8. typed_mixed_storage_forward_matches_complex32_execution
        {
            let plan = SftWgpuPlan::new(4, 2);
            let source_signal = two_tone_signal(4, &[(1, 3.0), (2, 1.25)]);
            let native_input: Vec<Complex32> = source_signal
                .iter()
                .map(|v| Complex32::new(v.re as f32, v.im as f32))
                .collect();

            let mixed_input: Vec<[f16; 2]> = native_input
                .iter()
                .map(|v| [f16::from_f32(v.re), f16::from_f32(v.im)])
                .collect();
            let represented_input: Vec<Complex32> = mixed_input
                .iter()
                .map(|v| Complex32::new(v[0].to_f32(), v[1].to_f32()))
                .collect();

            let f32_result = backend
                .execute_forward(&plan, &represented_input)
                .expect("represented f32 forward");
            let typed_result = backend
                .execute_forward_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &mixed_input,
                )
                .expect("typed mixed forward");

            assert_eq!(typed_result.frequencies, f32_result.frequencies);
            assert_eq!(typed_result.values.len(), f32_result.values.len());
            for (actual, expected) in typed_result.values.iter().zip(f32_result.values.iter()) {
                assert_complex64_close(*actual, *expected, 1.0e-3);
            }
        }

        // 9. typed_leto_forward_and_inverse_match_typed_slice
        {
            let plan = SftWgpuPlan::new(4, 2);
            let source_signal = two_tone_signal(4, &[(1, 3.0), (2, 1.25)]);
            let native_input: Vec<Complex32> = source_signal
                .iter()
                .map(|v| Complex32::new(v.re as f32, v.im as f32))
                .collect();
            let mixed_input: Vec<[f16; 2]> = native_input
                .iter()
                .map(|v| [f16::from_f32(v.re), f16::from_f32(v.im)])
                .collect();
            let leto_input = leto::Array1::from_shape_vec([mixed_input.len()], mixed_input.clone())
                .expect("input");

            let expected_forward = backend
                .execute_forward_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &mixed_input,
                )
                .expect("typed slice forward");
            let actual_forward = backend
                .execute_forward_leto_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    leto_input.view(),
                )
                .expect("typed leto forward");
            assert_eq!(actual_forward.frequencies, expected_forward.frequencies);
            assert_eq!(actual_forward.values, expected_forward.values);

            let mut expected_inverse = vec![[f16::from_f32(0.0); 2]; plan.len()];
            backend
                .execute_inverse_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &expected_forward,
                    &mut expected_inverse,
                )
                .expect("typed slice inverse");
            let actual_inverse = backend
                .execute_inverse_leto_typed::<[f16; 2]>(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &expected_forward,
                )
                .expect("typed leto inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice()
            );
        }

        // 10. typed_path_rejects_profile_mismatch
        {
            let plan = SftWgpuPlan::new(4, 2);
            let mixed_input: Vec<[f16; 2]> = vec![[f16::from_f32(0.0); 2]; 4];

            let fwd_err = backend
                .execute_forward_typed::<[f16; 2]>(
                    &plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    &mixed_input,
                )
                .expect_err("profile mismatch must fail");
            assert!(matches!(fwd_err, WgpuError::InvalidPrecisionProfile));

            let spectrum = SparseSpectrum::new(4);
            let mut mixed_output: Vec<[f16; 2]> = vec![[f16::from_f32(0.0); 2]; 4];
            let inv_err = backend
                .execute_inverse_typed_into::<[f16; 2]>(
                    &plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    &spectrum,
                    &mut mixed_output,
                )
                .expect_err("profile mismatch must fail");
            assert!(matches!(inv_err, WgpuError::InvalidPrecisionProfile));
        }

        // 11. high_accuracy_sparse_coefficients_are_not_silently_narrowed
        {
            let plan = SftWgpuPlan::new(4, 1);
            let mut spectrum = SparseSpectrum::new(4);
            spectrum
                .insert(1, Complex64::new(1.0 / 3.0, 0.0))
                .expect("in-range sparse coefficient");
            let error = backend
                .execute_inverse(&plan, &spectrum)
                .expect_err("non-f32 sparse coefficient must be rejected");
            assert!(matches!(
                error,
                WgpuError::PrecisionLoss {
                    component: "real",
                    value
                } if value == 1.0 / 3.0
            ));
        }

        // 12. explicit_quantization_is_required_and_value_visible
        {
            let plan = SftWgpuPlan::new(4, 1);
            let mut spectrum = SparseSpectrum::new(4);
            spectrum
                .insert(1, Complex64::new(1.0 / 3.0, -1.0 / 7.0))
                .expect("in-range sparse coefficient");
            let quantized = backend
                .quantize_spectrum(&plan, &spectrum)
                .expect("explicit accelerator quantization");
            assert_eq!(quantized.frequencies, spectrum.frequencies);
            assert_eq!(
                quantized.values,
                vec![Complex64::new(
                    f64::from((1.0_f64 / 3.0) as f32),
                    f64::from((-1.0_f64 / 7.0) as f32),
                )]
            );
            let actual = backend
                .execute_inverse(&plan, &quantized)
                .expect("quantized spectrum executes");
            assert_eq!(actual.len(), plan.len());
            assert_complex32_close(
                actual[0],
                Complex32::new(1.0 / 12.0, -1.0 / 28.0),
                INVERSE_N0_ERROR_BOUND,
            );
        }
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

    fn represented_signal(signal: &[Complex64]) -> Vec<Complex32> {
        signal
            .iter()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
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
