//! WGPU value-semantic verification for the SDFT backend.

#[cfg(test)]
mod tests {
    use crate::infrastructure::transport::gpu::{
        SdftWgpuBackend, SdftWgpuPlan, WgpuCapabilities, WgpuError,
    };
    use crate::SdftPlan;
    use apollo_fft::{f16, PrecisionProfile};
    use eunomia::Complex32;
    use leto::{SliceArg, Storage};

    #[test]
    fn plan_preserves_window_and_bin_parameters() {
        let plan = SdftWgpuPlan::new(16, 8);
        assert_eq!(plan.window_len(), 16);
        assert_eq!(plan.bin_count(), 8);
        assert_eq!(plan.len(), 16);
        assert!(!plan.is_empty());
        assert!(SdftWgpuPlan::new(0, 8).is_empty());
        assert!(SdftWgpuPlan::new(8, 0).is_empty());
    }

    #[test]
    fn capabilities_reflect_forward_and_inverse_surface() {
        let capabilities = WgpuCapabilities::forward_and_inverse(true);
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
    fn sdft_wgpu_execution_suite_when_device_exists() {
        let Ok(backend) = SdftWgpuBackend::try_default() else {
            return;
        };

        // 1. backend_reports_forward_and_inverse
        {
            let capabilities = backend.capabilities();
            assert!(capabilities.device_available);
            assert!(capabilities.supports_forward);
            assert!(capabilities.supports_inverse);
        }

        // 2. forward_matches_cpu_direct_bins
        {
            let samples: [f32; 8] = [1.0, 0.5, -0.5, -1.0, 0.5, 1.0, -0.25, 0.75];
            let reference_samples: Vec<f64> = samples.iter().map(|&x| f64::from(x)).collect();
            let plan = backend.plan(samples.len(), 4);
            let gpu = backend
                .execute_forward(&plan, &samples)
                .expect("wgpu sdft forward execution");
            let cpu_plan = SdftPlan::new(8, 4).expect("cpu sdft plan");
            let cpu = cpu_plan
                .direct_bins(&reference_samples)
                .expect("cpu direct bins");
            assert_eq!(gpu.len(), cpu.len(), "output bin count must match");
            for (index, (actual, expected)) in gpu.iter().zip(cpu.iter()).enumerate() {
                let real_error = (f64::from(actual.re) - expected.re).abs();
                let imag_error = (f64::from(actual.im) - expected.im).abs();
                assert!(
                    real_error < 1.0e-3 && imag_error < 1.0e-3,
                    "sdft mismatch at bin {index}: gpu=({},{}) cpu=({},{}) re_err={real_error} im_err={imag_error}",
                    actual.re,
                    actual.im,
                    expected.re,
                    expected.im,
                );
            }
        }

        // 3. leto_forward_matches_slice_forward
        {
            let window = vec![1.0_f32, 0.5, -0.5, -1.0, 0.5, 1.0, -0.25, 0.75];
            let input =
                leto::Array1::from_shape_vec([window.len()], window.clone()).expect("input");
            let plan = backend.plan(window.len(), 4);
            let expected = backend
                .execute_forward(&plan, &window)
                .expect("slice forward");
            let actual = backend
                .execute_forward_leto(&plan, input.view())
                .expect("leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 4. leto_strided_forward_matches_logical_slice_forward
        {
            let logical = vec![1.0_f32, 0.5, -0.5, -1.0, 0.5, 1.0, -0.25, 0.75];
            let mut backing = Vec::with_capacity(logical.len() * 2);
            for value in &logical {
                backing.push(*value);
                backing.push(99.0);
            }
            let input = leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
            let strided = input
                .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
                .expect("strided view");
            let plan = backend.plan(logical.len(), 4);
            let expected = backend
                .execute_forward(&plan, &logical)
                .expect("slice forward");
            let actual = backend
                .execute_forward_leto(&plan, strided)
                .expect("strided leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 5. rejects_invalid_plan_and_input_before_dispatch
        {
            let empty_err = backend
                .execute_forward(&SdftWgpuPlan::new(0, 4), &[])
                .expect_err("zero window_len must fail");
            assert!(matches!(empty_err, WgpuError::InvalidPlan { .. }));
            let mismatch_err = backend
                .execute_forward(&SdftWgpuPlan::new(8, 4), &[0.0_f32; 4])
                .expect_err("window length mismatch must fail");
            assert!(matches!(
                mismatch_err,
                WgpuError::LengthMismatch {
                    expected: 8,
                    actual: 4
                }
            ));
        }

        // 6. typed_mixed_storage_matches_represented_f32_execution
        {
            let samples: [f32; 8] = [1.0, 0.5, -0.5, -1.0, 0.5, 1.0, -0.25, 0.75];
            let reduced_samples: Vec<f16> = samples.iter().map(|&x| f16::from_f32(x)).collect();
            let represented: Vec<f32> = reduced_samples.iter().map(|v| v.to_f32()).collect();
            let plan = backend.plan(reduced_samples.len(), 4);
            let reference = backend
                .execute_forward(&plan, &represented)
                .expect("f32 reference");
            let mut typed_out: Vec<[f16; 2]> = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; 4];
            backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &reduced_samples,
                    &mut typed_out,
                )
                .expect("typed mixed forward");
            for (actual, expected) in typed_out.iter().zip(reference.iter()) {
                let expected_reduced = [f16::from_f32(expected.re), f16::from_f32(expected.im)];
                assert_eq!(
                    actual[0].to_bits(),
                    expected_reduced[0].to_bits(),
                    "re bits mismatch: actual={:?} expected={:?}",
                    actual[0],
                    expected_reduced[0]
                );
                assert_eq!(
                    actual[1].to_bits(),
                    expected_reduced[1].to_bits(),
                    "im bits mismatch: actual={:?} expected={:?}",
                    actual[1],
                    expected_reduced[1]
                );
            }
        }

        // 7. typed_leto_forward_matches_typed_slice_forward
        {
            let samples: [f32; 8] = [1.0, 0.5, -0.5, -1.0, 0.5, 1.0, -0.25, 0.75];
            let reduced_samples: Vec<f16> = samples.iter().map(|&x| f16::from_f32(x)).collect();
            let input =
                leto::Array1::from_shape_vec([reduced_samples.len()], reduced_samples.clone())
                    .expect("input");
            let plan = backend.plan(reduced_samples.len(), 4);
            let mut expected: Vec<[f16; 2]> = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; 4];
            backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &reduced_samples,
                    &mut expected,
                )
                .expect("typed slice forward");
            let actual = backend
                .execute_forward_leto_typed::<f16, [f16; 2]>(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    input.view(),
                )
                .expect("typed leto forward");
            let actual = actual.storage().as_slice();
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    actual[0].to_bits(),
                    expected[0].to_bits(),
                    "typed re mismatch at {index}"
                );
                assert_eq!(
                    actual[1].to_bits(),
                    expected[1].to_bits(),
                    "typed im mismatch at {index}"
                );
            }
        }

        // 8. typed_path_rejects_profile_mismatch
        {
            let reduced_samples: Vec<f16> = vec![f16::from_f32(1.0); 8];
            let mut out: Vec<[f16; 2]> = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; 4];
            let plan = backend.plan(reduced_samples.len(), 4);
            let error = backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &reduced_samples,
                    &mut out,
                )
                .expect_err("profile mismatch must fail");
            assert!(matches!(error, WgpuError::InvalidPrecisionProfile));
        }

        // 9. inverse_roundtrip_matches_original_signal
        {
            let original: [f32; 8] = [1.0, 0.5, -0.5, -1.0, 0.5, 1.0, -0.25, 0.75];
            let plan = backend.plan(original.len(), original.len());
            let bins = backend.execute_forward(&plan, &original).expect("forward");
            let reconstructed = backend.execute_inverse(&plan, &bins).expect("inverse");
            assert_eq!(reconstructed.len(), original.len());
            for (index, (actual, expected)) in reconstructed.iter().zip(original.iter()).enumerate()
            {
                let error = (f64::from(*actual) - f64::from(*expected)).abs();
                assert!(
                    error < 5.0e-4,
                    "roundtrip mismatch at index {index}: actual={actual}, expected={expected}, error={error}"
                );
            }
        }

        // 10. leto_inverse_matches_slice_inverse
        {
            let original: [f32; 8] = [1.0, 0.5, -0.5, -1.0, 0.5, 1.0, -0.25, 0.75];
            let plan = backend.plan(original.len(), original.len());
            let bins = backend.execute_forward(&plan, &original).expect("forward");
            let input = leto::Array1::from_shape_vec([bins.len()], bins.clone()).expect("bins");
            let expected = backend
                .execute_inverse(&plan, &bins)
                .expect("slice inverse");
            let actual = backend
                .execute_inverse_leto(&plan, input.view())
                .expect("leto inverse");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 11. inverse_matches_cpu_reference
        {
            let original: [f32; 2] = [3.0, 1.0];
            let plan = backend.plan(original.len(), original.len());
            let bins = backend.execute_forward(&plan, &original).expect("forward");

            assert!(
                (bins[0].re - 4.0_f32).abs() < 1.0e-3 && bins[0].im.abs() < 1.0e-3,
                "DC bin mismatch: re={} im={}",
                bins[0].re,
                bins[0].im,
            );
            assert!(
                (bins[1].re - 2.0_f32).abs() < 1.0e-3 && bins[1].im.abs() < 1.0e-3,
                "Nyquist bin mismatch: re={} im={}",
                bins[1].re,
                bins[1].im,
            );

            let gpu_inverse = backend.execute_inverse(&plan, &bins).expect("inverse");

            assert!(
                (gpu_inverse[0] - 3.0_f32).abs() < 5.0e-4,
                "inverse[0] mismatch: actual={}",
                gpu_inverse[0],
            );
            assert!(
                (gpu_inverse[1] - 1.0_f32).abs() < 5.0e-4,
                "inverse[1] mismatch: actual={}",
                gpu_inverse[1],
            );
        }

        // 12. inverse_rejects_bin_count_mismatch
        {
            let bins: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); 4];
            let plan = backend.plan(8, 8);
            let error = backend
                .execute_inverse(&plan, &bins)
                .expect_err("bin count mismatch must fail");
            assert!(matches!(
                error,
                WgpuError::LengthMismatch {
                    expected: 8,
                    actual: 4,
                }
            ));
        }
    }
}
