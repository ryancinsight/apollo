//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_fft::{f16, PrecisionProfile};
    use crate::MellinPlan;
    use leto::{SliceArg, Storage};

    use crate::infrastructure::transport::gpu::{MellinWgpuBackend, MellinWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_forward_inverse_kernel_surface() {
        let capabilities = WgpuCapabilities::forward_inverse(true);
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
    fn plan_preserves_scale_configuration() {
        let plan = MellinWgpuPlan::new(64, 0.25_f64.to_bits(), 4.0_f64.to_bits());
        assert_eq!(plan.samples(), 64);
        assert_eq!(plan.min_scale(), 0.25);
        assert_eq!(plan.max_scale(), 4.0);
        assert!(!plan.is_empty());
        assert!(MellinWgpuPlan::new(0, 0.25_f64.to_bits(), 4.0_f64.to_bits()).is_empty());
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
    fn mellin_wgpu_execution_suite_when_device_exists() {
        let Ok(backend) = MellinWgpuBackend::try_default() else {
            return;
        };

        // 1. backend_reports_forward_and_inverse
        {
            let capabilities = backend.capabilities();
            assert!(capabilities.device_available);
            assert!(capabilities.supports_forward);
            assert!(capabilities.supports_inverse);
        }

        // 2. forward_spectrum_matches_cpu_reference
        {
            let signal = vec![1.0_f32, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
            let signal_min = 1.0_f64;
            let signal_max = 8.0_f64;
            let plan = backend.plan(8, 1.0, 8.0);
            let gpu = backend
                .execute_forward(&plan, &signal, signal_min, signal_max)
                .expect("wgpu forward execution");

            let cpu_plan = MellinPlan::new(8, 1.0, 8.0).expect("cpu plan");
            let cpu = cpu_plan
                .forward_spectrum(
                    &signal
                        .iter()
                        .map(|&value| f64::from(value))
                        .collect::<Vec<_>>(),
                    signal_min,
                    signal_max,
                )
                .expect("cpu forward");

            assert_eq!(gpu.len(), cpu.values().len());
            for (actual, expected) in gpu.iter().zip(cpu.values().iter()) {
                assert!((f64::from(actual.re) - expected.re).abs() < 5.0e-4);
                assert!((f64::from(actual.im) - expected.im).abs() < 5.0e-4);
            }
        }

        // 3. typed_mixed_storage_matches_represented_f32_execution
        {
            let signal_f32 = [1.0_f32, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
            let signal_min = 1.0_f64;
            let signal_max = 8.0_f64;
            let plan = backend.plan(8, 1.0, 8.0);

            // Quantize to f16 and recover represented f32 for the reference path.
            let signal_f16: Vec<f16> = signal_f32.iter().copied().map(f16::from_f32).collect();
            let represented_f32: Vec<f32> = signal_f16.iter().map(|v| v.to_f32()).collect();

            let expected = backend
                .execute_forward(&plan, &represented_f32, signal_min, signal_max)
                .expect("represented f32 forward");
            let actual = backend
                .execute_forward_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &signal_f16,
                    signal_min,
                    signal_max,
                )
                .expect("typed mixed forward");

            assert_eq!(actual.len(), expected.len());
            for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (f64::from(a.re) - f64::from(e.re)).abs() < 1.0e-2,
                    "re mismatch at index {index}: actual={a:?} expected={e:?}"
                );
                assert!(
                    (f64::from(a.im) - f64::from(e.im)).abs() < 1.0e-2,
                    "im mismatch at index {index}: actual={a:?} expected={e:?}"
                );
            }
        }

        // 4. leto_forward_matches_slice
        {
            let signal = vec![1.0_f32, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
            let signal_min = 1.0_f64;
            let signal_max = 8.0_f64;
            let plan = backend.plan(8, 1.0, 8.0);
            let expected = backend
                .execute_forward(&plan, &signal, signal_min, signal_max)
                .expect("slice forward");
            let leto_signal =
                leto::Array1::from_shape_vec([signal.len()], signal).expect("leto signal");
            let actual = backend
                .execute_forward_leto(&plan, leto_signal.view(), signal_min, signal_max)
                .expect("leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 5. leto_strided_forward_matches_logical_slice
        {
            let logical = vec![1.0_f32, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
            let mut backing = Vec::with_capacity(logical.len() * 2);
            for value in logical.iter().copied() {
                backing.push(value);
                backing.push(99.0);
            }
            let signal_min = 1.0_f64;
            let signal_max = 8.0_f64;
            let plan = backend.plan(8, 1.0, 8.0);
            let expected = backend
                .execute_forward(&plan, &logical, signal_min, signal_max)
                .expect("slice forward");
            let leto_signal =
                leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
            let strided = leto_signal
                .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
                .expect("strided view");
            let actual = backend
                .execute_forward_leto(&plan, strided, signal_min, signal_max)
                .expect("strided leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 6. typed_leto_forward_matches_typed_slice
        {
            let signal_f32 = [1.0_f32, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
            let signal_min = 1.0_f64;
            let signal_max = 8.0_f64;
            let plan = backend.plan(8, 1.0, 8.0);
            let signal_f16: Vec<f16> = signal_f32.iter().copied().map(f16::from_f32).collect();
            let expected = backend
                .execute_forward_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &signal_f16,
                    signal_min,
                    signal_max,
                )
                .expect("typed forward");
            let leto_signal =
                leto::Array1::from_shape_vec([signal_f16.len()], signal_f16).expect("leto signal");
            let actual = backend
                .execute_forward_leto_typed::<f16>(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    leto_signal.view(),
                    signal_min,
                    signal_max,
                )
                .expect("typed leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 7. typed_path_rejects_profile_mismatch
        {
            let plan = backend.plan(8, 1.0, 8.0);
            let signal_f16: Vec<f16> = vec![f16::from_f32(1.0); 8];

            // f16 carries MIXED_PRECISION_F16_F32; passing LOW_PRECISION_F32 must fail.
            let err = backend
                .execute_forward_typed::<f16>(
                    &plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    &signal_f16,
                    1.0,
                    8.0,
                )
                .expect_err("profile mismatch must fail");
            assert_eq!(err, WgpuError::InvalidPrecisionProfile);
        }

        // 8. rejects_invalid_plan_and_signal_domain_before_dispatch
        {
            let invalid_plan = backend
                .execute_forward(
                    &MellinWgpuPlan::new(0, 1.0_f64.to_bits(), 8.0_f64.to_bits()),
                    &[1.0],
                    1.0,
                    2.0,
                )
                .expect_err("empty plan must fail");
            assert!(matches!(invalid_plan, WgpuError::InvalidPlan { .. }));

            let empty_signal = backend
                .execute_forward(
                    &MellinWgpuPlan::new(8, 1.0_f64.to_bits(), 8.0_f64.to_bits()),
                    &[],
                    1.0,
                    2.0,
                )
                .expect_err("empty signal must fail");
            assert_eq!(
                empty_signal,
                WgpuError::LengthMismatch {
                    expected: 1,
                    actual: 0,
                }
            );

            let invalid_domain = backend
                .execute_forward(
                    &MellinWgpuPlan::new(8, 1.0_f64.to_bits(), 8.0_f64.to_bits()),
                    &[1.0, 2.0, 3.0],
                    2.0,
                    1.0,
                )
                .expect_err("invalid signal domain must fail");
            assert!(matches!(
                invalid_domain,
                WgpuError::InvalidSignalDomain { .. }
            ));
        }

        // 9. gpu_inverse_roundtrip_constant_signal
        {
            let n = 16usize;
            let min_scale = 1.0_f64;
            let max_scale = 8.0_f64;
            let plan = backend.plan(n, min_scale, max_scale);
            let signal: Vec<f32> = vec![2.5; n];
            let spectrum = backend
                .execute_forward(&plan, &signal, min_scale, max_scale)
                .expect("GPU forward");
            let recovered = backend
                .execute_inverse(&plan, &spectrum, min_scale, max_scale, n)
                .expect("GPU inverse");
            assert_eq!(recovered.len(), n);
            for (i, &v) in recovered.iter().enumerate() {
                let err = (v - 2.5_f32).abs();
                assert!(
                    err < 5.0e-4,
                    "sample {i}: expected=2.5, got={v:.6}, err={err:.3e}"
                );
            }
        }

        // 10. leto_inverse_matches_slice
        {
            let n = 16usize;
            let min_scale = 1.0_f64;
            let max_scale = 8.0_f64;
            let plan = backend.plan(n, min_scale, max_scale);
            let signal = vec![2.5_f32; n];
            let spectrum = backend
                .execute_forward(&plan, &signal, min_scale, max_scale)
                .expect("forward");
            let expected = backend
                .execute_inverse(&plan, &spectrum, min_scale, max_scale, n)
                .expect("slice inverse");
            let leto_spectrum =
                leto::Array1::from_shape_vec([spectrum.len()], spectrum).expect("leto spectrum");
            let actual = backend
                .execute_inverse_leto(&plan, leto_spectrum.view(), min_scale, max_scale, n)
                .expect("leto inverse");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 11. gpu_inverse_rejects_invalid_output_domain
        {
            let n = 8usize;
            let plan = backend.plan(n, 1.0, 8.0);
            let spectrum = vec![eunomia::Complex32::new(0.0, 0.0); n];
            assert!(matches!(
                backend.execute_inverse(&plan, &spectrum, 0.0, 8.0, n),
                Err(WgpuError::InvalidSignalDomain { .. })
            ));
            assert!(matches!(
                backend.execute_inverse(&plan, &spectrum, 2.0, 1.0, n),
                Err(WgpuError::InvalidSignalDomain { .. })
            ));
        }
    }
}
