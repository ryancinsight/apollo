//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use crate::CztPlan;
    use apollo_fft::{f16, PrecisionProfile};
    use leto::{SliceArg, Storage};
    use ndarray::Array1;
    use num_complex::{Complex32, Complex64};

    use crate::infrastructure::transport::gpu::{
        Complex32 as GpuComplex32, CztWgpuBackend, CztWgpuPlan, WgpuCapabilities, WgpuError,
    };

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
    fn plan_preserves_logical_parameters() {
        let plan = CztWgpuPlan::new(
            64,
            96,
            [1.0_f32.to_bits(), 0.5_f32.to_bits()],
            [0.9_f32.to_bits(), (-0.25_f32).to_bits()],
        );
        assert_eq!(plan.input_len(), 64);
        assert_eq!(plan.output_len(), 96);
        assert_eq!(plan.a(), GpuComplex32::new(1.0, 0.5));
        assert_eq!(plan.w(), GpuComplex32::new(0.9, -0.25));
        assert!(!plan.is_empty());
        assert!(CztWgpuPlan::new(0, 64, [0, 0], [0, 0]).is_empty());
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
    fn czt_wgpu_execution_suite_when_device_exists() {
        let Ok(backend) = CztWgpuBackend::try_default() else {
            return;
        };

        // 1. backend_reports_forward_and_inverse
        {
            let capabilities = backend.capabilities();
            assert!(capabilities.device_available);
            assert!(capabilities.supports_forward);
            assert!(capabilities.supports_inverse);
        }

        // 2. forward_matches_cpu_direct_reference
        {
            let a32 = Complex32::new(0.95, 0.1);
            let w32 = Complex32::from_polar(1.0, -std::f32::consts::TAU / 9.0);
            let input = vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(-0.5, 1.0),
                Complex32::new(0.25, -0.75),
                Complex32::new(1.25, 0.5),
            ];
            let gpu_plan = backend.plan(input.len(), 6, a32, w32);
            let gpu = backend
                .execute_forward(&gpu_plan, &input)
                .expect("wgpu forward execution");

            let cpu_plan = CztPlan::new(
                input.len(),
                6,
                Complex64::new(f64::from(a32.re), f64::from(a32.im)),
                Complex64::new(f64::from(w32.re), f64::from(w32.im)),
            )
            .expect("cpu plan");
            let cpu_input = Array1::from_vec(
                input
                    .iter()
                    .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
                    .collect(),
            );
            let cpu = cpu_plan.forward_direct(&cpu_input).expect("cpu direct");

            assert_eq!(gpu.len(), cpu.len());
            for (actual, expected) in gpu.iter().zip(cpu.iter()) {
                assert!((f64::from(actual.re) - expected.re).abs() < 5.0e-4);
                assert!((f64::from(actual.im) - expected.im).abs() < 5.0e-4);
            }
        }

        // 3. typed_mixed_storage_matches_represented_f32_execution
        {
            let a32 = Complex32::new(0.95, 0.1);
            let w32 = Complex32::from_polar(1.0, -std::f32::consts::TAU / 9.0);
            let input_f32 = [
                Complex32::new(1.0, 0.0),
                Complex32::new(-0.5, 1.0),
                Complex32::new(0.25, -0.75),
                Complex32::new(1.25, 0.5),
            ];
            let input_f16: Vec<[f16; 2]> = input_f32
                .iter()
                .map(|c| [f16::from_f32(c.re), f16::from_f32(c.im)])
                .collect();
            let represented: Vec<Complex32> = input_f16
                .iter()
                .map(|v| Complex32::new(v[0].to_f32(), v[1].to_f32()))
                .collect();
            let gpu_plan = backend.plan(input_f16.len(), 6, a32, w32);
            let f32_result = backend
                .execute_forward(&gpu_plan, &represented)
                .expect("f32 reference");
            let mut typed_out = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; 6];
            backend
                .execute_forward_typed_into(
                    &gpu_plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &input_f16,
                    &mut typed_out,
                )
                .expect("typed mixed forward");
            for (actual, expected) in typed_out.iter().zip(f32_result.iter()) {
                let expected_f16 = [f16::from_f32(expected.re), f16::from_f32(expected.im)];
                assert_eq!(actual[0].to_bits(), expected_f16[0].to_bits());
                assert_eq!(actual[1].to_bits(), expected_f16[1].to_bits());
            }
        }

        // 4. leto_forward_matches_slice_forward
        {
            let a32 = Complex32::new(0.95, 0.1);
            let w32 = Complex32::from_polar(1.0, -std::f32::consts::TAU / 9.0);
            let input = vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(-0.5, 1.0),
                Complex32::new(0.25, -0.75),
                Complex32::new(1.25, 0.5),
            ];
            let plan = backend.plan(input.len(), 6, a32, w32);
            let expected = backend
                .execute_forward(&plan, &input)
                .expect("slice forward");
            let leto_input =
                leto::Array1::from_shape_vec([input.len()], input).expect("leto input");
            let actual = backend
                .execute_forward_leto(&plan, leto_input.view())
                .expect("leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 5. leto_strided_forward_matches_logical_slice_forward
        {
            let a32 = Complex32::new(0.95, 0.1);
            let w32 = Complex32::from_polar(1.0, -std::f32::consts::TAU / 9.0);
            let logical = vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(-0.5, 1.0),
                Complex32::new(0.25, -0.75),
                Complex32::new(1.25, 0.5),
            ];
            let sentinel = Complex32::new(99.0, -99.0);
            let mut backing = Vec::with_capacity(logical.len() * 2);
            for value in logical.iter().copied() {
                backing.push(value);
                backing.push(sentinel);
            }
            let plan = backend.plan(logical.len(), 6, a32, w32);
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

        // 6. typed_leto_forward_matches_typed_slice_forward
        {
            let a32 = Complex32::new(0.95, 0.1);
            let w32 = Complex32::from_polar(1.0, -std::f32::consts::TAU / 9.0);
            let input_f32 = [
                Complex32::new(1.0, 0.0),
                Complex32::new(-0.5, 1.0),
                Complex32::new(0.25, -0.75),
                Complex32::new(1.25, 0.5),
            ];
            let input_f16: Vec<[f16; 2]> = input_f32
                .iter()
                .map(|c| [f16::from_f32(c.re), f16::from_f32(c.im)])
                .collect();
            let plan = backend.plan(input_f16.len(), 6, a32, w32);
            let mut expected = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; 6];
            backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &input_f16,
                    &mut expected,
                )
                .expect("typed slice forward");
            let leto_input =
                leto::Array1::from_shape_vec([input_f16.len()], input_f16).expect("leto input");
            let actual = backend
                .execute_forward_leto_typed::<[f16; 2]>(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    leto_input.view(),
                )
                .expect("typed leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 7. typed_path_rejects_profile_mismatch
        {
            let a32 = Complex32::new(0.95, 0.1);
            let w32 = Complex32::from_polar(1.0, -std::f32::consts::TAU / 9.0);
            let input_f16: Vec<[f16; 2]> = vec![
                [f16::from_f32(1.0), f16::from_f32(0.0)],
                [f16::from_f32(-0.5), f16::from_f32(1.0)],
                [f16::from_f32(0.25), f16::from_f32(-0.75)],
                [f16::from_f32(1.25), f16::from_f32(0.5)],
            ];
            let mut out = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; 6];
            let gpu_plan = backend.plan(input_f16.len(), 6, a32, w32);
            let error = backend
                .execute_forward_typed_into(
                    &gpu_plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    &input_f16,
                    &mut out,
                )
                .expect_err("profile mismatch must fail");
            assert_eq!(error, WgpuError::InvalidPrecisionProfile);
        }

        // 8. rejects_invalid_lengths_and_parameters_before_dispatch
        {
            let empty_err = backend
                .execute_forward(&CztWgpuPlan::new(0, 5, [0, 0], [0, 0]), &[])
                .expect_err("empty plan must fail");
            assert!(matches!(empty_err, WgpuError::InvalidPlan { .. }));

            let mismatch_err = backend
                .execute_forward(
                    &CztWgpuPlan::new(
                        8,
                        8,
                        [1.0_f32.to_bits(), 0.0_f32.to_bits()],
                        [1.0_f32.to_bits(), 0.0_f32.to_bits()],
                    ),
                    &[Complex32::new(0.0, 0.0); 4],
                )
                .expect_err("length mismatch must fail");
            assert_eq!(
                mismatch_err,
                WgpuError::LengthMismatch {
                    expected: 8,
                    actual: 4,
                }
            );

            let invalid_param_err = backend
                .execute_forward(
                    &CztWgpuPlan::new(
                        4,
                        4,
                        [0.0_f32.to_bits(), 0.0_f32.to_bits()],
                        [1.0_f32.to_bits(), 0.0_f32.to_bits()],
                    ),
                    &[Complex32::new(0.0, 0.0); 4],
                )
                .expect_err("zero a must fail");
            assert!(matches!(invalid_param_err, WgpuError::InvalidPlan { .. }));
        }

        // 9. gpu_inverse_roundtrip_dft_parameters
        {
            let n = 8usize;
            let a32 = Complex32::new(1.0, 0.0);
            let w32 = Complex32::from_polar(1.0, -std::f32::consts::TAU / n as f32);
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.7).sin(), (i as f32 * 0.31).cos()))
                .collect();
            let gpu_plan = backend.plan(n, n, a32, w32);
            let spectrum = backend
                .execute_forward(&gpu_plan, &input)
                .expect("GPU forward");
            let recovered = backend
                .execute_inverse(&gpu_plan, &spectrum)
                .expect("GPU inverse");
            assert_eq!(recovered.len(), n);
            for (i, (rec, orig)) in recovered.iter().zip(input.iter()).enumerate() {
                let re_err = (rec.re - orig.re).abs();
                let im_err = (rec.im - orig.im).abs();
                assert!(re_err < 5.0e-4, "sample {i} re: err={re_err:.3e}");
                assert!(im_err < 5.0e-4, "sample {i} im: err={im_err:.3e}");
            }
        }

        // 10. leto_inverse_matches_slice_inverse_dft_parameters
        {
            let n = 8usize;
            let a32 = Complex32::new(1.0, 0.0);
            let w32 = Complex32::from_polar(1.0, -std::f32::consts::TAU / n as f32);
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.7).sin(), (i as f32 * 0.31).cos()))
                .collect();
            let plan = backend.plan(n, n, a32, w32);
            let spectrum = backend
                .execute_forward(&plan, &input)
                .expect("slice forward");
            let expected = backend
                .execute_inverse(&plan, &spectrum)
                .expect("slice inverse");
            let leto_spectrum =
                leto::Array1::from_shape_vec([spectrum.len()], spectrum).expect("leto spectrum");
            let actual = backend
                .execute_inverse_leto(&plan, leto_spectrum.view())
                .expect("leto inverse");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 11. gpu_inverse_rejects_non_square_plan
        {
            let a32 = Complex32::new(1.0, 0.0);
            let w32 = Complex32::from_polar(1.0, -0.5);
            let plan = backend.plan(4, 6, a32, w32);
            let spectrum = vec![Complex32::new(0.0, 0.0); 6];
            assert!(matches!(
                backend.execute_inverse(&plan, &spectrum),
                Err(WgpuError::LengthMismatch { .. })
            ));
        }
    }
}
