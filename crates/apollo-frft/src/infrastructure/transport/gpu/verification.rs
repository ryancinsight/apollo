//! WGPU value-semantic verification for the FrFT GPU backend.

#[cfg(test)]
mod tests {
    use apollo_fft::{f16, PrecisionProfile};
    use eunomia::{Complex32, Complex64};
    use leto::{SliceArg, Storage};

    use crate::infrastructure::transport::gpu::{
        FrftWgpuBackend, FrftWgpuPlan, UnitaryFrftWgpuPlan, WgpuCapabilities, WgpuError,
    };

    #[test]
    fn capabilities_reflect_implemented_kernel_surface() {
        let capabilities = WgpuCapabilities::implemented(true);
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
    fn plan_preserves_logical_length() {
        let plan = FrftWgpuPlan::new(64, 1.0_f32);
        assert_eq!(plan.len(), 64);

        assert_eq!(plan.order(), 1.0_f32);
        assert!(!FrftWgpuPlan::new(64, 1.0).is_empty());
        assert!(FrftWgpuPlan::new(0, 1.0).is_empty());
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
    fn frft_wgpu_execution_suite_when_device_exists() {
        let Ok(backend) = FrftWgpuBackend::try_default() else {
            return;
        };

        // 1. backend_reports_forward_and_inverse
        {
            let caps = backend.capabilities();
            assert!(caps.device_available);
            assert!(caps.supports_forward);
            assert!(caps.supports_inverse);
        }

        // 2. forward_at_order_zero_is_identity
        {
            let n = 8_usize;
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new(i as f32 * 0.1_f32, -(i as f32) * 0.05_f32))
                .collect();
            let plan = FrftWgpuPlan::new(n, 0.0_f32);
            let output = backend
                .execute_forward(&plan, &input)
                .expect("forward order 0");
            assert_eq!(output.len(), n);
            for (k, (actual, expected)) in output.iter().zip(input.iter()).enumerate() {
                assert!(
                    (actual.re - expected.re).abs() < 1.0e-6_f32,
                    "k={} identity re: got={} want={}",
                    k,
                    actual.re,
                    expected.re
                );
                assert!(
                    (actual.im - expected.im).abs() < 1.0e-6_f32,
                    "k={} identity im: got={} want={}",
                    k,
                    actual.im,
                    expected.im
                );
            }
        }

        // 3. forward_at_order_one_matches_cpu_frft
        {
            let n = 16_usize;
            let input_f32: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.31_f32).sin(), 0.0_f32))
                .collect();
            let input_f64 = leto::Array1::from(
                input_f32
                    .iter()
                    .map(|v| Complex64::new(v.re as f64, v.im as f64))
                    .collect::<Vec<_>>(),
            );
            let plan = FrftWgpuPlan::new(n, 1.0_f32);
            let gpu = backend
                .execute_forward(&plan, &input_f32)
                .expect("gpu forward order 1");
            let cpu = crate::frft(&input_f64, 1.0_f64).expect("cpu frft order 1");
            assert_eq!(gpu.len(), n);
            for (k, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
                assert!(
                    (g.re as f64 - c.re).abs() < 1.0e-3_f64,
                    "k={} re: gpu={} cpu={}",
                    k,
                    g.re,
                    c.re
                );
                assert!(
                    (g.im as f64 - c.im).abs() < 1.0e-3_f64,
                    "k={} im: gpu={} cpu={}",
                    k,
                    g.im,
                    c.im
                );
            }
        }

        // 4. general_order_matches_cpu_frft
        {
            let n = 8_usize;
            let order_f32 = 0.5_f32;
            let input_f32: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.4_f32).cos(), (i as f32 * 0.3_f32).sin()))
                .collect();
            let input_f64 = leto::Array1::from(
                input_f32
                    .iter()
                    .map(|v| Complex64::new(v.re as f64, v.im as f64))
                    .collect::<Vec<_>>(),
            );
            let plan = FrftWgpuPlan::new(n, order_f32);
            let gpu = backend
                .execute_forward(&plan, &input_f32)
                .expect("gpu general frft");
            let cpu = crate::frft(&input_f64, order_f32 as f64).expect("cpu general frft");
            assert_eq!(gpu.len(), n);
            for (k, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
                assert!(
                    (g.re as f64 - c.re).abs() < 1.0e-3_f64,
                    "k={} re: gpu={:.6} cpu={:.6}",
                    k,
                    g.re,
                    c.re
                );
                assert!(
                    (g.im as f64 - c.im).abs() < 1.0e-3_f64,
                    "k={} im: gpu={:.6} cpu={:.6}",
                    k,
                    g.im,
                    c.im
                );
            }
        }

        // 5. typed_mixed_storage_frft_matches_represented_f32
        {
            let n = 8_usize;
            let order = 0.5_f32;
            // Build [f16; 2] input from well-conditioned f32 source values.
            let source_re = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.5, 0.0, -0.75];
            let source_im = [-0.25_f32, 0.75, -1.5, 0.5, 1.0, -0.25, 0.125, -1.0];
            let input: Vec<[f16; 2]> = source_re
                .iter()
                .zip(source_im.iter())
                .map(|(&re, &im)| [f16::from_f32(re), f16::from_f32(im)])
                .collect();
            // Represented f32 form is the round-trip through f16 quantization.
            let represented_f32: Vec<Complex32> = input
                .iter()
                .map(|[re, im]| Complex32::new(re.to_f32(), im.to_f32()))
                .collect();
            let plan = FrftWgpuPlan::new(n, order);
            let expected = backend
                .execute_forward(&plan, &represented_f32)
                .expect("f32 forward reference");
            let mut typed_output = vec![[f16::from_f32(0.0); 2]; n];
            backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &input,
                    &mut typed_output,
                )
                .expect("typed mixed forward");
            for (actual, expected) in typed_output.iter().zip(expected.iter()) {
                let expected_re = f16::from_f32(expected.re);
                let expected_im = f16::from_f32(expected.im);
                assert_eq!(
                    actual[0].to_bits(),
                    expected_re.to_bits(),
                    "re bits mismatch: actual={}, expected={}",
                    actual[0].to_f32(),
                    expected_re.to_f32()
                );
                assert_eq!(
                    actual[1].to_bits(),
                    expected_im.to_bits(),
                    "im bits mismatch: actual={}, expected={}",
                    actual[1].to_f32(),
                    expected_im.to_f32()
                );
            }
        }

        // 6. leto_forward_and_inverse_match_slice
        {
            let n = 8_usize;
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.4_f32).cos(), (i as f32 * 0.3_f32).sin()))
                .collect();
            let plan = FrftWgpuPlan::new(n, 0.5_f32);
            let expected_forward = backend
                .execute_forward(&plan, &input)
                .expect("slice forward");
            let leto_input =
                leto::Array1::from_shape_vec([input.len()], input).expect("leto input");
            let actual_forward = backend
                .execute_forward_leto(&plan, leto_input.view())
                .expect("leto forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward.as_slice()
            );

            let expected_inverse = backend
                .execute_inverse(&plan, &expected_forward)
                .expect("slice inverse");
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

        // 7. leto_strided_forward_matches_logical_slice
        {
            let logical: Vec<Complex32> = (0..8)
                .map(|i| Complex32::new((i as f32 * 0.2_f32).sin(), (i as f32 * 0.5_f32).cos()))
                .collect();
            let sentinel = Complex32::new(99.0, -99.0);
            let mut backing = Vec::with_capacity(logical.len() * 2);
            for value in logical.iter().copied() {
                backing.push(value);
                backing.push(sentinel);
            }
            let plan = FrftWgpuPlan::new(logical.len(), 0.5_f32);
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

        // 8. typed_leto_forward_and_inverse_match_typed_slice
        {
            let source_re = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.5, 0.0, -0.75];
            let source_im = [-0.25_f32, 0.75, -1.5, 0.5, 1.0, -0.25, 0.125, -1.0];
            let input: Vec<[f16; 2]> = source_re
                .iter()
                .zip(source_im.iter())
                .map(|(&re, &im)| [f16::from_f32(re), f16::from_f32(im)])
                .collect();
            let plan = FrftWgpuPlan::new(input.len(), 0.5_f32);
            let mut expected_forward = vec![[f16::from_f32(0.0); 2]; input.len()];
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
                .execute_forward_leto_typed::<[f16; 2]>(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    leto_input.view(),
                )
                .expect("leto typed forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward.as_slice()
            );

            let mut expected_inverse = vec![[f16::from_f32(0.0); 2]; expected_forward.len()];
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
                .execute_inverse_leto_typed::<[f16; 2]>(
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

        // 9. typed_path_rejects_profile_storage_mismatch
        {
            let plan = FrftWgpuPlan::new(2, 0.5_f32);
            let input = [
                [f16::from_f32(1.0), f16::from_f32(0.0)],
                [f16::from_f32(-1.0), f16::from_f32(0.5)],
            ];
            let mut output = [[f16::from_f32(0.0); 2]; 2];
            let error = backend
                .execute_forward_typed_into(
                    &plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    &input,
                    &mut output,
                )
                .expect_err("profile mismatch must fail");
            assert_eq!(error, WgpuError::InvalidPrecisionProfile);
        }

        // 10. inverse_recovers_input
        {
            let n = 16_usize;
            let order = 1.0_f32;
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.31_f32).sin(), (i as f32 * 0.17_f32).cos()))
                .collect();
            let plan = FrftWgpuPlan::new(n, order);
            let fwd = backend
                .execute_forward(&plan, &input)
                .expect("forward for roundtrip");
            let recovered = backend
                .execute_inverse(&plan, &fwd)
                .expect("inverse for roundtrip");
            assert_eq!(recovered.len(), n);
            for (k, (actual, expected)) in recovered.iter().zip(input.iter()).enumerate() {
                assert!(
                    (actual.re - expected.re).abs() < 1.0e-3_f32,
                    "roundtrip k={} re: got={:.6} want={:.6}",
                    k,
                    actual.re,
                    expected.re
                );
                assert!(
                    (actual.im - expected.im).abs() < 1.0e-3_f32,
                    "roundtrip k={} im: got={:.6} want={:.6}",
                    k,
                    actual.im,
                    expected.im
                );
            }
        }

        // 11. unitary_forward_order_zero_is_identity
        {
            let n = 8_usize;
            let input: Vec<Complex32> = (0..n)
                .map(|i| {
                    Complex32::new(
                        i as f32 * 0.1_f32 + 0.5_f32,
                        -(i as f32 * 0.07_f32) + 0.2_f32,
                    )
                })
                .collect();
            let plan = UnitaryFrftWgpuPlan::new(n, 0.0_f32);
            let output = backend
                .execute_unitary_forward(&plan, &input)
                .expect("unitary forward order 0");
            assert_eq!(output.len(), n);
            for (k, (actual, expected)) in output.iter().zip(input.iter()).enumerate() {
                assert!(
                    (actual.re - expected.re).abs() < 1.0e-5_f32,
                    "k={} identity re: got={:.8} want={:.8}",
                    k,
                    actual.re,
                    expected.re
                );
                assert!(
                    (actual.im - expected.im).abs() < 1.0e-5_f32,
                    "k={} identity im: got={:.8} want={:.8}",
                    k,
                    actual.im,
                    expected.im
                );
            }
        }

        // 12. unitary_leto_forward_and_inverse_match_slice
        {
            let n = 8_usize;
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.23_f32).sin(), (i as f32 * 0.31_f32).cos()))
                .collect();
            let plan = UnitaryFrftWgpuPlan::new(n, 0.5_f32);
            let expected_forward = backend
                .execute_unitary_forward(&plan, &input)
                .expect("unitary forward");
            let leto_input =
                leto::Array1::from_shape_vec([input.len()], input).expect("leto input");
            let actual_forward = backend
                .execute_unitary_forward_leto(&plan, leto_input.view())
                .expect("leto unitary forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward.as_slice()
            );

            let expected_inverse = backend
                .execute_unitary_inverse(&plan, &expected_forward)
                .expect("unitary inverse");
            let leto_spectrum =
                leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                    .expect("leto spectrum");
            let actual_inverse = backend
                .execute_unitary_inverse_leto(&plan, leto_spectrum.view())
                .expect("leto unitary inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice()
            );
        }

        // 13. unitary_forward_order_two_is_reversal
        {
            let n = 8_usize;
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.31_f32).sin(), (i as f32 * 0.17_f32).cos()))
                .collect();
            let plan = UnitaryFrftWgpuPlan::new(n, 2.0_f32);
            let output = backend
                .execute_unitary_forward(&plan, &input)
                .expect("unitary forward order 2");
            assert_eq!(output.len(), n);
            for k in 0..n {
                let expected = input[n - 1 - k];
                assert!(
                    (output[k].re - expected.re).abs() < 1.0e-5_f32,
                    "k={} reversal re: got={:.8} want={:.8}",
                    k,
                    output[k].re,
                    expected.re
                );
                assert!(
                    (output[k].im - expected.im).abs() < 1.0e-5_f32,
                    "k={} reversal im: got={:.8} want={:.8}",
                    k,
                    output[k].im,
                    expected.im
                );
            }
        }

        // 14. unitary_forward_inverse_roundtrip
        {
            let n = 16_usize;
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.23_f32).sin(), (i as f32 * 0.31_f32).cos()))
                .collect();
            for order in [0.3_f32, 0.5, 0.7, 1.3, 2.5, 3.1] {
                let plan = UnitaryFrftWgpuPlan::new(n, order);
                let spectrum = backend
                    .execute_unitary_forward(&plan, &input)
                    .expect("unitary forward for roundtrip");
                let recovered = backend
                    .execute_unitary_inverse(&plan, &spectrum)
                    .expect("unitary inverse for roundtrip");
                assert_eq!(recovered.len(), n);
                let max_err = recovered
                    .iter()
                    .zip(input.iter())
                    .map(|(a, e)| (a - e).norm())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_err < 1.0e-4_f32,
                    "roundtrip failed at order={}: max_element_error={:.2e}",
                    order,
                    max_err
                );
            }
        }

        // 15. unitary_forward_preserves_l2_norm
        {
            let n = 16_usize;
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.37_f32).cos(), (i as f32 * 0.41_f32).sin()))
                .collect();
            let input_norm_sq: f32 = input.iter().map(|c| c.norm_sqr()).sum();
            for order in [0.3_f32, 0.7, 1.2, 1.8, 2.7] {
                let plan = UnitaryFrftWgpuPlan::new(n, order);
                let output = backend
                    .execute_unitary_forward(&plan, &input)
                    .expect("unitary forward for norm test");
                assert_eq!(output.len(), n);
                let output_norm_sq: f32 = output.iter().map(|c| c.norm_sqr()).sum();
                let rel_err = (output_norm_sq - input_norm_sq).abs() / input_norm_sq;
                assert!(
                    rel_err < 5.0e-5_f32,
                    "norm not preserved at order={}: ||output||²={:.8}, ||input||²={:.8}, rel_err={:.2e}",
                    order,
                    output_norm_sq,
                    input_norm_sq,
                    rel_err
                );
            }
        }

        // 16. unitary_gpu_matches_cpu_reference
        {
            let n = 8_usize;
            let order = 0.5_f32;
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.4_f32).cos(), (i as f32 * 0.3_f32).sin()))
                .collect();
            let gpu_plan = UnitaryFrftWgpuPlan::new(n, order);
            let gpu_out = backend
                .execute_unitary_forward(&gpu_plan, &input)
                .expect("gpu unitary forward");
            assert_eq!(gpu_out.len(), n);

            // CPU reference: crate::UnitaryFrftPlan with f64 precision.
            let cpu_input = leto::Array1::from(
                input
                    .iter()
                    .map(|c| Complex64::new(c.re as f64, c.im as f64))
                    .collect::<Vec<_>>(),
            );
            let cpu_plan = crate::UnitaryFrftPlan::new(n, order as f64).expect("cpu unitary plan");
            let cpu_out = cpu_plan.forward(&cpu_input).expect("cpu unitary forward");

            for (k, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
                assert!(
                    (g.re as f64 - c.re).abs() < 1.0e-3_f64,
                    "k={} re: gpu={:.6} cpu={:.6}",
                    k,
                    g.re,
                    c.re
                );
                assert!(
                    (g.im as f64 - c.im).abs() < 1.0e-3_f64,
                    "k={} im: gpu={:.6} cpu={:.6}",
                    k,
                    g.im,
                    c.im
                );
            }
        }
    }
}
