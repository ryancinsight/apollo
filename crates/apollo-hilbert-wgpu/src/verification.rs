//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use crate::{HilbertWgpuBackend, HilbertWgpuPlan, WgpuCapabilities, WgpuError};
    use apollo_hilbert::HilbertPlan;
    use leto::{SliceArg, Storage};

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
    fn plan_preserves_logical_length() {
        let plan = HilbertWgpuPlan::new(64);
        assert_eq!(plan.len(), 64);
        assert!(!HilbertWgpuPlan::new(64).is_empty());
        assert!(HilbertWgpuPlan::new(0).is_empty());
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
    fn hilbert_wgpu_execution_suite_when_device_exists() {
        let Ok(backend) = HilbertWgpuBackend::try_default() else {
            return;
        };

        // 1. backend_reports_forward_and_inverse
        {
            let capabilities = backend.capabilities();
            assert!(capabilities.device_available);
            assert!(capabilities.supports_forward);
            assert!(capabilities.supports_inverse);
        }

        // 2. analytic_signal_matches_cpu_reference
        {
            let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
            let plan = backend.plan(input.len());
            let gpu = backend
                .execute_analytic_signal(&plan, &input)
                .expect("wgpu analytic execution");

            let cpu_plan = HilbertPlan::new(input.len()).expect("cpu plan");
            let cpu = cpu_plan
                .analytic_signal(
                    &input
                        .iter()
                        .map(|&value| f64::from(value))
                        .collect::<Vec<_>>(),
                )
                .expect("cpu analytic");

            assert_eq!(gpu.len(), cpu.values().len());
            for (actual, expected) in gpu.iter().zip(cpu.values().iter()) {
                assert!((f64::from(actual.re) - expected.re).abs() < 5.0e-4);
                assert!((f64::from(actual.im) - expected.im).abs() < 5.0e-4);
            }
        }

        // 3. quadrature_matches_cpu_reference
        {
            let input = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0];
            let plan = backend.plan(input.len());
            let gpu = backend
                .execute_forward(&plan, &input)
                .expect("wgpu transform execution");

            let cpu_plan = HilbertPlan::new(input.len()).expect("cpu plan");
            let cpu = cpu_plan
                .transform(
                    &input
                        .iter()
                        .map(|&value| f64::from(value))
                        .collect::<Vec<_>>(),
                )
                .expect("cpu transform");

            assert_eq!(gpu.len(), cpu.len());
            for (actual, expected) in gpu.iter().zip(cpu.iter()) {
                assert!((f64::from(*actual) - *expected).abs() < 5.0e-4);
            }
        }

        // 4. rejects_invalid_lengths_before_dispatch
        {
            let empty_err = backend
                .execute_forward(&HilbertWgpuPlan::new(0), &[])
                .expect_err("empty plan must fail");
            assert!(matches!(empty_err, WgpuError::InvalidPlan { .. }));

            let mismatch_err = backend
                .execute_forward(&HilbertWgpuPlan::new(8), &[0.0; 4])
                .expect_err("length mismatch must fail");
            assert_eq!(
                mismatch_err,
                WgpuError::LengthMismatch {
                    expected: 8,
                    actual: 4,
                }
            );
        }

        // 5. typed_mixed_storage_matches_represented_f32_execution
        {
            use apollo_fft::{f16, PrecisionProfile};
            let represented = [1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
            let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
            let represented_input: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
            let plan = backend.plan(input.len());
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
        }

        // 6. leto_analytic_signal_matches_slice
        {
            let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
            let plan = backend.plan(input.len());
            let expected = backend
                .execute_analytic_signal(&plan, &input)
                .expect("analytic signal");
            let leto_input =
                leto::Array1::from_shape_vec([input.len()], input).expect("leto input");
            let actual = backend
                .execute_analytic_signal_leto(&plan, leto_input.view())
                .expect("leto analytic signal");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 7. leto_strided_forward_matches_logical_slice
        {
            let logical = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0];
            let mut backing = Vec::with_capacity(logical.len() * 2);
            for value in logical.iter().copied() {
                backing.push(value);
                backing.push(99.0);
            }
            let plan = backend.plan(logical.len());
            let expected = backend.execute_forward(&plan, &logical).expect("forward");
            let leto_input = leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
            let strided = leto_input
                .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
                .expect("strided view");
            let actual = backend
                .execute_forward_leto(&plan, strided)
                .expect("strided leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 8. leto_inverse_matches_slice
        {
            let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
            let plan = backend.plan(input.len());
            let quadrature = backend.execute_forward(&plan, &input).expect("forward");
            let expected = backend
                .execute_inverse(&plan, &quadrature)
                .expect("inverse");
            let leto_quadrature =
                leto::Array1::from_shape_vec([quadrature.len()], quadrature).expect("leto input");
            let actual = backend
                .execute_inverse_leto(&plan, leto_quadrature.view())
                .expect("leto inverse");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 9. typed_leto_forward_and_inverse_match_typed_slice
        {
            use apollo_fft::{f16, PrecisionProfile};

            let represented = [1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
            let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
            let plan = backend.plan(input.len());
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
            let leto_quadrature =
                leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                    .expect("leto typed quadrature");
            let actual_inverse = backend
                .execute_inverse_leto_typed::<f16>(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    leto_quadrature.view(),
                )
                .expect("leto typed inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice()
            );
        }

        // 10. inverse_roundtrip_recovers_zero_mean_signal
        {
            let original = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
            // The Hilbert transform of a constant is zero, so the DC component is
            // irretrievably lost in the forward path. The inverse recovers the
            // zero-mean version of the original signal.
            let plan = backend.plan(original.len());
            let quadrature = backend
                .execute_forward(&plan, &original)
                .expect("forward hilbert");
            let recovered = backend
                .execute_inverse(&plan, &quadrature)
                .expect("inverse hilbert");
            assert_eq!(recovered.len(), original.len());
            // The Hilbert transform loses both DC and Nyquist (even N):
            // H{k=0} = 0 (DC), H{k=N/2} = 0 (Nyquist). The inverse recovers
            // the original signal minus DC and Nyquist contributions.
            // DC contribution: (sum of original) / N
            // Nyquist contribution: X[N/2] * (-1)^n / N where X[N/2] is the
            // DFT Nyquist bin (real for real signals).
            let n = original.len();
            let dc: f64 = original.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
            // Compute X[N/2] via DFT
            let tau = std::f64::consts::TAU;
            let mut nyquist_re = 0.0_f64;
            for j in 0..n {
                let angle = -tau * (n as f64) / 2.0 * (j as f64) / (n as f64);
                nyquist_re += f64::from(original[j]) * angle.cos();
            }
            let nyquist_contrib = |idx: usize| -> f64 {
                if n % 2 == 0 {
                    nyquist_re * if idx % 2 == 0 { 1.0 } else { -1.0 } / n as f64
                } else {
                    0.0 // No Nyquist bin for odd N
                }
            };
            let expected: Vec<f64> = original
                .iter()
                .enumerate()
                .map(|(idx, &x)| f64::from(x) - dc - nyquist_contrib(idx))
                .collect();
            const TOL: f64 = 5.0e-3;
            for (index, (actual, exp)) in recovered.iter().zip(expected.iter()).enumerate() {
                let error = (f64::from(*actual) - *exp).abs();
                assert!(
                    error < TOL,
                    "roundtrip mismatch at index {index}: actual={actual}, expected={exp}, error={error}"
                );
            }
        }

        // 11. inverse_matches_cpu_frequency_domain_reference
        {
            let original_f32 = [1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
            let original_f64: Vec<f64> = original_f32.iter().map(|&x| f64::from(x)).collect();
            let plan = backend.plan(original_f32.len());

            // Compute CPU Hilbert transform to get quadrature
            let cpu_plan = HilbertPlan::new(original_f32.len()).expect("cpu plan");
            let cpu_quadrature = cpu_plan
                .transform(&original_f64)
                .expect("cpu forward hilbert");

            // GPU inverse from the CPU quadrature (promoted to f32)
            let gpu_quadrature_f32: Vec<f32> = cpu_quadrature.iter().map(|&x| x as f32).collect();
            let gpu_recovered = backend
                .execute_inverse(&plan, &gpu_quadrature_f32)
                .expect("inverse hilbert");

            // CPU reference: frequency-domain inverse Hilbert.
            // Q[k] = -j * sgn(k) * X[k], so X[k] = Q[k] * j / sgn(k).
            // Positive: X[k] = j * Q[k]; Negative: X[k] = -j * Q[k];
            // DC and Nyquist: unrecoverable (zero).
            let n = cpu_quadrature.len();
            let positive_end = (n + 1) / 2;
            let tau = std::f64::consts::TAU;
            // DFT of quadrature
            let mut q_dft = vec![num_complex::Complex64::new(0.0, 0.0); n];
            for k in 0..n {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for j in 0..n {
                    let angle = -tau * (k as f64) * (j as f64) / (n as f64);
                    re += cpu_quadrature[j] * angle.cos();
                    im += cpu_quadrature[j] * angle.sin();
                }
                q_dft[k] = num_complex::Complex64::new(re, im);
            }
            // Inverse mask: X[k] = j * Q[k] for positive, -j * Q[k] for negative
            let mut x_dft = vec![num_complex::Complex64::new(0.0, 0.0); n];
            for k in 0..n {
                if k == 0 || (n % 2 == 0 && k == n / 2) {
                    x_dft[k] = num_complex::Complex64::new(0.0, 0.0); // DC/Nyquist lost
                } else if k < positive_end {
                    // Q[k] = -j * X[k], so X[k] = j * Q[k] = (-Q.im, Q.re)
                    x_dft[k] = num_complex::Complex64::new(-q_dft[k].im, q_dft[k].re);
                } else {
                    // Q[k] = j * X[k], so X[k] = -j * Q[k] = (Q.im, -Q.re)
                    x_dft[k] = num_complex::Complex64::new(q_dft[k].im, -q_dft[k].re);
                }
            }
            // IDFT of recovered X
            let mut cpu_recovered = vec![0.0_f64; n];
            for j in 0..n {
                let mut re = 0.0_f64;
                for k in 0..n {
                    let angle = tau * (k as f64) * (j as f64) / (n as f64);
                    re += x_dft[k].re * angle.cos() - x_dft[k].im * angle.sin();
                }
                cpu_recovered[j] = re / (n as f64);
            }

            for (index, (actual, expected)) in
                gpu_recovered.iter().zip(cpu_recovered.iter()).enumerate()
            {
                let error = (f64::from(*actual) - *expected).abs();
                assert!(
                    error < 5.0e-3,
                    "inverse mismatch at index {index}: gpu={actual}, cpu={expected}, error={error}"
                );
            }
        }

        // 12. typed_path_rejects_profile_mismatch
        {
            use apollo_fft::{f16, PrecisionProfile};
            let plan = backend.plan(8);
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
