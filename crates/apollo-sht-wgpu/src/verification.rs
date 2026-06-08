//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_fft::{f16, PrecisionProfile};
    use apollo_sht::{ShtPlan, SphericalHarmonicCoefficients};
    use ndarray::Array2;
    use num_complex::{Complex32, Complex64};

    use crate::{ShtWgpuBackend, ShtWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_advertise_direct_complex_execution() {
        let capabilities = WgpuCapabilities::direct_complex(true);
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
    fn plan_preserves_grid_and_bandlimit() {
        let plan = ShtWgpuPlan::new(4, 5, 2);
        assert_eq!(plan.latitudes(), 4);
        assert_eq!(plan.longitudes(), 5);
        assert_eq!(plan.max_degree(), 2);
        assert_eq!(plan.sample_count(), 20);
        assert_eq!(plan.mode_count(), 9);
        assert!(!plan.is_empty());
        assert!(ShtWgpuPlan::new(0, 5, 0).is_empty());
        assert!(ShtWgpuPlan::new(4, 0, 0).is_empty());
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
    fn invalid_plan_rejects_under_sampled_bandlimit_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let samples = Array2::from_elem((2, 3), Complex32::new(1.0, 0.0));
        let error = backend
            .execute_forward(&ShtWgpuPlan::new(2, 3, 2), &samples)
            .expect_err("undersampled bandlimit must fail");
        assert!(matches!(error, WgpuError::InvalidPlan { .. }));
    }

    #[test]
    fn sample_shape_mismatch_reports_dimensions_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let samples = Array2::from_elem((3, 4), Complex32::new(1.0, 0.0));
        let error = backend
            .execute_forward(&ShtWgpuPlan::new(4, 5, 1), &samples)
            .expect_err("shape mismatch must fail");
        assert!(matches!(error, WgpuError::ShapeMismatch { .. }));
    }

    #[test]
    fn forward_matches_cpu_complex_coefficients_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let plan = ShtWgpuPlan::new(4, 5, 1);
        let cpu_plan = ShtPlan::new(plan.latitudes(), plan.longitudes(), plan.max_degree())
            .expect("valid CPU SHT plan");
        let samples = Array2::from_shape_fn((plan.latitudes(), plan.longitudes()), |(lat, lon)| {
            Complex64::new(
                0.25 + lat as f64 * 0.5 - lon as f64 * 0.125,
                0.1 * (lat as f64 + 1.0) * (lon as f64 + 1.0),
            )
        });
        let samples_f32 = samples.mapv(|value| Complex32::new(value.re as f32, value.im as f32));

        let expected = cpu_plan.forward_complex(&samples).expect("CPU forward");
        let actual = backend
            .execute_forward(&plan, &samples_f32)
            .expect("GPU forward");

        for degree in 0..=plan.max_degree() {
            for order in -(degree as isize)..=(degree as isize) {
                assert_complex64_close(
                    actual.get(degree, order),
                    expected.get(degree, order),
                    2.0e-5,
                );
            }
        }
    }

    #[test]
    fn inverse_matches_cpu_complex_samples_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let plan = ShtWgpuPlan::new(4, 5, 1);
        let cpu_plan = ShtPlan::new(plan.latitudes(), plan.longitudes(), plan.max_degree())
            .expect("valid CPU SHT plan");
        let samples = Array2::from_shape_fn((plan.latitudes(), plan.longitudes()), |(lat, lon)| {
            Complex64::new(
                0.25 + lat as f64 * 0.5 - lon as f64 * 0.125,
                0.1 * (lat as f64 + 1.0) * (lon as f64 + 1.0),
            )
        });
        let coefficients = cpu_plan.forward_complex(&samples).expect("CPU forward");
        let expected = cpu_plan
            .inverse_complex(&coefficients)
            .expect("CPU inverse");

        let actual = backend
            .execute_inverse(&plan, &coefficients)
            .expect("GPU inverse");

        assert_eq!(actual.dim(), expected.dim());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 2.0e-5);
        }
    }

    #[test]
    fn typed_flat_mixed_storage_matches_represented_forward_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        // latitudes=3, longitudes=5, max_degree=2: satisfies max_degree(2)<latitudes(3)
        // and 2*max_degree+1=5<=longitudes(5).
        let plan = ShtWgpuPlan::new(3, 5, 2);
        let flat_len = plan.latitudes() * plan.longitudes(); // 15

        // Build a non-trivial signal as f32, quantize to [f16; 2], recover represented f32.
        let signal_f32: Vec<Complex32> = (0..flat_len)
            .map(|i| Complex32::new(0.5 + i as f32 * 0.1, 0.1 * (i as f32 + 1.0)))
            .collect();
        let input_f16: Vec<[f16; 2]> = signal_f32
            .iter()
            .map(|v| [f16::from_f32(v.re), f16::from_f32(v.im)])
            .collect();
        let represented_f32: Vec<Complex32> = input_f16
            .iter()
            .map(|v| Complex32::new(v[0].to_f32(), v[1].to_f32()))
            .collect();
        let samples_2d =
            Array2::from_shape_vec((plan.latitudes(), plan.longitudes()), represented_f32)
                .expect("reshape");

        let expected = backend
            .execute_forward(&plan, &samples_2d)
            .expect("represented f32 forward");
        let actual = backend
            .execute_forward_flat_typed(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &input_f16,
            )
            .expect("typed flat mixed forward");

        assert_eq!(actual.max_degree(), plan.max_degree());
        assert_eq!(actual.max_degree(), expected.max_degree());
        for degree in 0..=plan.max_degree() {
            for order in -(degree as isize)..=(degree as isize) {
                let a = actual.get(degree, order);
                let e = expected.get(degree, order);
                assert!(
                    (a.re - e.re).abs() < 1.0e-3,
                    "re mismatch degree={degree} order={order}: actual={a:?} expected={e:?}"
                );
                assert!(
                    (a.im - e.im).abs() < 1.0e-3,
                    "im mismatch degree={degree} order={order}: actual={a:?} expected={e:?}"
                );
            }
        }
    }

    #[test]
    fn typed_flat_path_rejects_profile_mismatch_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let plan = ShtWgpuPlan::new(3, 5, 2);
        let flat_len = plan.latitudes() * plan.longitudes();
        let flat_input: Vec<[f16; 2]> = vec![[f16::from_f32(0.0); 2]; flat_len];

        // Forward: [f16; 2] carries MIXED_PRECISION_F16_F32; passing LOW_PRECISION_F32 must fail.
        let fwd_err = backend
            .execute_forward_flat_typed::<[f16; 2]>(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                &flat_input,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(fwd_err, WgpuError::InvalidPrecisionProfile);

        // Inverse: same mismatch on the output typed path.
        let coefficients = SphericalHarmonicCoefficients::zeros(plan.max_degree());
        let mut output: Vec<[f16; 2]> = vec![[f16::from_f32(0.0); 2]; flat_len];
        let inv_err = backend
            .execute_inverse_flat_typed_into::<[f16; 2]>(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                &coefficients,
                &mut output,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(inv_err, WgpuError::InvalidPrecisionProfile);
    }

    fn backend_or_skip() -> Option<ShtWgpuBackend> {
        match ShtWgpuBackend::try_default() {
            Ok(backend) => Some(backend),
            Err(error) => {
                eprintln!("skipping WGPU-dependent SHT test: {error}");
                None
            }
        }
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
}
