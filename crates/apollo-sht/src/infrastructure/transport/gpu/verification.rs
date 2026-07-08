//! WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use crate::{ShtPlan, SphericalHarmonicCoefficients};
    use apollo_fft::{f16, PrecisionProfile};
    use eunomia::{Complex32, Complex64};
    use leto::Array2;
    use leto::{SliceArg, Storage};

    use crate::infrastructure::transport::gpu::{
        ShtWgpuBackend, ShtWgpuPlan, WgpuCapabilities, WgpuError,
    };

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
    fn sht_wgpu_execution_suite_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };

        // 1. invalid_plan_rejects_under_sampled_bandlimit
        {
            let samples = Array2::from_elem([2, 3], Complex32::new(1.0, 0.0));
            let error = backend
                .execute_forward(&ShtWgpuPlan::new(2, 3, 2), &samples)
                .expect_err("undersampled bandlimit must fail");
            assert!(matches!(error, WgpuError::InvalidPlan { .. }));
        }

        // 2. sample_shape_mismatch_reports_dimensions
        {
            let samples = Array2::from_elem([3, 4], Complex32::new(1.0, 0.0));
            let error = backend
                .execute_forward(&ShtWgpuPlan::new(4, 5, 1), &samples)
                .expect_err("shape mismatch must fail");
            assert!(matches!(error, WgpuError::ShapeMismatch { .. }));
        }

        // 3. forward_matches_cpu_complex_coefficients
        {
            let plan = ShtWgpuPlan::new(4, 5, 1);
            let cpu_plan = ShtPlan::new(plan.latitudes(), plan.longitudes(), plan.max_degree())
                .expect("valid CPU SHT plan");
            let samples =
                Array2::from_shape_fn([plan.latitudes(), plan.longitudes()], |[lat, lon]| {
                    Complex64::new(
                        0.25 + lat as f64 * 0.5 - lon as f64 * 0.125,
                        0.1 * (lat as f64 + 1.0) * (lon as f64 + 1.0),
                    )
                });
            let samples_f32 =
                samples.mapv(|value| Complex32::new(value.re as f32, value.im as f32));

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

        // 4. inverse_matches_cpu_complex_samples
        {
            let plan = ShtWgpuPlan::new(4, 5, 1);
            let cpu_plan = ShtPlan::new(plan.latitudes(), plan.longitudes(), plan.max_degree())
                .expect("valid CPU SHT plan");
            let samples =
                Array2::from_shape_fn([plan.latitudes(), plan.longitudes()], |[lat, lon]| {
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

            assert_eq!(actual.shape(), expected.shape());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_complex64_close(*actual, *expected, 2.0e-5);
            }
        }

        // 5. leto_forward_and_inverse_match_leto
        {
            let plan = ShtWgpuPlan::new(4, 5, 1);
            let samples =
                Array2::from_shape_fn([plan.latitudes(), plan.longitudes()], |[lat, lon]| {
                    Complex32::new(
                        0.25 + lat as f32 * 0.5 - lon as f32 * 0.125,
                        0.1 * (lat as f32 + 1.0) * (lon as f32 + 1.0),
                    )
                });
            let expected_forward = backend
                .execute_forward(&plan, &samples)
                .expect("leto forward");
            let samples_leto = leto::Array::from_mnemosyne_slice(
                [plan.latitudes(), plan.longitudes()],
                &samples.iter().copied().collect::<Vec<_>>(),
            )
            .expect("leto samples");
            let actual_forward = backend
                .execute_forward_leto(&plan, samples_leto.view())
                .expect("leto forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward
                    .values()
                    .as_slice()
                    .expect("contiguous coeffs")
            );

            let expected_inverse = backend
                .execute_inverse(&plan, &expected_forward)
                .expect("leto inverse");
            let actual_inverse = backend
                .execute_inverse_leto(&plan, actual_forward.view())
                .expect("leto inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice().expect("contiguous inverse")
            );
        }

        // 6. leto_strided_forward_matches_logical_leto
        {
            let plan = ShtWgpuPlan::new(4, 5, 1);
            let samples =
                Array2::from_shape_fn([plan.latitudes(), plan.longitudes()], |[lat, lon]| {
                    Complex32::new(lat as f32 + lon as f32 * 0.25, 0.5 + lon as f32 * 0.1)
                });
            let mut interleaved = Vec::with_capacity(plan.latitudes() * plan.longitudes() * 2);
            for value in samples.iter().copied() {
                interleaved.push(value);
                interleaved.push(Complex32::new(99.0, -99.0));
            }
            let interleaved_leto = leto::Array::from_mnemosyne_slice(
                [plan.latitudes(), plan.longitudes() * 2],
                &interleaved,
            )
            .expect("interleaved samples");
            let strided = interleaved_leto
                .slice_with::<2>(&[
                    SliceArg::range(Some(0), None, 1),
                    SliceArg::range(Some(0), None, 2),
                ])
                .expect("strided samples");
            let expected = backend
                .execute_forward(&plan, &samples)
                .expect("leto forward");
            let actual = backend
                .execute_forward_leto(&plan, strided)
                .expect("strided leto forward");
            assert_eq!(
                actual.storage().as_slice(),
                expected.values().as_slice().expect("contiguous coeffs")
            );
        }

        // 7. typed_flat_mixed_storage_matches_represented_forward
        {
            let plan = ShtWgpuPlan::new(3, 5, 2);
            let flat_len = plan.latitudes() * plan.longitudes();

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
                Array2::from_shape_vec([plan.latitudes(), plan.longitudes()], represented_f32)
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

        // 8. typed_flat_leto_forward_and_inverse_match_slice
        {
            let plan = ShtWgpuPlan::new(3, 5, 2);
            let flat_len = plan.sample_count();
            let input: Vec<[f16; 2]> = (0..flat_len)
                .map(|i| {
                    [
                        f16::from_f32(0.5 + i as f32 * 0.1),
                        f16::from_f32(0.1 * (i as f32 + 1.0)),
                    ]
                })
                .collect();

            let expected_forward = backend
                .execute_forward_flat_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &input,
                )
                .expect("typed slice forward");
            let input_leto =
                leto::Array::from_mnemosyne_slice([input.len()], &input).expect("typed leto input");
            let actual_forward = backend
                .execute_forward_flat_leto_typed(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    input_leto.view(),
                )
                .expect("typed leto forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward
                    .values()
                    .as_slice()
                    .expect("contiguous coeffs")
            );

            let mut expected_inverse = vec![[f16::from_f32(0.0); 2]; flat_len];
            backend
                .execute_inverse_flat_typed_into(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    &expected_forward,
                    &mut expected_inverse,
                )
                .expect("typed slice inverse");
            let actual_inverse = backend
                .execute_inverse_flat_leto_typed::<[f16; 2]>(
                    &plan,
                    PrecisionProfile::MIXED_PRECISION_F16_F32,
                    actual_forward.view(),
                )
                .expect("typed leto inverse");
            let actual_bits: Vec<[u16; 2]> = actual_inverse
                .storage()
                .as_slice()
                .iter()
                .map(|value| [value[0].to_bits(), value[1].to_bits()])
                .collect();
            let expected_bits: Vec<[u16; 2]> = expected_inverse
                .iter()
                .map(|value| [value[0].to_bits(), value[1].to_bits()])
                .collect();
            assert_eq!(actual_bits, expected_bits);
        }

        // 9. typed_flat_path_rejects_profile_mismatch
        {
            let plan = ShtWgpuPlan::new(3, 5, 2);
            let flat_len = plan.latitudes() * plan.longitudes();
            let flat_input: Vec<[f16; 2]> = vec![[f16::from_f32(0.0); 2]; flat_len];

            let fwd_err = backend
                .execute_forward_flat_typed::<[f16; 2]>(
                    &plan,
                    PrecisionProfile::LOW_PRECISION_F32,
                    &flat_input,
                )
                .expect_err("profile mismatch must fail");
            assert_eq!(fwd_err, WgpuError::InvalidPrecisionProfile);

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
