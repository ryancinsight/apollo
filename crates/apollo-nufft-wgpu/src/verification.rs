//! NUFFT WGPU value-semantic verification.

#[cfg(test)]
mod tests {
    use apollo_fft::{f16, PrecisionProfile};
    use apollo_nufft::{
        nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
        nufft_type2_1d_fast, nufft_type2_3d, nufft_type2_3d_fast, UniformDomain1D, UniformGrid3D,
        DEFAULT_NUFFT_KERNEL_WIDTH, DEFAULT_NUFFT_OVERSAMPLING,
    };
    use leto::{SliceArg, Storage};
    use ndarray::{Array1, Array3};
    use num_complex::{Complex32, Complex64};

    use crate::{
        NufftGpuBuffers1D, NufftGpuBuffers3D, NufftWgpuBackend, NufftWgpuCapabilities,
        NufftWgpuError, NufftWgpuPlan1D, NufftWgpuPlan3D,
    };

    #[test]
    fn capabilities_advertise_all_direct_execution() {
        let capabilities = NufftWgpuCapabilities::direct_all(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_type1_1d);
        assert!(capabilities.supports_type2_1d);
        assert!(capabilities.supports_type1_3d);
        assert!(capabilities.supports_type2_3d);
        assert!(!capabilities.supports_fast_type1_1d);
        assert!(!capabilities.supports_fast_type2_1d);
        assert!(!capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn capabilities_advertise_all_fast_when_enabled() {
        let capabilities = NufftWgpuCapabilities::direct_all_fast_all(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_type1_1d);
        assert!(capabilities.supports_type2_1d);
        assert!(capabilities.supports_type1_3d);
        assert!(capabilities.supports_type2_3d);
        assert!(capabilities.supports_fast_type1_1d);
        assert!(capabilities.supports_fast_type2_1d);
        assert!(capabilities.supports_fast_type1_3d);
        assert!(capabilities.supports_fast_type2_3d);
        assert!(capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_1d_preserves_validated_metadata() {
        let domain = UniformDomain1D::new(16, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        assert_eq!(plan.domain(), domain);
        assert_eq!(plan.oversampling(), 2);
        assert_eq!(plan.kernel_width(), 6);
    }

    #[test]
    fn plan_3d_preserves_validated_metadata() {
        let grid = UniformGrid3D::new(4, 8, 16, 0.1, 0.2, 0.3).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        assert_eq!(plan.grid(), grid);
        assert_eq!(plan.oversampling(), 2);
        assert_eq!(plan.kernel_width(), 6);
    }

    #[test]
    fn unsupported_execution_error_identifies_operation() {
        let err = NufftWgpuError::UnsupportedExecution {
            operation: "type2_1d",
        };
        assert_eq!(
            err.to_string(),
            "type2_1d is unsupported by the current apollo-nufft-wgpu capability set"
        );
    }

    #[test]
    fn length_mismatch_reports_expected_and_actual_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let error = backend
            .execute_type1_1d(&plan, &[0.0, 0.25], &[Complex32::new(1.0, 0.0)])
            .expect_err("length mismatch must fail");
        assert_eq!(
            error,
            NufftWgpuError::InputLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn fast_1d_reusable_buffers_reject_sample_capacity_overflow_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let buffers = NufftGpuBuffers1D::new(backend.device().as_ref(), 8, 16, 1);
        let error = backend
            .execute_fast_type1_1d_with_buffers(
                &plan,
                &buffers,
                &[0.0, 0.25],
                &[Complex32::new(1.0, 0.0), Complex32::new(0.5, -0.25)],
            )
            .expect_err("sample capacity overflow must fail");
        assert_eq!(
            error,
            NufftWgpuError::InputLengthMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn fast_3d_reusable_buffers_reject_sample_capacity_overflow_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let buffers = NufftGpuBuffers3D::new(backend.device().as_ref(), (3, 2, 2), (16, 16, 16), 1);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5)];
        let values = [Complex32::new(1.0, 0.0), Complex32::new(-0.25, 0.5)];
        let error = backend
            .execute_fast_type1_3d_with_buffers(&plan, &buffers, &positions, &values)
            .expect_err("sample capacity overflow must fail");
        assert_eq!(
            error,
            NufftWgpuError::InputLengthMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn type1_1d_matches_cpu_exact_reference_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15];
        let values = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(-0.75, 0.5),
            Complex32::new(0.25, 0.75),
        ];
        let expected_positions: Vec<f64> = positions.iter().map(|value| *value as f64).collect();
        let expected_values: Vec<Complex64> = values
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let expected = nufft_type1_1d(&expected_positions, &expected_values, domain);

        let actual = backend
            .execute_type1_1d(&plan, &positions, &values)
            .expect("GPU type1 1D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 4.0e-5);
        }
    }

    #[test]
    fn leto_type1_1d_matches_slice_path_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = vec![0.0_f32, 0.25, 0.7, 1.15];
        let values = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(-0.75, 0.5),
            Complex32::new(0.25, 0.75),
        ];
        let expected = backend
            .execute_type1_1d(&plan, &positions, &values)
            .expect("slice type1");
        let leto_positions =
            leto::Array1::from_shape_vec([positions.len()], positions).expect("positions");
        let leto_values = leto::Array1::from_shape_vec([values.len()], values).expect("values");

        let actual = backend
            .execute_type1_1d_leto(&plan, leto_positions.view(), leto_values.view())
            .expect("leto type1");

        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 1.0e-6);
        }
    }

    #[test]
    fn leto_strided_type2_1d_matches_slice_path_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(4, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let coefficients = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.25, -0.5),
            Complex32::new(-0.75, 0.25),
            Complex32::new(0.5, 0.75),
        ];
        let logical_positions = vec![0.0_f32, 0.25, 0.7, 1.15];
        let expected = backend
            .execute_type2_1d(&plan, &coefficients, &logical_positions)
            .expect("slice type2");
        let positions = leto::Array1::from_shape_vec(
            [8],
            vec![0.0_f32, 99.0, 0.25, 99.0, 0.7, 99.0, 1.15, 99.0],
        )
        .expect("positions");
        let strided_positions = positions
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided positions");
        let leto_coefficients =
            leto::Array1::from_shape_vec([coefficients.len()], coefficients).expect("coeffs");

        let actual = backend
            .execute_type2_1d_leto(&plan, leto_coefficients.view(), strided_positions)
            .expect("leto type2");

        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 1.0e-6);
        }
    }

    #[test]
    fn type1_1d_typed_mixed_storage_matches_represented_input_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15];
        let values16 = [
            [f16::from_f32(1.0), f16::from_f32(0.0)],
            [f16::from_f32(0.5), f16::from_f32(-0.25)],
            [f16::from_f32(-0.75), f16::from_f32(0.5)],
            [f16::from_f32(0.25), f16::from_f32(0.75)],
        ];
        let represented: Vec<Complex32> = values16
            .iter()
            .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
            .collect();
        let expected = backend
            .execute_type1_1d(&plan, &positions, &represented)
            .expect("represented type1 1D");
        let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; domain.n];

        backend
            .execute_type1_1d_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &positions,
                &values16,
                &mut actual,
            )
            .expect("mixed type1 1D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let expected_re = f16::from_f32(expected.re as f32);
            let expected_im = f16::from_f32(expected.im as f32);
            assert_eq!(actual[0].to_bits(), expected_re.to_bits());
            assert_eq!(actual[1].to_bits(), expected_im.to_bits());
        }
    }

    #[test]
    fn typed_leto_fast_type1_1d_matches_typed_slice_path_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15];
        let values16 = [
            [f16::from_f32(1.0), f16::from_f32(0.0)],
            [f16::from_f32(0.5), f16::from_f32(-0.25)],
            [f16::from_f32(-0.75), f16::from_f32(0.5)],
            [f16::from_f32(0.25), f16::from_f32(0.75)],
        ];
        let mut expected = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; domain.n];
        backend
            .execute_fast_type1_1d_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &positions,
                &values16,
                &mut expected,
            )
            .expect("typed fast slice");
        let leto_positions =
            leto::Array1::from_shape_vec([positions.len()], positions.to_vec()).expect("positions");
        let leto_values =
            leto::Array1::from_shape_vec([values16.len()], values16.to_vec()).expect("values");

        let actual = backend
            .execute_fast_type1_1d_leto_typed::<[f16; 2]>(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                leto_positions.view(),
                leto_values.view(),
            )
            .expect("typed fast leto");

        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_eq!(actual[0].to_bits(), expected[0].to_bits());
            assert_eq!(actual[1].to_bits(), expected[1].to_bits());
        }
    }

    #[test]
    fn type1_3d_matches_cpu_exact_reference_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let values = [
            Complex32::new(1.0, 0.0),
            Complex32::new(-0.25, 0.5),
            Complex32::new(0.75, -0.5),
        ];
        let expected_positions: Vec<(f64, f64, f64)> = positions
            .iter()
            .map(|(x, y, z)| (*x as f64, *y as f64, *z as f64))
            .collect();
        let expected_values: Vec<Complex64> = values
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let expected = nufft_type1_3d(&expected_positions, &expected_values, grid);

        let actual = backend
            .execute_type1_3d(&plan, &positions, &values)
            .expect("GPU type1 3D");

        assert_eq!(actual.dim(), expected.dim());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 8.0e-5);
        }
    }

    #[test]
    fn leto_type1_3d_matches_slice_path_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let values = [
            Complex32::new(1.0, 0.0),
            Complex32::new(-0.25, 0.5),
            Complex32::new(0.75, -0.5),
        ];
        let expected = backend
            .execute_type1_3d(&plan, &positions, &values)
            .expect("slice type1 3D");
        let leto_positions = leto::Array2::from_shape_vec(
            [positions.len(), 3],
            positions
                .iter()
                .flat_map(|(x, y, z)| [*x, *y, *z])
                .collect(),
        )
        .expect("positions");
        let leto_values =
            leto::Array1::from_shape_vec([values.len()], values.to_vec()).expect("values");

        let actual = backend
            .execute_type1_3d_leto(&plan, leto_positions.view(), leto_values.view())
            .expect("leto type1 3D");

        assert_eq!(actual.shape(), [grid.nx, grid.ny, grid.nz]);
        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 1.0e-6);
        }
    }

    #[test]
    fn type1_3d_typed_mixed_storage_matches_represented_input_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let values16 = [
            [f16::from_f32(1.0), f16::from_f32(0.0)],
            [f16::from_f32(-0.25), f16::from_f32(0.5)],
            [f16::from_f32(0.75), f16::from_f32(-0.5)],
        ];
        let represented: Vec<Complex32> = values16
            .iter()
            .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
            .collect();
        let expected = backend
            .execute_type1_3d(&plan, &positions, &represented)
            .expect("represented type1 3D");
        let mut actual = Array3::from_elem(
            (grid.nx, grid.ny, grid.nz),
            [f16::from_f32(0.0), f16::from_f32(0.0)],
        );

        backend
            .execute_type1_3d_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &positions,
                &values16,
                &mut actual,
            )
            .expect("mixed type1 3D");

        assert_eq!(actual.dim(), expected.dim());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let expected_re = f16::from_f32(expected.re as f32);
            let expected_im = f16::from_f32(expected.im as f32);
            assert_eq!(actual[0].to_bits(), expected_re.to_bits());
            assert_eq!(actual[1].to_bits(), expected_im.to_bits());
        }
    }

    #[test]
    fn type2_1d_rejects_coefficient_length_mismatch_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let error = backend
            .execute_type2_1d(&plan, &[Complex32::new(1.0, 0.0); 4], &[0.0_f32, 0.25, 0.7])
            .expect_err("coefficient length mismatch must fail");
        assert_eq!(
            error,
            NufftWgpuError::InputLengthMismatch {
                expected: 8,
                actual: 4
            }
        );
    }

    #[test]
    fn type2_1d_matches_cpu_exact_reference_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15, 1.8];
        let coefficients = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(-0.75, 0.5),
            Complex32::new(0.25, 0.75),
            Complex32::new(-0.5, -0.1),
            Complex32::new(0.125, 0.25),
            Complex32::new(0.8, -0.6),
            Complex32::new(-0.3, 0.4),
        ];
        let expected_positions: Vec<f64> = positions.iter().map(|value| *value as f64).collect();
        let expected_coefficients: Vec<Complex64> = coefficients
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let expected = nufft_type2_1d(
            &Array1::from_vec(expected_coefficients),
            &expected_positions,
            domain,
        );

        let actual = backend
            .execute_type2_1d(&plan, &coefficients, &positions)
            .expect("GPU type2 1D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 8.0e-5);
        }
    }

    #[test]
    fn type2_1d_typed_mixed_storage_matches_represented_input_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15, 1.8];
        let coefficients16 = [
            [f16::from_f32(1.0), f16::from_f32(0.0)],
            [f16::from_f32(0.5), f16::from_f32(-0.25)],
            [f16::from_f32(-0.75), f16::from_f32(0.5)],
            [f16::from_f32(0.25), f16::from_f32(0.75)],
            [f16::from_f32(-0.5), f16::from_f32(-0.1)],
            [f16::from_f32(0.125), f16::from_f32(0.25)],
            [f16::from_f32(0.8), f16::from_f32(-0.6)],
            [f16::from_f32(-0.3), f16::from_f32(0.4)],
        ];
        let represented: Vec<Complex32> = coefficients16
            .iter()
            .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
            .collect();
        let expected = backend
            .execute_type2_1d(&plan, &represented, &positions)
            .expect("represented type2 1D");
        let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];

        backend
            .execute_type2_1d_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &coefficients16,
                &positions,
                &mut actual,
            )
            .expect("mixed type2 1D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let expected_re = f16::from_f32(expected.re as f32);
            let expected_im = f16::from_f32(expected.im as f32);
            assert_eq!(actual[0].to_bits(), expected_re.to_bits());
            assert_eq!(actual[1].to_bits(), expected_im.to_bits());
        }
    }

    #[test]
    fn fast_type1_1d_matches_cpu_gridded_reference_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15];
        let values = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(-0.75, 0.5),
            Complex32::new(0.25, 0.75),
        ];
        let expected_positions: Vec<f64> = positions.iter().map(|value| *value as f64).collect();
        let expected_values: Vec<Complex64> = values
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let expected = nufft_type1_1d_fast(&expected_positions, &expected_values, domain, 6);

        let actual = backend
            .execute_fast_type1_1d(&plan, &positions, &values)
            .expect("GPU fast type1 1D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 1.5e-3);
        }
    }

    #[test]
    fn fast_type1_1d_typed_mixed_storage_matches_represented_input_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15];
        let values16 = [
            [f16::from_f32(1.0), f16::from_f32(0.0)],
            [f16::from_f32(0.5), f16::from_f32(-0.25)],
            [f16::from_f32(-0.75), f16::from_f32(0.5)],
            [f16::from_f32(0.25), f16::from_f32(0.75)],
        ];
        let represented: Vec<Complex32> = values16
            .iter()
            .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
            .collect();
        let expected = backend
            .execute_fast_type1_1d(&plan, &positions, &represented)
            .expect("represented fast type1 1D");
        let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; domain.n];

        backend
            .execute_fast_type1_1d_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &positions,
                &values16,
                &mut actual,
            )
            .expect("mixed fast type1 1D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let expected_re = f16::from_f32(expected.re as f32);
            let expected_im = f16::from_f32(expected.im as f32);
            assert_eq!(actual[0].to_bits(), expected_re.to_bits());
            assert_eq!(actual[1].to_bits(), expected_im.to_bits());
        }
    }

    #[test]
    fn fast_type2_1d_matches_cpu_gridded_reference_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15, 1.8];
        let coefficients = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(-0.75, 0.5),
            Complex32::new(0.25, 0.75),
            Complex32::new(-0.5, -0.1),
            Complex32::new(0.125, 0.25),
            Complex32::new(0.8, -0.6),
            Complex32::new(-0.3, 0.4),
        ];
        let expected_positions: Vec<f64> = positions.iter().map(|value| *value as f64).collect();
        let expected_coefficients: Vec<Complex64> = coefficients
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let expected = nufft_type2_1d_fast(
            &Array1::from_vec(expected_coefficients),
            &expected_positions,
            domain,
            6,
        );

        let actual = backend
            .execute_fast_type2_1d(&plan, &coefficients, &positions)
            .expect("GPU fast type2 1D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 1.5e-3);
        }
    }

    #[test]
    fn fast_type2_1d_typed_mixed_storage_matches_represented_input_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15, 1.8];
        let coefficients16 = [
            [f16::from_f32(1.0), f16::from_f32(0.0)],
            [f16::from_f32(0.5), f16::from_f32(-0.25)],
            [f16::from_f32(-0.75), f16::from_f32(0.5)],
            [f16::from_f32(0.25), f16::from_f32(0.75)],
            [f16::from_f32(-0.5), f16::from_f32(-0.1)],
            [f16::from_f32(0.125), f16::from_f32(0.25)],
            [f16::from_f32(0.8), f16::from_f32(-0.6)],
            [f16::from_f32(-0.3), f16::from_f32(0.4)],
        ];
        let represented: Vec<Complex32> = coefficients16
            .iter()
            .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
            .collect();
        let expected = backend
            .execute_fast_type2_1d(&plan, &represented, &positions)
            .expect("represented fast type2 1D");
        let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];

        backend
            .execute_fast_type2_1d_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &coefficients16,
                &positions,
                &mut actual,
            )
            .expect("mixed fast type2 1D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let expected_re = f16::from_f32(expected.re as f32);
            let expected_im = f16::from_f32(expected.im as f32);
            assert_eq!(actual[0].to_bits(), expected_re.to_bits());
            assert_eq!(actual[1].to_bits(), expected_im.to_bits());
        }
    }

    #[test]
    fn fast_type1_3d_matches_cpu_gridded_reference_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let values = [
            Complex32::new(1.0, 0.0),
            Complex32::new(-0.25, 0.5),
            Complex32::new(0.75, -0.5),
        ];
        let expected_positions: Vec<(f64, f64, f64)> = positions
            .iter()
            .map(|(x, y, z)| (*x as f64, *y as f64, *z as f64))
            .collect();
        let expected_values: Vec<Complex64> = values
            .iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .collect();
        let expected = nufft_type1_3d_fast(&expected_positions, &expected_values, grid, 6);

        let actual = backend
            .execute_fast_type1_3d(&plan, &positions, &values)
            .expect("GPU fast type1 3D");

        assert_eq!(actual.dim(), expected.dim());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 2.0e-3);
        }
    }

    #[test]
    fn fast_type1_3d_typed_mixed_storage_matches_represented_input_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let values16 = [
            [f16::from_f32(1.0), f16::from_f32(0.0)],
            [f16::from_f32(-0.25), f16::from_f32(0.5)],
            [f16::from_f32(0.75), f16::from_f32(-0.5)],
        ];
        let represented: Vec<Complex32> = values16
            .iter()
            .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
            .collect();
        let expected = backend
            .execute_fast_type1_3d(&plan, &positions, &represented)
            .expect("represented fast type1 3D");
        let mut actual = Array3::from_elem(
            (grid.nx, grid.ny, grid.nz),
            [f16::from_f32(0.0), f16::from_f32(0.0)],
        );

        backend
            .execute_fast_type1_3d_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &positions,
                &values16,
                &mut actual,
            )
            .expect("mixed fast type1 3D");

        assert_eq!(actual.dim(), expected.dim());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let expected_re = f16::from_f32(expected.re as f32);
            let expected_im = f16::from_f32(expected.im as f32);
            assert_eq!(actual[0].to_bits(), expected_re.to_bits());
            assert_eq!(actual[1].to_bits(), expected_im.to_bits());
        }
    }

    #[test]
    fn fast_type2_3d_matches_cpu_gridded_reference_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let modes = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(kx, ky, kz)| {
            Complex32::new(
                0.25 + 0.1 * kx as f32 - 0.05 * ky as f32 + 0.03 * kz as f32,
                -0.4 + 0.07 * kx as f32 + 0.11 * ky as f32 - 0.02 * kz as f32,
            )
        });
        let expected_positions: Vec<(f64, f64, f64)> = positions
            .iter()
            .map(|(x, y, z)| (*x as f64, *y as f64, *z as f64))
            .collect();
        let expected_modes = modes.mapv(|v| Complex64::new(v.re as f64, v.im as f64));
        let expected = nufft_type2_3d_fast(&expected_positions, &expected_modes, grid, 6);

        let actual = backend
            .execute_fast_type2_3d(&plan, &modes, &positions)
            .expect("GPU fast type2 3D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 2.0e-3);
        }
    }

    #[test]
    fn fast_type2_3d_typed_mixed_storage_matches_represented_input_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let modes16 = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(kx, ky, kz)| {
            [
                f16::from_f32(0.25 + 0.1 * kx as f32 - 0.05 * ky as f32 + 0.03 * kz as f32),
                f16::from_f32(-0.4 + 0.07 * kx as f32 + 0.11 * ky as f32 - 0.02 * kz as f32),
            ]
        });
        let represented =
            modes16.mapv(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()));
        let expected = backend
            .execute_fast_type2_3d(&plan, &represented, &positions)
            .expect("represented fast type2 3D");
        let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];

        backend
            .execute_fast_type2_3d_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &modes16,
                &positions,
                &mut actual,
            )
            .expect("mixed fast type2 3D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let expected_re = f16::from_f32(expected.re as f32);
            let expected_im = f16::from_f32(expected.im as f32);
            assert_eq!(actual[0].to_bits(), expected_re.to_bits());
            assert_eq!(actual[1].to_bits(), expected_im.to_bits());
        }
    }

    #[test]
    fn fast_type2_3d_diagnostics_capture_load_and_ifft_grids_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let modes = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(kx, ky, kz)| {
            Complex32::new(
                0.25 + 0.1 * kx as f32 - 0.05 * ky as f32 + 0.03 * kz as f32,
                -0.4 + 0.07 * kx as f32 + 0.11 * ky as f32 - 0.02 * kz as f32,
            )
        });

        let expected = backend
            .execute_fast_type2_3d(&plan, &modes, &positions)
            .expect("standard fast type2 3D");
        let (actual, diagnostics) = backend
            .execute_fast_type2_3d_with_diagnostics(&plan, &modes, &positions)
            .expect("diagnostic fast type2 3D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 1.0e-6);
        }

        let mx = (grid.nx * plan.oversampling())
            .max(2 * plan.kernel_width() + 1)
            .next_power_of_two();
        let my = (grid.ny * plan.oversampling())
            .max(2 * plan.kernel_width() + 1)
            .next_power_of_two();
        let mz = (grid.nz * plan.oversampling())
            .max(2 * plan.kernel_width() + 1)
            .next_power_of_two();
        let grid_len = mx * my * mz;
        assert_eq!(diagnostics.after_load.re.len(), grid_len);
        assert_eq!(diagnostics.after_load.im.len(), grid_len);
        assert_eq!(diagnostics.after_ifft.re.len(), grid_len);
        assert_eq!(diagnostics.after_ifft.im.len(), grid_len);
        assert!(diagnostics
            .after_load
            .re
            .iter()
            .chain(diagnostics.after_load.im.iter())
            .all(|value| value.is_finite()));
        assert!(diagnostics
            .after_ifft
            .re
            .iter()
            .chain(diagnostics.after_ifft.im.iter())
            .all(|value| value.is_finite()));
        assert!(
            diagnostics
                .after_load
                .re
                .iter()
                .chain(diagnostics.after_load.im.iter())
                .any(|value| value.abs() > 0.0),
            "loaded 3D diagnostic grid must contain deconvolved Fourier coefficients"
        );
        assert!(
            diagnostics
                .after_ifft
                .re
                .iter()
                .chain(diagnostics.after_ifft.im.iter())
                .any(|value| value.abs() > 0.0),
            "3D IFFT diagnostic grid must contain interpolable spatial samples"
        );
    }

    #[test]
    fn type2_3d_rejects_mode_shape_mismatch_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let modes = Array3::from_elem((2, 2, 2), Complex32::new(1.0, 0.0));
        let error = backend
            .execute_type2_3d(&plan, &modes, &[(0.0, 0.0, 0.0)])
            .expect_err("mode shape mismatch must fail");
        assert_eq!(
            error,
            NufftWgpuError::InvalidPlan {
                message: "mode shape must match 3D plan grid dimensions"
            }
        );
    }

    #[test]
    fn type2_3d_matches_cpu_exact_reference_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let modes = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(kx, ky, kz)| {
            Complex32::new(
                0.25 + 0.1 * kx as f32 - 0.05 * ky as f32 + 0.03 * kz as f32,
                -0.4 + 0.07 * kx as f32 + 0.11 * ky as f32 - 0.02 * kz as f32,
            )
        });
        let expected_positions: Vec<(f64, f64, f64)> = positions
            .iter()
            .map(|(x, y, z)| (*x as f64, *y as f64, *z as f64))
            .collect();
        let expected_modes = modes.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
        let expected = nufft_type2_3d(&expected_positions, &expected_modes, grid);

        let actual = backend
            .execute_type2_3d(&plan, &modes, &positions)
            .expect("GPU type2 3D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 1.2e-4);
        }
    }

    #[test]
    fn leto_type2_3d_matches_slice_path_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let modes = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(kx, ky, kz)| {
            Complex32::new(
                0.25 + 0.1 * kx as f32 - 0.05 * ky as f32 + 0.03 * kz as f32,
                -0.4 + 0.07 * kx as f32 + 0.11 * ky as f32 - 0.02 * kz as f32,
            )
        });
        let expected = backend
            .execute_type2_3d(&plan, &modes, &positions)
            .expect("slice type2 3D");
        let leto_positions = leto::Array2::from_shape_vec(
            [positions.len(), 3],
            positions
                .iter()
                .flat_map(|(x, y, z)| [*x, *y, *z])
                .collect(),
        )
        .expect("positions");
        let leto_modes = leto::Array3::from_shape_vec(
            [grid.nx, grid.ny, grid.nz],
            modes.iter().copied().collect(),
        )
        .expect("modes");

        let actual = backend
            .execute_type2_3d_leto(&plan, leto_modes.view(), leto_positions.view())
            .expect("leto type2 3D");

        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 1.0e-6);
        }
    }

    #[test]
    fn type2_3d_typed_mixed_storage_matches_represented_input_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
        let modes16 = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(kx, ky, kz)| {
            [
                f16::from_f32(0.25 + 0.1 * kx as f32 - 0.05 * ky as f32 + 0.03 * kz as f32),
                f16::from_f32(-0.4 + 0.07 * kx as f32 + 0.11 * ky as f32 - 0.02 * kz as f32),
            ]
        });
        let represented =
            modes16.mapv(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()));
        let expected = backend
            .execute_type2_3d(&plan, &represented, &positions)
            .expect("represented type2 3D");
        let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];

        backend
            .execute_type2_3d_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &modes16,
                &positions,
                &mut actual,
            )
            .expect("mixed type2 3D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let expected_re = f16::from_f32(expected.re as f32);
            let expected_im = f16::from_f32(expected.im as f32);
            assert_eq!(actual[0].to_bits(), expected_re.to_bits());
            assert_eq!(actual[1].to_bits(), expected_im.to_bits());
        }
    }

    /// Normalization invariance for fast type-2 1D with a single non-zero
    /// coefficient at k=0 (delta function in Fourier domain).
    ///
    /// Mathematical contract: for f_k = δ(k), the type-2 sum collapses to
    ///   g(x_j) = f_0 × exp(2πi × 0 × x_j / L) = f_0 = 1
    /// for all x_j. The fast (gridded) path approximates this with
    /// deconvolution correction, so the output equals deconv[0]/N at every
    /// position. Any accidental 1/m rescaling would make this constant
    /// deconv[0]/(m) instead, which is off by a factor of σ (=2 with
    /// default oversampling), so this test detects that regression class.
    #[test]
    fn fast_type2_1d_normalization_invariance_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };

        let n = 16;
        let domain = UniformDomain1D::new(n, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(
            domain,
            DEFAULT_NUFFT_OVERSAMPLING,
            DEFAULT_NUFFT_KERNEL_WIDTH,
        );

        // Single non-zero coefficient at k=0: delta function in Fourier domain.
        let mut coefficients = [Complex32::new(0.0, 0.0); 16];
        coefficients[0] = Complex32::new(1.0, 0.0);

        // Positions spanning the domain to verify constancy across all x_j.
        let positions: [f32; 5] = [0.0, 0.5, 1.25, 2.75, 3.875];

        let expected_coefficients: Vec<Complex64> = coefficients
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let expected_positions: Vec<f64> = positions.iter().map(|&x| x as f64).collect();

        // CPU reference via the same fast (gridded) path.
        let expected = nufft_type2_1d_fast(
            &Array1::from_vec(expected_coefficients),
            &expected_positions,
            domain,
            DEFAULT_NUFFT_KERNEL_WIDTH,
        );

        // GPU fast type-2 1D execution.
        let actual = backend
            .execute_fast_type2_1d(&plan, &coefficients, &positions)
            .expect("GPU fast type2 1D with single nonzero coefficient");

        assert_eq!(
            actual.len(),
            expected.len(),
            "output length must match CPU reference"
        );

        // Verify each output element matches the CPU reference.
        // Tolerance 1e-4 accounts for f32 GPU arithmetic rounding.
        for (i, (actual_val, expected_val)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx::abs_diff_eq!(actual_val.re, expected_val.re, epsilon = 1e-4),
                "real mismatch at position index {i}: actual={actual_val:?}, expected={expected_val:?}"
            );
            assert!(
                approx::abs_diff_eq!(actual_val.im, expected_val.im, epsilon = 1e-4),
                "imag mismatch at position index {i}: actual={actual_val:?}, expected={expected_val:?}"
            );
        }

        // Additionally verify the output is approximately constant across all
        // positions (invariant property of k=0 delta input).
        // A 1/m rescaling bug would shift all values by factor σ=2, so this
        // constancy check is a direct regression guard.
        let reference = actual[0];
        for (i, value) in actual.iter().enumerate() {
            assert!(
                approx::abs_diff_eq!(value.re, reference.re, epsilon = 1e-4),
                "constancy regression at position index {i}: {value:?} vs reference {reference:?}"
            );
            assert!(
                approx::abs_diff_eq!(value.im, reference.im, epsilon = 1e-4),
                "constancy regression at position index {i}: {value:?} vs reference {reference:?}"
            );
        }
    }

    #[test]
    fn fast_type2_1d_diagnostics_capture_load_and_ifft_grids_when_device_exists() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let plan = NufftWgpuPlan1D::new(domain, 2, 6);
        let positions = [0.0_f32, 0.25, 0.7, 1.15, 1.8];
        let coefficients = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(-0.75, 0.5),
            Complex32::new(0.25, 0.75),
            Complex32::new(-0.5, -0.1),
            Complex32::new(0.125, 0.25),
            Complex32::new(0.8, -0.6),
            Complex32::new(-0.3, 0.4),
        ];

        let expected = backend
            .execute_fast_type2_1d(&plan, &coefficients, &positions)
            .expect("standard fast type2 1D");
        let (actual, diagnostics) = backend
            .execute_fast_type2_1d_with_diagnostics(&plan, &coefficients, &positions)
            .expect("diagnostic fast type2 1D");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex64_close(*actual, *expected, 1.0e-6);
        }

        let oversampled_len = plan.oversampling() * plan.domain().n;
        assert_eq!(diagnostics.after_load.re.len(), oversampled_len);
        assert_eq!(diagnostics.after_load.im.len(), oversampled_len);
        assert_eq!(diagnostics.after_ifft.re.len(), oversampled_len);
        assert_eq!(diagnostics.after_ifft.im.len(), oversampled_len);
        assert!(diagnostics
            .after_load
            .re
            .iter()
            .chain(diagnostics.after_load.im.iter())
            .all(|value| value.is_finite()));
        assert!(diagnostics
            .after_ifft
            .re
            .iter()
            .chain(diagnostics.after_ifft.im.iter())
            .all(|value| value.is_finite()));
        assert!(
            diagnostics
                .after_load
                .re
                .iter()
                .chain(diagnostics.after_load.im.iter())
                .any(|value| value.abs() > 0.0),
            "loaded diagnostic grid must contain deconvolved Fourier coefficients"
        );
        assert!(
            diagnostics
                .after_ifft
                .re
                .iter()
                .chain(diagnostics.after_ifft.im.iter())
                .any(|value| value.abs() > 0.0),
            "IFFT diagnostic grid must contain interpolable spatial samples"
        );
    }

    fn backend_or_skip() -> Option<NufftWgpuBackend> {
        match NufftWgpuBackend::try_default() {
            Ok(backend) => Some(backend),
            Err(error) => {
                eprintln!("skipping WGPU-dependent NUFFT test: {error}");
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
