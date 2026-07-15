#[cfg(test)]
mod tests {
    use super::super::support::{
        assert_complex64_close, assert_input_length_mismatch, assert_invalid_plan, backend,
    };

    use crate::{
        nufft_type1_1d_fast, nufft_type1_3d_fast, nufft_type2_1d_fast, nufft_type2_3d,
        nufft_type2_3d_fast, UniformDomain1D, UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
        DEFAULT_NUFFT_OVERSAMPLING,
    };
    use apollo_fft::{f16, PrecisionProfile};
    use eunomia::{Complex32, Complex64};
    use leto::Storage;
    use leto::{Array1, Array3};

    use crate::infrastructure::transport::gpu::{
        NufftGpuBuffers1D, NufftGpuBuffers3D, NufftWgpuPlan1D, NufftWgpuPlan3D,
    };

    #[test]
    fn nufft_wgpu_execution_suite_when_device_exists() {
        let Some(backend) = backend() else {
            return;
        };

        // 1. length_mismatch_reports_expected_and_actual
        {
            let domain = UniformDomain1D::new(8, 0.25).expect("domain");
            let plan = NufftWgpuPlan1D::new(domain, 2, 6);
            let error = backend
                .execute_type1_1d(&plan, &[0.0, 0.25], &[Complex32::new(1.0, 0.0)])
                .expect_err("length mismatch must fail");
            assert_input_length_mismatch(error, 2, 1);
        }

        // 2. fast_1d_reusable_buffers_reject_sample_capacity_overflow
        {
            let domain = UniformDomain1D::new(8, 0.25).expect("domain");
            let plan = NufftWgpuPlan1D::new(domain, 2, 6);
            let buffers = NufftGpuBuffers1D::new(backend.device(), 8, 16, 1)
                .expect("provider buffer allocation");
            let error = backend
                .execute_fast_type1_1d_with_buffers(
                    &plan,
                    &buffers,
                    &[0.0, 0.25],
                    &[Complex32::new(1.0, 0.0), Complex32::new(0.5, -0.25)],
                )
                .expect_err("sample capacity overflow must fail");
            assert_input_length_mismatch(error, 1, 2);
        }

        // 3. fast_1d_reusable_type2_supports_more_samples_than_modes
        {
            let domain = UniformDomain1D::new(8, 0.25).expect("domain");
            let plan = NufftWgpuPlan1D::new(domain, 2, 6);
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
            let positions = [
                0.0_f32, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6,
            ];
            let expected = backend
                .execute_fast_type2_1d(&plan, &coefficients, &positions)
                .expect("non-reusable fast type2");
            let buffers = NufftGpuBuffers1D::new(backend.device(), 8, 16, positions.len())
                .expect("provider buffer allocation");
            let actual = backend
                .execute_fast_type2_1d_with_buffers(&plan, &buffers, &coefficients, &positions)
                .expect("reusable fast type2");

            assert_eq!(actual, expected);
        }

        // 4. fast_3d_reusable_buffers_reject_sample_capacity_overflow
        {
            let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
            let plan = NufftWgpuPlan3D::new(grid, 2, 6);
            let buffers = NufftGpuBuffers3D::new(backend.device(), (3, 2, 2), (16, 16, 16), 1)
                .expect("provider buffer allocation");
            let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5)];
            let values = [Complex32::new(1.0, 0.0), Complex32::new(-0.25, 0.5)];
            let error = backend
                .execute_fast_type1_3d_with_buffers(&plan, &buffers, &positions, &values)
                .expect_err("sample capacity overflow must fail");
            assert_input_length_mismatch(error, 1, 2);
        }

        // 8. typed_leto_fast_type1_1d_matches_typed_slice_path
        {
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
                leto::Array1::from_shape_vec([positions.len()], positions.to_vec())
                    .expect("positions");
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

        // 15. fast_type1_1d_matches_cpu_gridded_reference
        {
            let domain = UniformDomain1D::new(8, 0.25).expect("domain");
            let plan = NufftWgpuPlan1D::new(domain, 2, 6);
            let positions = [0.0_f32, 0.25, 0.7, 1.15];
            let values = [
                Complex32::new(1.0, 0.0),
                Complex32::new(0.5, -0.25),
                Complex32::new(-0.75, 0.5),
                Complex32::new(0.25, 0.75),
            ];
            let expected_positions: Vec<f64> =
                positions.iter().map(|value| *value as f64).collect();
            let expected_values: Vec<Complex64> = values
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect();
            let expected = nufft_type1_1d_fast(&expected_positions, &expected_values, domain, 6);

            let actual = backend
                .execute_fast_type1_1d(&plan, &positions, &values)
                .expect("GPU fast type1 1D");

            assert_eq!(actual.size(), expected.size());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_complex64_close(*actual, *expected, 1.5e-3);
            }
        }

        // 16. fast_type1_1d_typed_mixed_storage_matches_represented_input
        {
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

            assert_eq!(actual.len(), expected.size());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                let expected_re = f16::from_f32(expected.re as f32);
                let expected_im = f16::from_f32(expected.im as f32);
                assert_eq!(actual[0].to_bits(), expected_re.to_bits());
                assert_eq!(actual[1].to_bits(), expected_im.to_bits());
            }
        }

        // 17. fast_type2_1d_matches_cpu_gridded_reference
        {
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
            let expected_positions: Vec<f64> =
                positions.iter().map(|value| *value as f64).collect();
            let expected_coefficients: Vec<Complex64> = coefficients
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect();
            let expected = nufft_type2_1d_fast(
                &Array1::from(expected_coefficients),
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

        // 18. fast_type2_1d_typed_mixed_storage_matches_represented_input
        {
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

        // 19. fast_type1_3d_matches_cpu_gridded_reference
        {
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

            assert_eq!(actual.shape(), expected.shape());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_complex64_close(*actual, *expected, 2.0e-3);
            }
        }

        // 20. fast_type1_3d_typed_mixed_storage_matches_represented_input
        {
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
                [grid.nx, grid.ny, grid.nz],
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

            assert_eq!(actual.shape(), expected.shape());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                let expected_re = f16::from_f32(expected.re as f32);
                let expected_im = f16::from_f32(expected.im as f32);
                assert_eq!(actual[0].to_bits(), expected_re.to_bits());
                assert_eq!(actual[1].to_bits(), expected_im.to_bits());
            }
        }

        // 21. fast_type2_3d_matches_cpu_gridded_reference
        {
            let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
            let plan = NufftWgpuPlan3D::new(grid, 2, 6);
            let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
            let modes = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |[kx, ky, kz]| {
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

        // 22. fast_type2_3d_typed_mixed_storage_matches_represented_input
        {
            let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
            let plan = NufftWgpuPlan3D::new(grid, 2, 6);
            let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
            let modes16 = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |[kx, ky, kz]| {
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

        // 23. fast_type2_3d_diagnostics_capture_load_and_ifft_grids
        {
            let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
            let plan = NufftWgpuPlan3D::new(grid, 2, 6);
            let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
            let modes = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |[kx, ky, kz]| {
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

        // 24. type2_3d_rejects_mode_shape_mismatch
        {
            let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
            let plan = NufftWgpuPlan3D::new(grid, 2, 6);
            let modes = Array3::from_elem([2, 2, 2], Complex32::new(1.0, 0.0));
            let error = backend
                .execute_type2_3d(&plan, &modes, &[(0.0, 0.0, 0.0)])
                .expect_err("mode shape mismatch must fail");
            assert_invalid_plan(error, "mode shape must match 3D plan grid dimensions");
        }

        // 25. type2_3d_matches_cpu_exact_reference
        {
            let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
            let plan = NufftWgpuPlan3D::new(grid, 2, 6);
            let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
            let modes = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |[kx, ky, kz]| {
                Complex32::new(
                    0.25 + 0.1 * kx as f32 - 0.05 * ky as f32 + 0.03 * kz as f32,
                    -0.4 + 0.07 * kx as f32 + 0.11 * ky as f32 - 0.02 * kz as f32,
                )
            });
            let expected_positions: Vec<(f64, f64, f64)> = positions
                .iter()
                .map(|(x, y, z)| (*x as f64, *y as f64, *z as f64))
                .collect();
            let expected_modes =
                modes.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
            let expected = nufft_type2_3d(&expected_positions, &expected_modes, grid);

            let actual = backend
                .execute_type2_3d(&plan, &modes, &positions)
                .expect("GPU type2 3D");

            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_complex64_close(*actual, *expected, 1.2e-4);
            }
        }

        // 26. leto_type2_3d_matches_slice_path
        {
            let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
            let plan = NufftWgpuPlan3D::new(grid, 2, 6);
            let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
            let modes = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |[kx, ky, kz]| {
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

        // 27. type2_3d_typed_mixed_storage_matches_represented_input
        {
            let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
            let plan = NufftWgpuPlan3D::new(grid, 2, 6);
            let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
            let modes16 = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |[kx, ky, kz]| {
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

        // 28. fast_type2_1d_normalization_invariance
        {
            let n = 16;
            let domain = UniformDomain1D::new(n, 0.25).expect("domain");
            let plan = NufftWgpuPlan1D::new(
                domain,
                DEFAULT_NUFFT_OVERSAMPLING,
                DEFAULT_NUFFT_KERNEL_WIDTH,
            );

            let mut coefficients = [Complex32::new(0.0, 0.0); 16];
            coefficients[0] = Complex32::new(1.0, 0.0);

            let positions: [f32; 5] = [0.0, 0.5, 1.25, 2.75, 3.875];

            let expected_coefficients: Vec<Complex64> = coefficients
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect();
            let expected_positions: Vec<f64> = positions.iter().map(|&x| x as f64).collect();

            let expected = nufft_type2_1d_fast(
                &Array1::from(expected_coefficients),
                &expected_positions,
                domain,
                DEFAULT_NUFFT_KERNEL_WIDTH,
            );

            let actual = backend
                .execute_fast_type2_1d(&plan, &coefficients, &positions)
                .expect("GPU fast type2 1D with single nonzero coefficient");

            assert_eq!(
                actual.len(),
                expected.len(),
                "output length must match CPU reference"
            );

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

        // 29. fast_type2_1d_diagnostics_capture_load_and_ifft_grids
        {
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
    }
}
