#[cfg(test)]
mod tests {
    use apollo_fft::{f16, PrecisionProfile};
    use eunomia::{Complex32, Complex64};
    use leto::Array3;

    use crate::infrastructure::transport::gpu::NufftWgpuPlan3D;
    use crate::nufft_type2_3d_fast;

    use super::super::support::{
        assert_complex64_close, backend, grid3d, mode_components3d, modes3d, positions3d,
    };

    #[test]
    fn fast_type2_3d_matches_cpu_gridded_reference() {
        let Some(backend) = backend() else {
            return;
        };

        let grid = grid3d();
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = positions3d();
        let modes = modes3d(grid);
        let expected_positions: Vec<(f64, f64, f64)> = positions
            .iter()
            .map(|(x, y, z)| (*x as f64, *y as f64, *z as f64))
            .collect();
        let expected_modes = modes.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
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
    fn fast_type2_3d_typed_mixed_storage_matches_represented_input() {
        let Some(backend) = backend() else {
            return;
        };

        let grid = grid3d();
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = positions3d();
        let modes16 = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |[kx, ky, kz]| {
            let (re, im) = mode_components3d(kx, ky, kz);
            [f16::from_f32(re), f16::from_f32(im)]
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
    fn fast_type2_3d_diagnostics_capture_load_and_ifft_grids() {
        let Some(backend) = backend() else {
            return;
        };

        let grid = grid3d();
        let plan = NufftWgpuPlan3D::new(grid, 2, 6);
        let positions = positions3d();
        let modes = modes3d(grid);
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
}
