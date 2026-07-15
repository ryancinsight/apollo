//! Direct three-dimensional Type-2 rejection, CPU, Leto, and typed-storage contracts.

use apollo_fft::{f16, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::{Array3, Storage};

use crate::{infrastructure::transport::gpu::NufftWgpuPlan3D, nufft_type2_3d};

use super::support::{
    assert_complex64_close, assert_invalid_plan, backend, grid3d, mode_components3d, modes3d,
    positions3d,
};

#[test]
fn type2_rejects_mode_shape_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = NufftWgpuPlan3D::new(grid3d(), 2, 6);
    let error = backend
        .execute_type2_3d(
            &plan,
            &Array3::from_elem([2, 2, 2], Complex32::new(1.0, 0.0)),
            &[(0.0, 0.0, 0.0)],
        )
        .expect_err("mode shape mismatch must fail");
    assert_invalid_plan(error, "mode shape must match 3D plan grid dimensions");
}

#[test]
fn type2_matches_cpu_exact_reference() {
    let Some(backend) = backend() else {
        return;
    };
    let grid = grid3d();
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let positions = positions3d();
    let modes = modes3d(grid);
    let expected_positions = positions
        .iter()
        .map(|(x, y, z)| (*x as f64, *y as f64, *z as f64))
        .collect::<Vec<_>>();
    let expected = nufft_type2_3d(
        &expected_positions,
        &modes.mapv(|value| Complex64::new(value.re as f64, value.im as f64)),
        grid,
    );
    let actual = backend
        .execute_type2_3d(&plan, &modes, &positions)
        .expect("GPU type2 3D");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 1.2e-4);
    }
}

#[test]
fn type2_leto_matches_slice_path() {
    let Some(backend) = backend() else {
        return;
    };
    let grid = grid3d();
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let positions = positions3d();
    let modes = modes3d(grid);
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
    let leto_modes =
        leto::Array3::from_shape_vec([grid.nx, grid.ny, grid.nz], modes.iter().copied().collect())
            .expect("modes");
    let actual = backend
        .execute_type2_3d_leto(&plan, leto_modes.view(), leto_positions.view())
        .expect("leto type2 3D");
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 1.0e-6);
    }
}

#[test]
fn type2_typed_storage_matches_represented_input() {
    let Some(backend) = backend() else {
        return;
    };
    let grid = grid3d();
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let positions = positions3d();
    let modes = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |[kx, ky, kz]| {
        let (re, im) = mode_components3d(kx, ky, kz);
        [f16::from_f32(re), f16::from_f32(im)]
    });
    let expected = backend
        .execute_type2_3d(
            &plan,
            &modes.mapv(|value| Complex32::new(value[0].to_f32(), value[1].to_f32())),
            &positions,
        )
        .expect("represented type2 3D");
    let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];
    backend
        .execute_type2_3d_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &modes,
            &positions,
            &mut actual,
        )
        .expect("mixed type2 3D");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_eq!(
            actual[0].to_bits(),
            f16::from_f32(expected.re as f32).to_bits()
        );
        assert_eq!(
            actual[1].to_bits(),
            f16::from_f32(expected.im as f32).to_bits()
        );
    }
}
