//! Direct three-dimensional Type-1 CPU, Leto, and typed-storage contracts.

use apollo_fft::{f16, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::{Array3, Storage};

use crate::{infrastructure::transport::gpu::NufftWgpuPlan3D, nufft_type1_3d, UniformGrid3D};

use super::support::{assert_complex64_close, backend};

#[test]
fn type1_matches_cpu_exact_reference() {
    let Some(backend) = backend() else {
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
    let expected_positions = positions
        .iter()
        .map(|(x, y, z)| (*x as f64, *y as f64, *z as f64))
        .collect::<Vec<_>>();
    let expected_values = values
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect::<Vec<_>>();
    let expected = nufft_type1_3d(&expected_positions, &expected_values, grid);
    let actual = backend
        .execute_type1_3d(&plan, &positions, &values)
        .expect("GPU type1 3D");
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 8.0e-5);
    }
}

#[test]
fn type1_leto_matches_slice_path() {
    let Some(backend) = backend() else {
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
fn type1_typed_storage_matches_represented_input() {
    let Some(backend) = backend() else {
        return;
    };
    let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let positions = [(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)];
    let values = [
        [f16::from_f32(1.0), f16::from_f32(0.0)],
        [f16::from_f32(-0.25), f16::from_f32(0.5)],
        [f16::from_f32(0.75), f16::from_f32(-0.5)],
    ];
    let represented = values
        .iter()
        .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
        .collect::<Vec<_>>();
    let expected = backend
        .execute_type1_3d(&plan, &positions, &represented)
        .expect("represented type1 3D");
    let mut actual = Array3::from_elem(
        [grid.nx, grid.ny, grid.nz],
        [f16::from_f32(0.0), f16::from_f32(0.0)],
    );
    backend
        .execute_type1_3d_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &positions,
            &values,
            &mut actual,
        )
        .expect("mixed type1 3D");
    assert_eq!(actual.shape(), expected.shape());
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
