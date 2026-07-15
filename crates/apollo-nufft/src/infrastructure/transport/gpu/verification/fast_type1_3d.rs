use apollo_fft::{f16, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::Array3;

use crate::infrastructure::transport::gpu::NufftWgpuPlan3D;
use crate::nufft_type1_3d_fast;

use super::support::{assert_complex64_close, backend, grid3d, positions3d, type1_values3d};

#[test]
fn fast_type1_3d_matches_cpu_gridded_reference() {
    let Some(backend) = backend() else {
        return;
    };

    let grid = grid3d();
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let positions = positions3d();
    let values = type1_values3d();
    let expected_positions: Vec<(f64, f64, f64)> = positions
        .iter()
        .map(|(x, y, z)| (*x as f64, *y as f64, *z as f64))
        .collect();
    let expected_values: Vec<Complex64> = values
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
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

#[test]
fn fast_type1_3d_typed_mixed_storage_matches_represented_input() {
    let Some(backend) = backend() else {
        return;
    };

    let grid = grid3d();
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let positions = positions3d();
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
