//! Shared device availability and assertion contracts for NUFFT verification.

use eunomia::{Complex32, Complex64};
use leto::Array3;

use crate::{
    infrastructure::transport::gpu::{NufftWgpuBackend, NufftWgpuError},
    UniformGrid3D,
};

pub(super) fn backend() -> Option<NufftWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
        "apollo-nufft-wgpu",
        hephaestus_core::DevicePreference::HighPerformance,
        &[],
        NufftWgpuBackend::required_device_limits(),
    ) {
        Ok(device) => Some(NufftWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("NUFFT GPU verification requires a working provider: {error}"),
    }
}

pub(super) fn assert_complex64_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    assert!(
        (actual.re - expected.re).abs() <= tolerance,
        "real mismatch: actual={actual:?}, expected={expected:?}"
    );
    assert!(
        (actual.im - expected.im).abs() <= tolerance,
        "imag mismatch: actual={actual:?}, expected={expected:?}"
    );
}

pub(super) fn assert_input_length_mismatch(error: NufftWgpuError, expected: usize, actual: usize) {
    match error {
        NufftWgpuError::InputLengthMismatch {
            expected: actual_expected,
            actual: actual_actual,
        } => {
            assert_eq!(actual_expected, expected);
            assert_eq!(actual_actual, actual);
        }
        other => panic!("expected input-length mismatch, received {other:?}"),
    }
}

pub(super) fn assert_invalid_plan(error: NufftWgpuError, expected_message: &'static str) {
    match error {
        NufftWgpuError::InvalidPlan { message } => assert_eq!(message, expected_message),
        other => panic!("expected invalid plan, received {other:?}"),
    }
}

pub(super) fn grid3d() -> UniformGrid3D {
    UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid")
}

pub(super) fn positions3d() -> [(f32, f32, f32); 3] {
    [(0.0, 0.0, 0.0), (0.35, 0.7, 0.5), (1.1, 0.2, 1.4)]
}

pub(super) fn type1_values3d() -> [Complex32; 3] {
    [
        Complex32::new(1.0, 0.0),
        Complex32::new(-0.25, 0.5),
        Complex32::new(0.75, -0.5),
    ]
}

pub(super) fn mode_components3d(kx: usize, ky: usize, kz: usize) -> (f32, f32) {
    (
        0.25 + 0.1 * kx as f32 - 0.05 * ky as f32 + 0.03 * kz as f32,
        -0.4 + 0.07 * kx as f32 + 0.11 * ky as f32 - 0.02 * kz as f32,
    )
}

pub(super) fn modes3d(grid: UniformGrid3D) -> Array3<Complex32> {
    Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |[kx, ky, kz]| {
        let (re, im) = mode_components3d(kx, ky, kz);
        Complex32::new(re, im)
    })
}
