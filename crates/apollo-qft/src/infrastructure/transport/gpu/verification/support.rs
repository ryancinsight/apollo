use crate::infrastructure::transport::gpu::{QftWgpuBackend, WgpuResult};
use eunomia::{Complex32, Complex64};
use leto::Array1;

pub(super) const CPU_DIFFERENTIAL_TOLERANCE: f64 = 2.0e-4;
pub(super) const ROUNDTRIP_TOLERANCE: f32 = 5.0e-4;

pub(super) fn backend() -> WgpuResult<QftWgpuBackend> {
    hephaestus_wgpu::WgpuDevice::try_default("apollo-qft-wgpu")
        .map(QftWgpuBackend::new)
        .map_err(Into::into)
}

pub(super) fn forward_input() -> Vec<Complex32> {
    vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(-0.5, 0.75),
        Complex32::new(0.25, -1.25),
        Complex32::new(2.0, 0.5),
    ]
}

pub(super) fn inverse_input() -> Vec<Complex32> {
    vec![
        Complex32::new(0.25, -0.5),
        Complex32::new(1.0, 1.5),
        Complex32::new(-2.0, 0.25),
        Complex32::new(0.75, -1.0),
    ]
}

pub(super) fn roundtrip_input() -> Vec<Complex32> {
    vec![
        Complex32::new(0.5, -0.25),
        Complex32::new(-1.25, 0.75),
        Complex32::new(2.0, 1.0),
        Complex32::new(-0.5, -1.5),
    ]
}

pub(super) fn cpu_input(input: &[Complex32]) -> Array1<Complex64> {
    Array1::from(
        input
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect::<Vec<_>>(),
    )
}

pub(super) fn assert_matches_cpu(
    actual: &[Complex32],
    expected: &Array1<Complex64>,
    operation: &str,
) {
    assert_eq!(actual.len(), expected.size());
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let real_error = (f64::from(actual_value.re) - expected_value.re).abs();
        let imag_error = (f64::from(actual_value.im) - expected_value.im).abs();
        assert!(
            real_error < CPU_DIFFERENTIAL_TOLERANCE && imag_error < CPU_DIFFERENTIAL_TOLERANCE,
            "{operation} mismatch at index {index}: actual=({},{}) expected=({},{}) real_error={} imag_error={}",
            actual_value.re,
            actual_value.im,
            expected_value.re,
            expected_value.im,
            real_error,
            imag_error
        );
    }
}
