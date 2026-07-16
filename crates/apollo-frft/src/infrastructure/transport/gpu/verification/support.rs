use eunomia::{Complex32, Complex64};

use crate::infrastructure::transport::gpu::FrftWgpuBackend;

/// Preserves the established f32 identity bound for standard FrFT execution.
pub(super) const STANDARD_IDENTITY_TOLERANCE: f32 = 1.0e-6;
/// Preserves the established f32 round-trip bound for standard FrFT execution.
pub(super) const STANDARD_ROUNDTRIP_TOLERANCE: f32 = 1.0e-3;
/// Preserves the established f32-to-f64 CPU differential bound.
pub(super) const CPU_DIFFERENTIAL_TOLERANCE: f64 = 1.0e-3;
/// Preserves the established f32 identity and reversal bound for unitary FrFT execution.
pub(super) const UNITARY_VALUE_TOLERANCE: f32 = 1.0e-5;

pub(super) fn backend() -> Option<FrftWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_default("apollo-frft-wgpu") {
        Ok(device) => Some(FrftWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("FrFT GPU verification requires a working provider: {error}"),
    }
}

pub(super) fn cpu_input(input: &[Complex32]) -> leto::Array1<Complex64> {
    leto::Array1::from(
        input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>(),
    )
}

pub(super) fn assert_cpu_differential(actual: &[Complex32], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual.re as f64 - expected.re).abs() < CPU_DIFFERENTIAL_TOLERANCE,
            "k={index} re: gpu={} cpu={}",
            actual.re,
            expected.re
        );
        assert!(
            (actual.im as f64 - expected.im).abs() < CPU_DIFFERENTIAL_TOLERANCE,
            "k={index} im: gpu={} cpu={}",
            actual.im,
            expected.im
        );
    }
}

pub(super) fn assert_complex32_close(
    actual: &[Complex32],
    expected: &[Complex32],
    tolerance: f32,
    contract: &str,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual.re - expected.re).abs() < tolerance,
            "{contract} k={index} re: got={} want={}",
            actual.re,
            expected.re
        );
        assert!(
            (actual.im - expected.im).abs() < tolerance,
            "{contract} k={index} im: got={} want={}",
            actual.im,
            expected.im
        );
    }
}
