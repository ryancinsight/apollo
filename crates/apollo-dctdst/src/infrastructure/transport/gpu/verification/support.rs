use crate::infrastructure::transport::gpu::DctDstWgpuBackend;
use crate::{DctDstPlan, RealTransformKind};

/// Preserves the established f32 CPU-differential bound for one-dimensional kernels.
pub(super) const ONE_DIMENSIONAL_TOLERANCE: f64 = 1.0e-4;
/// Preserves the established accumulated f32 bound for separable dimensional kernels.
pub(super) const DIMENSIONAL_TOLERANCE: f64 = 1.0e-3;

pub(super) fn backend() -> Option<DctDstWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_default("apollo-dctdst-wgpu") {
        Ok(device) => Some(DctDstWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("DCT/DST GPU verification requires a working provider: {error}"),
    }
}

pub(super) fn cpu_forward(input: &[f32], kind: RealTransformKind) -> Vec<f64> {
    let plan = DctDstPlan::new(input.len(), kind).expect("CPU reference plan");
    plan.forward(&input.iter().copied().map(f64::from).collect::<Vec<_>>())
        .expect("CPU reference forward")
}

pub(super) fn assert_cpu_differential(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (f64::from(*actual) - expected).abs() < tolerance,
            "GPU value {actual} differs from CPU value {expected} by more than {tolerance}"
        );
    }
}

pub(super) fn assert_roundtrip(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < tolerance,
            "roundtrip value {actual} differs from input value {expected} by more than {tolerance}"
        );
    }
}
