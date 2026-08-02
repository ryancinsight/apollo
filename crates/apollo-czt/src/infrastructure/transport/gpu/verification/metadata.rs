//! Value-semantic CZT GPU metadata contracts.

use apollo_fft::Complex32 as GpuComplex32;

use crate::infrastructure::transport::gpu::{ChirpPlan, CztWgpuPlan, WgpuCapabilities};

use super::support::backend;

#[test]
fn capabilities_reflect_forward_inverse_kernel_surface() {
    let capabilities = WgpuCapabilities::implemented(true);
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
    assert!(capabilities.supports_mixed_precision);
    assert_eq!(
        capabilities.default_precision_profile,
        apollo_fft::PrecisionProfile::LOW_PRECISION_F32
    );
}

#[test]
fn plan_preserves_logical_parameters() {
    let plan = CztWgpuPlan::new(ChirpPlan::new(
        64,
        96,
        GpuComplex32::new(1.0, 0.5),
        GpuComplex32::new(0.9, -0.25),
    ));
    assert_eq!(plan.len(), 64);
    assert_eq!(plan.output_len(), 96);
    assert_eq!(plan.payload().a(), GpuComplex32::new(1.0, 0.5));
    assert_eq!(plan.payload().w(), GpuComplex32::new(0.9, -0.25));
    assert!(!plan.is_empty());
    assert!(CztWgpuPlan::new(ChirpPlan::new(
        0,
        64,
        GpuComplex32::new(0.0, 0.0),
        GpuComplex32::new(0.0, 0.0)
    ))
    .is_empty());
}

#[test]
fn backend_reports_forward_and_inverse_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let capabilities = backend.capabilities();
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
}
