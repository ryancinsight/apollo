//! Value-semantic CZT GPU metadata contracts.

use crate::infrastructure::transport::gpu::{
    Complex32 as GpuComplex32, CztWgpuPlan, WgpuCapabilities,
};

use super::support::backend;

#[test]
fn capabilities_reflect_forward_inverse_kernel_surface() {
    let capabilities = WgpuCapabilities::forward_inverse(true);
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
    let plan = CztWgpuPlan::new(
        64,
        96,
        [1.0_f32.to_bits(), 0.5_f32.to_bits()],
        [0.9_f32.to_bits(), (-0.25_f32).to_bits()],
    );
    assert_eq!(plan.input_len(), 64);
    assert_eq!(plan.output_len(), 96);
    assert_eq!(plan.a(), GpuComplex32::new(1.0, 0.5));
    assert_eq!(plan.w(), GpuComplex32::new(0.9, -0.25));
    assert!(!plan.is_empty());
    assert!(CztWgpuPlan::new(0, 64, [0, 0], [0, 0]).is_empty());
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
