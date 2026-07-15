use crate::{
    infrastructure::transport::gpu::{DctDstWgpuPlan, WgpuCapabilities},
    RealTransformKind,
};

use super::support::backend;

#[test]
fn capabilities_reflect_full_kernel_surface() {
    let capabilities = WgpuCapabilities::full(true);
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
    assert!(capabilities.supports_dct);
    assert!(capabilities.supports_dst);
    assert!(capabilities.supports_mixed_precision);
    assert_eq!(
        capabilities.default_precision_profile,
        apollo_fft::PrecisionProfile::LOW_PRECISION_F32
    );
}

#[test]
fn plan_preserves_logical_length() {
    let plan = DctDstWgpuPlan::new(64, RealTransformKind::DctII);
    assert_eq!(plan.len(), 64);
    assert_eq!(plan.kind(), RealTransformKind::DctII);
    assert!(!DctDstWgpuPlan::new(64, RealTransformKind::DctIII).is_empty());
    assert!(DctDstWgpuPlan::new(0, RealTransformKind::DctII).is_empty());
}

#[test]
fn available_backend_reports_dct_and_dst() {
    let Some(backend) = backend() else {
        return;
    };
    let capabilities = backend.capabilities();
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
    assert!(capabilities.supports_dct);
    assert!(capabilities.supports_dst);
}
