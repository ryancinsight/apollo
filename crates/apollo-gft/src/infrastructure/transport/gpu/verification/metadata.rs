//! Value-semantic GFT GPU metadata and capability contracts.

use crate::infrastructure::transport::gpu::{GftWgpuPlan, WgpuCapabilities, WgpuError};

use super::support::backend;

#[test]
fn capabilities_reflect_implemented_kernel_surface() {
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
fn plan_preserves_logical_length() {
    let plan = GftWgpuPlan::new(4);
    assert_eq!(plan.len(), 4);
    assert!(!GftWgpuPlan::new(4).is_empty());
    assert!(GftWgpuPlan::new(0).is_empty());
}

#[test]
fn invalid_precision_error_preserves_typed_contract() {
    let error = WgpuError::InvalidPrecisionProfile;
    assert_eq!(
        error.to_string(),
        "precision profile does not match typed GPU storage"
    );
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
