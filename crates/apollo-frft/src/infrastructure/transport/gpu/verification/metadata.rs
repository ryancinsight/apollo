use apollo_fft::PrecisionProfile;

use crate::infrastructure::transport::gpu::{FrftWgpuPlan, WgpuCapabilities, WgpuError};

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
        PrecisionProfile::LOW_PRECISION_F32
    );
}

#[test]
fn available_backend_reports_forward_and_inverse_capabilities() {
    let Some(backend) = backend() else {
        return;
    };
    let capabilities = backend.capabilities();
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
}

#[test]
fn plan_preserves_logical_length() {
    let plan = FrftWgpuPlan::new(64, 1.0_f32);
    assert_eq!(plan.len(), 64);
    assert_eq!(plan.order(), 1.0_f32);
    assert!(!FrftWgpuPlan::new(64, 1.0).is_empty());
    assert!(FrftWgpuPlan::new(0, 1.0).is_empty());
}

#[test]
fn unsupported_execution_error_identifies_operation() {
    let error = WgpuError::UnsupportedExecution {
        operation: "forward",
    };
    assert_eq!(
        error.to_string(),
        "forward is unsupported by the current WGPU capability set"
    );
}
