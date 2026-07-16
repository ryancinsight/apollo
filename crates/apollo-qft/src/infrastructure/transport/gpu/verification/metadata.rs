use crate::infrastructure::transport::gpu::{QftWgpuPlan, WgpuCapabilities, WgpuError};

use super::support::backend;

#[test]
fn capabilities_reflect_direct_unitary_kernel_surface() {
    let capabilities = WgpuCapabilities::direct_unitary(true);
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
    let plan = QftWgpuPlan::new(64);
    assert_eq!(plan.len(), 64);
    assert!(!QftWgpuPlan::new(64).is_empty());
    assert!(QftWgpuPlan::new(0).is_empty());
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
