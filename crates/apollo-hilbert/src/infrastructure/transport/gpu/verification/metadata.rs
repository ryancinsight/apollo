//! Value-semantic Hilbert GPU metadata and rejection contracts.

use crate::infrastructure::transport::gpu::{HilbertWgpuPlan, WgpuCapabilities, WgpuError};

use super::support::backend;

#[test]
fn capabilities_reflect_forward_only_kernel_surface() {
    let capabilities = WgpuCapabilities::forward_only(true);
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(!capabilities.supports_inverse);
    assert!(capabilities.supports_mixed_precision);
    assert_eq!(
        capabilities.default_precision_profile,
        apollo_fft::PrecisionProfile::LOW_PRECISION_F32
    );
}

#[test]
fn capabilities_reflect_forward_and_inverse_surface() {
    let capabilities = WgpuCapabilities::forward_and_inverse(true);
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
    let plan = HilbertWgpuPlan::new(64);
    assert_eq!(plan.len(), 64);
    assert!(!HilbertWgpuPlan::new(64).is_empty());
    assert!(HilbertWgpuPlan::new(0).is_empty());
}

#[test]
fn unsupported_execution_error_identifies_operation() {
    let err = WgpuError::UnsupportedExecution {
        operation: "forward",
    };
    assert_eq!(
        err.to_string(),
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

#[test]
fn rejects_invalid_lengths_before_dispatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let empty_err = backend
        .execute_forward(&HilbertWgpuPlan::new(0), &[])
        .expect_err("empty plan must fail");
    assert!(matches!(empty_err, WgpuError::InvalidPlan { .. }));

    let mismatch_err = backend
        .execute_forward(&HilbertWgpuPlan::new(8), &[0.0; 4])
        .expect_err("length mismatch must fail");
    assert!(matches!(
        mismatch_err,
        WgpuError::LengthMismatch {
            expected: 8,
            actual: 4,
        }
    ));
}
