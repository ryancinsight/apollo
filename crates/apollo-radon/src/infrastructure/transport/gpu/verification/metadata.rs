//! Value-semantic Radon GPU metadata and availability contracts.

use crate::infrastructure::transport::gpu::{RadonWgpuPlan, WgpuCapabilities, WgpuError};

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
    let caps = WgpuCapabilities::forward_and_inverse(true);
    assert!(caps.device_available);
    assert!(caps.supports_forward);
    assert!(caps.supports_inverse);
    assert!(caps.supports_mixed_precision);
    assert_eq!(
        caps.default_precision_profile,
        apollo_fft::PrecisionProfile::LOW_PRECISION_F32
    );
    let caps_off = WgpuCapabilities::forward_and_inverse(false);
    assert!(!caps_off.device_available);
    assert!(!caps_off.supports_forward);
    assert!(!caps_off.supports_inverse);
}

#[test]
fn plan_preserves_geometry_configuration() {
    let plan = RadonWgpuPlan::new(8, 9, 3, 11, 0.5_f64.to_bits());
    assert_eq!(plan.rows(), 8);
    assert_eq!(plan.cols(), 9);
    assert_eq!(plan.angle_count(), 3);
    assert_eq!(plan.detector_count(), 11);
    assert_eq!(plan.detector_spacing(), 0.5);
    assert!(!plan.is_empty());
    assert!(RadonWgpuPlan::new(0, 9, 3, 11, 0.5_f64.to_bits()).is_empty());
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
fn capabilities_include_filtered_backprojection() {
    let caps = WgpuCapabilities::forward_inverse_and_fbp(true);
    assert!(caps.device_available);
    assert!(caps.supports_forward);
    assert!(caps.supports_inverse);
    assert!(caps.supports_filtered_backprojection);
    assert!(caps.supports_mixed_precision);
    assert_eq!(
        caps.default_precision_profile,
        apollo_fft::PrecisionProfile::LOW_PRECISION_F32
    );
    let caps_off = WgpuCapabilities::forward_inverse_and_fbp(false);
    assert!(!caps_off.device_available);
    assert!(!caps_off.supports_forward);
    assert!(!caps_off.supports_inverse);
    assert!(!caps_off.supports_filtered_backprojection);
}

#[test]
fn backend_reports_forward_and_backproject() {
    let Some(backend) = backend() else {
        return;
    };
    let capabilities = backend.capabilities();
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
}
