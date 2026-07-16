//! Value-semantic Wavelet GPU metadata and availability contracts.

use crate::infrastructure::transport::gpu::{WaveletWgpuPlan, WgpuCapabilities, WgpuError};

use super::support::backend;

#[test]
fn capabilities_reflect_forward_and_inverse() {
    let caps = WgpuCapabilities::implemented(true);
    assert!(caps.device_available);
    assert!(caps.supports_forward);
    assert!(caps.supports_inverse);
    assert!(caps.supports_mixed_precision);
    assert_eq!(
        caps.default_precision_profile,
        apollo_fft::PrecisionProfile::LOW_PRECISION_F32
    );
    let caps_off = WgpuCapabilities::implemented(false);
    assert!(!caps_off.device_available);
    assert!(!caps_off.supports_forward);
    assert!(!caps_off.supports_inverse);
    assert!(caps_off.supports_mixed_precision);
    assert_eq!(
        caps_off.default_precision_profile,
        apollo_fft::PrecisionProfile::LOW_PRECISION_F32
    );
}

#[test]
fn plan_preserves_len_and_levels() {
    let plan = WaveletWgpuPlan::new(64, 3);
    assert_eq!(plan.len(), 64);
    assert_eq!(plan.levels(), 3);
    assert!(!plan.is_empty());
    assert!(WaveletWgpuPlan::new(0, 3).is_empty());
    assert!(WaveletWgpuPlan::new(64, 0).is_empty());
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
    let caps = backend.capabilities();
    assert!(caps.device_available);
    assert!(caps.supports_forward);
    assert!(caps.supports_inverse);
}
