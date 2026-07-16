//! Value-semantic STFT GPU verification for one bounded contract.

use crate::infrastructure::transport::gpu::{StftWgpuPlan, WgpuCapabilities, WgpuError};

use super::support::backend;

#[test]
fn capabilities_reflect_forward_only_surface() {
    let caps = WgpuCapabilities::forward_only(true);
    assert!(caps.device_available);
    assert!(caps.supports_forward);
    assert!(!caps.supports_inverse);
    assert!(caps.supports_mixed_precision);
    assert_eq!(
        caps.default_precision_profile,
        apollo_fft::PrecisionProfile::LOW_PRECISION_F32
    );
    let caps_off = WgpuCapabilities::forward_only(false);
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
fn capabilities_reflect_forward_and_inverse_surface() {
    let caps = WgpuCapabilities::forward_and_inverse(true);
    assert!(caps.device_available);
    assert!(caps.supports_forward);
    assert!(caps.supports_inverse);
    assert!(caps.supports_mixed_precision);
    let caps_off = WgpuCapabilities::forward_and_inverse(false);
    assert!(!caps_off.device_available);
    assert!(!caps_off.supports_forward);
    assert!(!caps_off.supports_inverse);
}

#[test]
fn plan_preserves_frame_and_hop_length() {
    let plan = StftWgpuPlan::new(8, 4);
    assert_eq!(plan.frame_len(), 8);
    assert_eq!(plan.hop_len(), 4);
    assert_eq!(plan.len(), 8);
    assert!(!plan.is_empty());
    assert!(StftWgpuPlan::new(0, 4).is_empty());
    assert!(StftWgpuPlan::new(8, 0).is_empty());
}

#[test]
fn unsupported_execution_error_identifies_operation() {
    let err = WgpuError::UnsupportedExecution {
        operation: "inverse",
    };
    assert_eq!(
        err.to_string(),
        "inverse is unsupported by the current WGPU capability set"
    );
}

#[test]
fn stft_wgpu_rejects_invalid_plan() {
    let Some(backend) = backend() else {
        return;
    };
    // zero frame_len
    let r = backend.execute_forward(&StftWgpuPlan::new(0, 4), &[0.0f32; 8]);
    assert!(matches!(r, Err(WgpuError::InvalidPlan { .. })), "{r:?}");

    // zero hop_len
    let r = backend.execute_forward(&StftWgpuPlan::new(8, 0), &[0.0f32; 8]);
    assert!(matches!(r, Err(WgpuError::InvalidPlan { .. })), "{r:?}");

    // hop > frame
    let r = backend.execute_forward(&StftWgpuPlan::new(4, 8), &[0.0f32; 4]);
    assert!(matches!(r, Err(WgpuError::InvalidPlan { .. })), "{r:?}");

    // signal too short
    let r = backend.execute_forward(&StftWgpuPlan::new(8, 4), &[0.0f32; 4]);
    assert!(matches!(r, Err(WgpuError::InputTooShort { .. })), "{r:?}");
}

#[test]
fn stft_wgpu_capabilities() {
    let Some(backend) = backend() else {
        return;
    };
    let caps = backend.capabilities();
    assert!(caps.device_available);
    assert!(caps.supports_forward);
    assert!(caps.supports_inverse);
}
