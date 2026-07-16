//! Value-semantic SFT GPU metadata and rejection contracts.

use crate::infrastructure::transport::gpu::{SftWgpuPlan, WgpuCapabilities, WgpuError};
use eunomia::Complex32;

use super::support::backend;

#[test]
fn capabilities_advertise_direct_dense_sparse_execution() {
    let capabilities = WgpuCapabilities::direct_dense_spectrum(true);
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
fn plan_preserves_logical_length_and_sparsity() {
    let plan = SftWgpuPlan::new(64, 5);
    assert_eq!(plan.len(), 64);
    assert_eq!(plan.sparsity(), 5);
    assert!(!SftWgpuPlan::new(64, 5).is_empty());
    assert!(SftWgpuPlan::new(0, 5).is_empty());
    assert!(SftWgpuPlan::new(64, 0).is_empty());
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
fn invalid_plan_rejects_zero_length_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let error = backend
        .execute_forward(&SftWgpuPlan::new(0, 1), &[])
        .expect_err("zero length must be invalid");
    assert!(matches!(error, WgpuError::InvalidPlan { .. }));
}

#[test]
fn input_length_mismatch_reports_expected_and_actual_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let error = backend
        .execute_forward(&SftWgpuPlan::new(8, 2), &[Complex32::new(0.0, 0.0); 4])
        .expect_err("mismatched input length must be invalid");
    assert!(matches!(
        error,
        WgpuError::LengthMismatch {
            expected: 8,
            actual: 4
        }
    ));
}
