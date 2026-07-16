use crate::infrastructure::transport::gpu::{ShtWgpuPlan, WgpuError};
use eunomia::Complex32;
use leto::Array2;

use super::support::backend;

#[test]
fn rejects_under_sampled_bandlimit_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let samples = Array2::from_elem([2, 3], Complex32::new(1.0, 0.0));
    let error = backend
        .execute_forward(&ShtWgpuPlan::new(2, 3, 2), &samples)
        .expect_err("undersampled bandlimit must fail");
    assert!(matches!(error, WgpuError::InvalidPlan { .. }));
}

#[test]
fn reports_sample_shape_mismatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let samples = Array2::from_elem([3, 4], Complex32::new(1.0, 0.0));
    let error = backend
        .execute_forward(&ShtWgpuPlan::new(4, 5, 1), &samples)
        .expect_err("shape mismatch must fail");
    assert!(matches!(error, WgpuError::ShapeMismatch { .. }));
}
