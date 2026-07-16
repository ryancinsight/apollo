use apollo_fft::{f16, PrecisionProfile};
use eunomia::Complex32;

use crate::infrastructure::transport::gpu::{QftWgpuPlan, WgpuError};

use super::support::backend;

#[test]
fn typed_path_rejects_profile_storage_mismatch_when_device_exists() {
    let Ok(backend) = backend() else {
        return;
    };
    let plan = backend.plan(2);
    let input = [
        [f16::from_f32(1.0), f16::from_f32(0.0)],
        [f16::from_f32(-1.0), f16::from_f32(0.5)],
    ];
    let mut output = [[f16::from_f32(0.0); 2]; 2];
    let error = backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::LOW_PRECISION_F32,
            &input,
            &mut output,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(error, WgpuError::InvalidPrecisionProfile));
}

#[test]
fn rejects_invalid_plan_and_length_mismatch_before_dispatch_when_device_exists() {
    let Ok(backend) = backend() else {
        return;
    };
    let invalid_error = backend
        .execute_forward(&QftWgpuPlan::new(0), &[])
        .expect_err("zero-length plan must fail");
    assert!(matches!(invalid_error, WgpuError::InvalidPlan { .. }));

    let mismatch_error = backend
        .execute_forward(
            &QftWgpuPlan::new(4),
            &[Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)],
        )
        .expect_err("length mismatch must fail");
    assert!(matches!(
        mismatch_error,
        WgpuError::LengthMismatch {
            expected: 4,
            actual: 2,
        }
    ));
}
