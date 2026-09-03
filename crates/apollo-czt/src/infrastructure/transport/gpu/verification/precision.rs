//! Value-semantic CZT GPU precision-rejection contracts.

use apollo_fft::{PrecisionProfile, F16};

use crate::infrastructure::transport::gpu::WgpuError;

use super::support::{backend, reference_parameters};
use crate::infrastructure::transport::gpu::ChirpPlan;

#[test]
fn typed_path_rejects_profile_mismatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (a, w) = reference_parameters();
    let input = vec![
        [F16::from_f32(1.0), F16::from_f32(0.0)],
        [F16::from_f32(-0.5), F16::from_f32(1.0)],
        [F16::from_f32(0.25), F16::from_f32(-0.75)],
        [F16::from_f32(1.25), F16::from_f32(0.5)],
    ];
    let mut output = vec![[F16::from_f32(0.0), F16::from_f32(0.0)]; 6];
    let plan = backend.plan(ChirpPlan::new(input.len(), 6, a, w));
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
