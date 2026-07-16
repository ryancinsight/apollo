//! Value-semantic CZT GPU precision-rejection contracts.

use apollo_fft::{f16, PrecisionProfile};

use crate::infrastructure::transport::gpu::WgpuError;

use super::support::{backend, reference_parameters};

#[test]
fn typed_path_rejects_profile_mismatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (a, w) = reference_parameters();
    let input = vec![
        [f16::from_f32(1.0), f16::from_f32(0.0)],
        [f16::from_f32(-0.5), f16::from_f32(1.0)],
        [f16::from_f32(0.25), f16::from_f32(-0.75)],
        [f16::from_f32(1.25), f16::from_f32(0.5)],
    ];
    let mut output = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; 6];
    let plan = backend.plan(input.len(), 6, a, w);
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
