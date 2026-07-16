//! Value-semantic Hilbert GPU explicit-precision contracts.

use apollo_fft::{f16, PrecisionProfile};

use crate::infrastructure::transport::gpu::WgpuError;

use super::support::backend;

#[test]
fn typed_path_rejects_profile_mismatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = backend.plan(8);
    let input = vec![f16::from_f32(1.0); 8];
    let mut output = vec![f16::from_f32(0.0); 8];
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
