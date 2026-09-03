//! Value-semantic GFT GPU precision-rejection contract.

use apollo_fft::{PrecisionProfile, F16};

use crate::infrastructure::transport::gpu::{GftWgpuPlan, WgpuError};

use super::support::{backend, path4_plan_and_basis};
use crate::infrastructure::transport::gpu::BasisTransform;

#[test]
fn typed_path_rejects_profile_mismatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (_cpu_plan, basis, _) = path4_plan_and_basis();
    let plan = GftWgpuPlan::new(4);
    let input = vec![F16::from_f32(1.0); 4];
    let mut output = vec![F16::from_f32(0.0); 4];
    let error = backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::LOW_PRECISION_F32,
            &input,
            &basis,
            &mut output,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(error, WgpuError::InvalidPrecisionProfile));
}
