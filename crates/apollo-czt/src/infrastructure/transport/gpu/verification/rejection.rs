//! Value-semantic CZT GPU pre-dispatch rejection contracts.

use eunomia::Complex32;

use crate::infrastructure::transport::gpu::{CztWgpuPlan, WgpuError};

use super::support::backend;

#[test]
fn rejects_invalid_lengths_and_parameters_before_dispatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let empty_err = backend
        .execute_forward(&CztWgpuPlan::new(0, 5, [0, 0], [0, 0]), &[])
        .expect_err("empty plan must fail");
    assert!(matches!(empty_err, WgpuError::InvalidPlan { .. }));

    let mismatch_err = backend
        .execute_forward(
            &CztWgpuPlan::new(
                8,
                8,
                [1.0_f32.to_bits(), 0.0_f32.to_bits()],
                [1.0_f32.to_bits(), 0.0_f32.to_bits()],
            ),
            &[Complex32::new(0.0, 0.0); 4],
        )
        .expect_err("length mismatch must fail");
    assert!(matches!(
        mismatch_err,
        WgpuError::LengthMismatch {
            expected: 8,
            actual: 4,
        }
    ));

    let plan = CztWgpuPlan::new(
        4,
        5,
        [1.0_f32.to_bits(), 0.0_f32.to_bits()],
        [1.0_f32.to_bits(), 0.0_f32.to_bits()],
    );
    let output_err = backend
        .execute_forward_into(
            &plan,
            &[Complex32::new(0.0, 0.0); 4],
            &mut [Complex32::new(0.0, 0.0); 4],
        )
        .expect_err("output length mismatch must fail");
    assert!(matches!(
        output_err,
        WgpuError::LengthMismatch {
            expected: 5,
            actual: 4,
        }
    ));

    let invalid_param_err = backend
        .execute_forward(
            &CztWgpuPlan::new(
                4,
                4,
                [0.0_f32.to_bits(), 0.0_f32.to_bits()],
                [1.0_f32.to_bits(), 0.0_f32.to_bits()],
            ),
            &[Complex32::new(0.0, 0.0); 4],
        )
        .expect_err("zero a must fail");
    assert!(matches!(invalid_param_err, WgpuError::InvalidPlan { .. }));
}
