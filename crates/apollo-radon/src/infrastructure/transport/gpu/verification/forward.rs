//! Value-semantic Radon GPU projection and dispatch-validation contracts.

use crate::{
    infrastructure::transport::gpu::{RadonWgpuPlan, WgpuError},
    RadonPlan,
};

use super::support::backend;

#[test]
fn forward_projection_matches_cpu_reference() {
    let Some(backend) = backend() else {
        return;
    };
    let image = leto::Array2::from_shape_vec(
        [3, 3],
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
    let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
    let gpu = backend
        .execute_forward(&plan, &image, &angles)
        .expect("wgpu forward execution");

    let cpu_plan = RadonPlan::new(
        3,
        3,
        angles.iter().map(|&angle| f64::from(angle)).collect(),
        5,
        1.0,
    )
    .expect("cpu plan");
    let cpu = cpu_plan
        .forward(&image.mapv(f64::from))
        .expect("cpu forward");

    assert_eq!(gpu.shape(), cpu.values().shape());
    for (index, (actual, expected)) in gpu.iter().zip(cpu.values().iter()).enumerate() {
        let error = (f64::from(*actual) - *expected).abs();
        assert!(
            error < 5.0e-4,
            "mismatch at linear index {index}: actual={}, expected={}, error={error}",
            actual,
            expected
        );
    }
}

#[test]
fn rejects_invalid_plan_and_input_shape_before_dispatch() {
    let Some(backend) = backend() else {
        return;
    };
    let empty_plan_err = backend
        .execute_forward(
            &RadonWgpuPlan::new(0, 3, 1, 3, 1.0_f64.to_bits()),
            &leto::Array2::from_shape_vec([1, 1], vec![1.0_f32]).unwrap(),
            &[0.0_f32],
        )
        .expect_err("empty plan must fail");
    assert!(matches!(empty_plan_err, WgpuError::InvalidPlan { .. }));

    let shape_err = backend
        .execute_forward(
            &RadonWgpuPlan::new(3, 3, 1, 3, 1.0_f64.to_bits()),
            &leto::Array2::from_shape_vec([1, 2], vec![1.0_f32, 2.0]).unwrap(),
            &[0.0_f32],
        )
        .expect_err("image shape mismatch must fail");
    assert!(matches!(shape_err, WgpuError::ShapeMismatch { .. }));

    let angle_err = backend
        .execute_forward(
            &RadonWgpuPlan::new(3, 3, 2, 3, 1.0_f64.to_bits()),
            &leto::Array2::from_shape_vec(
                [3, 3],
                vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            )
            .unwrap(),
            &[0.0_f32],
        )
        .expect_err("angle mismatch must fail");
    assert!(matches!(
        angle_err,
        WgpuError::LengthMismatch {
            expected: 2,
            actual: 1,
        }
    ));
}
