//! Value-semantic Radon GPU filtered-backprojection contracts.

use crate::{infrastructure::transport::gpu::WgpuError, RadonPlan};
use leto::Array2;

use super::support::backend;

#[test]
fn filtered_backproject_matches_cpu_reference() {
    let Some(backend) = backend() else {
        return;
    };
    let image_f64 = leto::Array2::from_shape_vec(
        [3, 3],
        vec![0.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let angles_f64: Vec<f64> = (0..4)
        .map(|i| i as f64 * std::f64::consts::FRAC_PI_4)
        .collect();
    let angles_f32: Vec<f32> = angles_f64.iter().map(|&a| a as f32).collect();

    let cpu_plan = RadonPlan::new(3, 3, angles_f64, 5, 1.0).expect("cpu plan");
    let cpu_sinogram = cpu_plan.forward(&image_f64).expect("cpu forward");
    let cpu_fbp = cpu_plan
        .filtered_backprojection(&cpu_sinogram)
        .expect("cpu fbp");

    let gpu_plan = backend.plan(3, 3, 4, 5, 1.0);
    let sinogram_f32 = cpu_sinogram.values().mapv(|v| v as f32);
    let gpu_fbp = backend
        .execute_filtered_backproject(&gpu_plan, &sinogram_f32, &angles_f32)
        .expect("gpu fbp");

    assert_eq!(gpu_fbp.shape(), [3, 3]);
    const TOL: f32 = 5e-2;
    for ([r, c], gpu_val) in gpu_fbp.indexed_iter() {
        let cpu_val = cpu_fbp[[r, c]] as f32;
        let err = (gpu_val - cpu_val).abs();
        assert!(
            err < TOL,
            "FBP mismatch at ({r},{c}): gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
        );
    }
}

#[test]
fn filtered_backproject_rejects_sinogram_shape_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = backend.plan(3, 3, 2, 5, 1.0);
    let wrong_sinogram = Array2::<f32>::zeros([3, 5]);
    let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
    let err = backend
        .execute_filtered_backproject(&plan, &wrong_sinogram, &angles)
        .expect_err("sinogram shape mismatch must fail");
    assert!(matches!(err, WgpuError::ShapeMismatch { .. }));
}
