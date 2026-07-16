//! Value-semantic Radon GPU adjoint-backprojection contracts.

use crate::{infrastructure::transport::gpu::WgpuError, RadonPlan};
use leto::Array2;

use super::support::backend;

#[test]
fn backproject_matches_cpu_reference() {
    let Some(backend) = backend() else {
        return;
    };
    let image_f64 = leto::Array2::from_shape_vec(
        [3, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let angles_f64: Vec<f64> = vec![
        0.0,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
    ];
    let angles_f32: Vec<f32> = angles_f64.iter().map(|&a| a as f32).collect();
    // CPU forward then backproject (f64 reference path).
    let cpu_plan = RadonPlan::new(3, 3, angles_f64.clone(), 7, 1.0).expect("cpu plan");
    let sinogram_cpu = cpu_plan.forward(&image_f64).expect("cpu forward");
    let cpu_bp = cpu_plan
        .backproject(&sinogram_cpu)
        .expect("cpu backproject");
    // GPU backproject using f32 sinogram derived from the f64 CPU result.
    let sinogram_f32 = sinogram_cpu.values().mapv(|v| v as f32);
    let gpu_plan = backend.plan(3, 3, angles_f32.len(), 7, 1.0);
    let gpu_bp = backend
        .execute_inverse(&gpu_plan, &sinogram_f32, &angles_f32)
        .expect("gpu backproject");
    assert_eq!(gpu_bp.shape(), [3, 3]);
    for ([r, c], gpu_val) in gpu_bp.indexed_iter() {
        let cpu_val = cpu_bp[[r, c]] as f32;
        let err = (gpu_val - cpu_val).abs();
        assert!(
            err < 5e-3,
            "mismatch at ({r},{c}): gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
        );
    }
}

#[test]
fn execute_inverse_rejects_sinogram_shape_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = backend.plan(3, 3, 2, 5, 1.0);
    let wrong_sinogram = Array2::<f32>::zeros([3, 5]);
    let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
    let err = backend
        .execute_inverse(&plan, &wrong_sinogram, &angles)
        .expect_err("sinogram shape mismatch must fail");
    assert!(matches!(err, WgpuError::ShapeMismatch { .. }));
}

#[test]
fn backproject_satisfies_adjoint_identity() {
    let Some(backend) = backend() else {
        return;
    };
    let f_f64 = leto::Array2::from_shape_vec(
        [3, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let angles_f64 = vec![
        0.0_f64,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
    ];
    let angles_f32: Vec<f32> = angles_f64.iter().map(|&a| a as f32).collect();

    let cpu_plan = RadonPlan::new(3, 3, angles_f64, 5, 1.0).expect("cpu plan");
    let af = cpu_plan.forward(&f_f64).expect("cpu forward");

    let g = leto::Array2::from_shape_vec(
        [3, 5],
        vec![
            1.0_f32, 0.5, -0.5, -1.0, 0.5, 0.0, 1.0, 0.0, -1.0, 0.0, 0.5, -0.5, 0.5, -0.5, 0.5,
        ],
    )
    .unwrap();

    let lhs: f64 = af
        .values()
        .iter()
        .zip(g.iter())
        .map(|(a, b)| *a * f64::from(*b))
        .sum();

    let gpu_plan = backend.plan(3, 3, 3, 5, 1.0);
    let adj_g = backend
        .execute_inverse(&gpu_plan, &g, &angles_f32)
        .expect("gpu backproject");

    let rhs: f64 = f_f64
        .iter()
        .zip(adj_g.iter())
        .map(|(f, a)| *f * f64::from(*a))
        .sum();

    let magnitude = lhs.abs().max(rhs.abs());
    assert!(
        magnitude > 0.0,
        "inner products must be non-zero for a meaningful test"
    );
    let rel_err = (lhs - rhs).abs() / magnitude;
    assert!(
        rel_err < 5e-3,
        "adjoint identity violated: <Af,g>={lhs:.6}, <f,A†g>={rhs:.6}, rel_err={rel_err:.2e}"
    );
}
