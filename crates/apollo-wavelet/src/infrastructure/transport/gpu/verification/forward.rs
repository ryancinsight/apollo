//! Value-semantic Wavelet GPU forward and validation contracts.

use crate::{
    infrastructure::transport::gpu::{WaveletWgpuPlan, WgpuError},
    DiscreteWavelet, DwtPlan,
};

use super::support::backend;

#[test]
fn rejects_invalid_plan_before_dispatch() {
    let Some(backend) = backend() else {
        return;
    };
    let r = backend.execute_forward(&WaveletWgpuPlan::new(6, 1), &[0.0f32; 6]);
    assert!(
        matches!(r, Err(WgpuError::InvalidPlan { .. })),
        "non-pow2: {r:?}"
    );
    let r = backend.execute_forward(&WaveletWgpuPlan::new(4, 0), &[0.0f32; 4]);
    assert!(
        matches!(r, Err(WgpuError::InvalidPlan { .. })),
        "zero levels: {r:?}"
    );
    let r = backend.execute_forward(&WaveletWgpuPlan::new(4, 3), &[0.0f32; 4]);
    assert!(
        matches!(r, Err(WgpuError::InvalidPlan { .. })),
        "levels too large: {r:?}"
    );
    let r = backend.execute_forward(&WaveletWgpuPlan::new(8, 1), &[0.0f32; 4]);
    assert!(
        matches!(r, Err(WgpuError::LengthMismatch { .. })),
        "len mismatch: {r:?}"
    );
}

#[test]
fn analytical_haar_two_sample_forward() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = WaveletWgpuPlan::new(2, 1);
    let signal = [2.0f32, 0.0f32];
    let out = backend.execute_forward(&plan, &signal).expect("forward");
    assert_eq!(out.len(), 2);
    let expected = std::f32::consts::SQRT_2;
    assert!(
        (out[0] - expected).abs() < 1e-5,
        "approx: got {} expected {}",
        out[0],
        expected
    );
    assert!(
        (out[1] - expected).abs() < 1e-5,
        "detail: got {} expected {}",
        out[1],
        expected
    );
}

#[test]
fn forward_matches_cpu_haar_coefficients_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let signal = vec![1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
    let plan = WaveletWgpuPlan::new(signal.len(), 3);
    let gpu = backend
        .execute_forward(&plan, &signal)
        .expect("wgpu forward Haar DWT");

    let cpu_plan = DwtPlan::new(signal.len(), 3, DiscreteWavelet::Haar).expect("cpu Haar DWT plan");
    let cpu_coeffs = cpu_plan
        .forward(
            &signal
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        )
        .expect("cpu forward Haar DWT");
    let mut expected = cpu_coeffs
        .approximation()
        .iter()
        .map(|&value| value as f32)
        .collect::<Vec<_>>();
    for detail in cpu_coeffs.details().iter().rev() {
        expected.extend(detail.iter().map(|&value| value as f32));
    }

    assert_eq!(gpu.len(), expected.len());
    for (index, (actual, expected)) in gpu.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "coefficient mismatch at index {index}: gpu={actual}, cpu={expected}"
        );
    }
}
