//! Value-semantic Wavelet GPU inverse-law contracts.

use crate::{infrastructure::transport::gpu::WaveletWgpuPlan, DiscreteWavelet, DwtPlan};

use super::support::backend;

#[test]
fn analytical_haar_two_sample_inverse() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = WaveletWgpuPlan::new(2, 1);
    let sqrt2 = std::f32::consts::SQRT_2;
    let coeffs = [sqrt2, sqrt2];
    let out = backend.execute_inverse(&plan, &coeffs).expect("inverse");
    assert_eq!(out.len(), 2);
    assert!((out[0] - 2.0f32).abs() < 1e-5, "x[0]: got {}", out[0]);
    assert!((out[1] - 0.0f32).abs() < 1e-5, "x[1]: got {}", out[1]);
}

#[test]
fn roundtrip_forward_inverse_single_level() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = WaveletWgpuPlan::new(8, 1);
    let signal: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).sin()).collect();
    let coeffs = backend.execute_forward(&plan, &signal).expect("forward");
    let recovered = backend.execute_inverse(&plan, &coeffs).expect("inverse");
    assert_eq!(recovered.len(), signal.len());
    for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (orig - rec).abs() < 1e-5,
            "idx {i}: orig={orig:.8}, rec={rec:.8}"
        );
    }
}

#[test]
fn roundtrip_forward_inverse_multi_level() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = WaveletWgpuPlan::new(8, 3);
    let signal = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let coeffs = backend.execute_forward(&plan, &signal).expect("forward");
    let recovered = backend.execute_inverse(&plan, &coeffs).expect("inverse");
    assert_eq!(recovered.len(), signal.len());
    for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (orig - rec).abs() < 1e-5,
            "idx {i}: orig={orig:.8}, rec={rec:.8}"
        );
    }
}

#[test]
fn forward_preserves_energy_parseval() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = WaveletWgpuPlan::new(16, 2);
    let signal: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();
    let coeffs = backend.execute_forward(&plan, &signal).expect("forward");
    let energy_in: f32 = signal.iter().map(|x| x * x).sum();
    let energy_out: f32 = coeffs.iter().map(|x| x * x).sum();
    assert!(
        (energy_in - energy_out).abs() < energy_in * 1e-5,
        "Parseval: in={energy_in:.8} out={energy_out:.8}"
    );
}

#[test]
fn inverse_matches_cpu_haar_reconstruction_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let signal = vec![0.5_f32, 1.25, -0.75, 2.0, -1.0, 0.25, 1.5, -2.5];
    let plan = WaveletWgpuPlan::new(signal.len(), 3);
    let gpu_coeffs = backend
        .execute_forward(&plan, &signal)
        .expect("wgpu forward Haar DWT");
    let gpu_reconstructed = backend
        .execute_inverse(&plan, &gpu_coeffs)
        .expect("wgpu inverse Haar DWT");

    let cpu_plan = DwtPlan::new(signal.len(), 3, DiscreteWavelet::Haar).expect("cpu Haar DWT plan");
    let cpu_coeffs = cpu_plan
        .forward(
            &signal
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        )
        .expect("cpu forward Haar DWT");
    let cpu_reconstructed = cpu_plan.inverse(&cpu_coeffs).expect("cpu inverse Haar DWT");

    assert_eq!(gpu_reconstructed.len(), cpu_reconstructed.len());
    for (index, (actual, expected)) in gpu_reconstructed
        .iter()
        .zip(cpu_reconstructed.iter())
        .enumerate()
    {
        assert!(
            (f64::from(*actual) - *expected).abs() < 1.0e-5,
            "reconstruction mismatch at index {index}: gpu={actual}, cpu={expected}"
        );
    }
}
