//! Value-semantic STFT GPU verification for one bounded contract.

use crate::infrastructure::transport::gpu::StftWgpuPlan;
use leto::Array1;

use super::support::backend;

#[test]
fn stft_wgpu_forward_matches_cpu() {
    let Some(backend) = backend() else {
        return;
    };
    // 16-sample alternating signal: values 0, 1, 0, -1, ...
    let signal_f32: Vec<f32> = vec![
        0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
    ];
    let signal_f64: Array1<f64> =
        Array1::from(signal_f32.iter().map(|x| *x as f64).collect::<Vec<_>>());

    let plan = StftWgpuPlan::new(8, 4);

    let gpu_out = backend
        .execute_forward(&plan, &signal_f32)
        .expect("GPU forward STFT");

    let cpu_plan = crate::StftPlan::new(8, 4).expect("CPU plan");
    let cpu_out = cpu_plan.forward(&signal_f64).expect("CPU forward STFT");

    assert_eq!(
        gpu_out.len(),
        cpu_out.size(),
        "output length mismatch: gpu={}, cpu={}",
        gpu_out.len(),
        cpu_out.size()
    );

    const TOL: f32 = 1e-3;
    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let re_err = (g.re - c.re as f32).abs();
        let im_err = (g.im - c.im as f32).abs();
        assert!(
            re_err < TOL,
            "re mismatch at index {i}: gpu={:.6}, cpu={:.6}, err={:.2e}",
            g.re,
            c.re,
            re_err
        );
        assert!(
            im_err < TOL,
            "im mismatch at index {i}: gpu={:.6}, cpu={:.6}, err={:.2e}",
            g.im,
            c.im,
            im_err
        );
    }
}

#[test]
fn stft_wgpu_forward_non_pot() {
    let Some(backend) = backend() else {
        return;
    };
    let signal = vec![
        0.5_f32, -1.0, 0.25, 0.75, -0.5, 1.5, -0.25, 0.125, 0.875, -0.625, 0.375, -0.125,
    ];
    let plan = StftWgpuPlan::new(6, 3);
    let actual = backend
        .execute_forward(&plan, &signal)
        .expect("GPU Chirp-Z forward");
    let cpu_plan = crate::StftPlan::new(6, 3).expect("CPU plan");
    let expected = cpu_plan
        .forward(&Array1::from(
            signal
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        ))
        .expect("CPU Chirp-Z forward");

    assert_eq!(actual.len(), expected.size());
    // This preserves the established f32 CPU-differential bound for small frames.
    const TOL: f32 = 1.0e-3;
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual.re - expected.re as f32).abs() < TOL
                && (actual.im - expected.im as f32).abs() < TOL,
            "Chirp-Z forward mismatch at {index}: actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn stft_wgpu_forward_large_frame() {
    let Some(backend) = backend() else {
        return;
    };
    const FRAME_LEN: usize = 1024;
    const HOP_LEN: usize = 512;
    const SIGNAL_LEN: usize = 4096;
    const TOL: f32 = 1e-2;

    let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
        .map(|n| {
            let t = n as f32;
            (2.0 * std::f32::consts::PI * 16.0 * t / FRAME_LEN as f32).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * 64.0 * t / FRAME_LEN as f32).sin()
        })
        .collect();

    let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
    let frame_count = 1 + SIGNAL_LEN.div_ceil(HOP_LEN);

    let spectrum = backend
        .execute_forward(&plan, &signal_f32)
        .expect("GPU forward FFT");
    assert_eq!(
        spectrum.len(),
        frame_count * FRAME_LEN,
        "spectrum length mismatch"
    );

    let recovered = backend
        .execute_inverse(&plan, &spectrum, SIGNAL_LEN)
        .expect("GPU inverse FFT");
    assert_eq!(recovered.len(), SIGNAL_LEN, "recovered length mismatch");

    let margin = FRAME_LEN / 2;
    for i in margin..(SIGNAL_LEN - margin) {
        let err = (recovered[i] - signal_f32[i]).abs();
        assert!(
            err < TOL,
            "sample {i}: recovered={:.6}, expected={:.6}, err={:.2e}",
            recovered[i],
            signal_f32[i],
            err
        );
    }
}

#[test]
fn stft_wgpu_forward_chirpz_400() {
    let Some(backend) = backend() else {
        return;
    };
    use crate::StftPlan;

    const FRAME_LEN: usize = 400;
    const HOP_LEN: usize = 200;
    const SIGNAL_LEN: usize = 2000;
    const TOL: f32 = 2e-2;

    let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
        .map(|n| (2.0 * std::f32::consts::PI * 10.0 * n as f32 / FRAME_LEN as f32).sin())
        .collect();
    let signal_f64: leto::Array1<f64> =
        leto::Array1::from(signal_f32.iter().map(|&x| x as f64).collect::<Vec<_>>());

    let gpu_plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
    let gpu_out = backend
        .execute_forward(&gpu_plan, &signal_f32)
        .expect("GPU forward Chirp-Z");

    let cpu_plan = StftPlan::new(FRAME_LEN, HOP_LEN).expect("CPU plan");
    let cpu_out = cpu_plan.forward(&signal_f64).expect("CPU forward STFT");

    assert_eq!(
        gpu_out.len(),
        cpu_out.size(),
        "length mismatch: gpu={}, cpu={}",
        gpu_out.len(),
        cpu_out.size()
    );

    let max_err = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| {
            let re = (g.re - c.re as f32).abs();
            let im = (g.im - c.im as f32).abs();
            re.max(im)
        })
        .fold(0.0f32, f32::max);

    assert!(
        max_err < TOL,
        "max Chirp-Z forward error {max_err:.2e} exceeds tolerance {TOL:.2e}"
    );
}
