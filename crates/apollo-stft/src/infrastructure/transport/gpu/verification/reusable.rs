//! Value-semantic STFT GPU verification for one bounded contract.

use crate::infrastructure::transport::gpu::StftWgpuPlan;
use eunomia::Complex32;

use super::support::backend;

#[test]
fn stft_wgpu_reusable_buffers() {
    let Some(backend) = backend() else {
        return;
    };
    const FRAME_LEN: usize = 512;
    const HOP_LEN: usize = 256;
    const SIGNAL_LEN: usize = 2048;
    const TOL: f32 = 1e-6;

    let signal: Vec<f32> = (0..SIGNAL_LEN)
        .map(|n| (2.0 * std::f32::consts::PI * 16.0 * n as f32 / FRAME_LEN as f32).sin())
        .collect();

    let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
    let frame_count = 1 + SIGNAL_LEN.div_ceil(HOP_LEN);

    // ── Allocating path (reference) ───────────────────────────────────────
    let alloc_fwd = backend
        .execute_forward(&plan, &signal)
        .expect("allocating forward");
    assert_eq!(alloc_fwd.len(), frame_count * FRAME_LEN);

    let alloc_inv = backend
        .execute_inverse(&plan, &alloc_fwd, SIGNAL_LEN)
        .expect("allocating inverse");
    assert_eq!(alloc_inv.len(), SIGNAL_LEN);

    // ── Buffered path ─────────────────────────────────────────────────────
    let mut buffers = backend
        .make_buffers(&plan, SIGNAL_LEN)
        .expect("make_buffers");

    backend
        .execute_forward_with_buffers(&plan, &signal, &mut buffers)
        .expect("buffered forward");
    let buffered_fwd = buffers.fwd_output();
    assert_eq!(buffered_fwd.len(), frame_count * FRAME_LEN);

    // Forward output must match allocating path.
    let max_fwd_err = alloc_fwd
        .iter()
        .zip(buffered_fwd.iter())
        .map(|(a, b)| {
            let re_err = (a.re - b.re).abs();
            let im_err = (a.im - b.im).abs();
            re_err.max(im_err)
        })
        .fold(0.0f32, f32::max);
    assert!(
        max_fwd_err < TOL,
        "forward max error {max_fwd_err:.2e} exceeds tolerance {TOL:.2e}"
    );

    backend
        .execute_inverse_with_buffers(&plan, &alloc_fwd, SIGNAL_LEN, &mut buffers)
        .expect("buffered inverse");
    let buffered_inv = buffers.inv_output();
    assert_eq!(buffered_inv.len(), SIGNAL_LEN);

    // Inverse output must match allocating path.
    let max_inv_err = alloc_inv
        .iter()
        .zip(buffered_inv.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_inv_err < TOL,
        "inverse max error {max_inv_err:.2e} exceeds tolerance {TOL:.2e}"
    );

    // Second call with same buffers: verify buffer reuse (no panic / corruption).
    backend
        .execute_forward_with_buffers(&plan, &signal, &mut buffers)
        .expect("buffered forward second call");
    let buffered_fwd2 = buffers.fwd_output();
    let max_fwd2_err = alloc_fwd
        .iter()
        .zip(buffered_fwd2.iter())
        .map(|(a, b)| {
            let re_err = (a.re - b.re).abs();
            let im_err = (a.im - b.im).abs();
            re_err.max(im_err)
        })
        .fold(0.0f32, f32::max);
    assert!(
        max_fwd2_err < TOL,
        "second-call forward max error {max_fwd2_err:.2e} exceeds tolerance {TOL:.2e}"
    );
}

#[test]
fn stft_wgpu_non_pot_structural() {
    let Some(backend) = backend() else {
        return;
    };
    let signal = vec![0.0f32; 24];
    let plan = StftWgpuPlan::new(6, 3);
    let output = backend
        .execute_forward(&plan, &signal)
        .expect("GPU Chirp-Z structural forward");
    let frame_count = 1 + signal.len().div_ceil(plan.hop_len());
    assert_eq!(output.len(), frame_count * plan.frame_len());
    assert_eq!(output, vec![Complex32::new(0.0, 0.0); output.len()]);
}

#[test]
fn stft_wgpu_make_buffers_non_pot() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = StftWgpuPlan::new(6, 3);
    let signal_len = 24usize;
    let buffers = backend
        .make_buffers(&plan, signal_len)
        .expect("non-PoT buffer allocation");
    let frame_count = 1 + signal_len.div_ceil(plan.hop_len());
    assert_eq!(buffers.frame_count(), frame_count);
    assert_eq!(buffers.frame_len(), plan.frame_len());
    assert_eq!(buffers.hop_len(), plan.hop_len());
    assert_eq!(buffers.signal_len(), signal_len);
    assert_eq!(
        buffers.fwd_output(),
        vec![Complex32::new(0.0, 0.0); frame_count * plan.frame_len()]
    );
    assert_eq!(buffers.inv_output(), vec![0.0_f32; signal_len]);
}

#[test]
fn stft_wgpu_forward_buffers_400() {
    let Some(backend) = backend() else {
        return;
    };
    use crate::StftPlan;

    const FRAME_LEN: usize = 400;
    const HOP_LEN: usize = 200;
    const SIGNAL_LEN: usize = 1000;
    const TOL: f32 = 2e-2;

    let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
        .map(|n| (2.0 * std::f32::consts::PI * 10.0 * n as f32 / FRAME_LEN as f32).sin())
        .collect();
    let signal_f64: leto::Array1<f64> =
        leto::Array1::from(signal_f32.iter().map(|&x| x as f64).collect::<Vec<_>>());

    let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
    let mut buffers = backend
        .make_buffers(&plan, SIGNAL_LEN)
        .expect("make_buffers non-PoT");

    backend
        .execute_forward_with_buffers(&plan, &signal_f32, &mut buffers)
        .expect("forward with buffers");

    let cpu_plan = StftPlan::new(FRAME_LEN, HOP_LEN).expect("CPU plan");
    let cpu_out = cpu_plan.forward(&signal_f64).expect("CPU forward");

    let gpu_spectrum = buffers.fwd_output();
    assert_eq!(gpu_spectrum.len(), cpu_out.size());

    let max_err = gpu_spectrum
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
        "max forward buffered error {max_err:.2e} exceeds tolerance {TOL:.2e}"
    );
}

#[test]
fn stft_wgpu_inverse_buffers_400() {
    let Some(backend) = backend() else {
        return;
    };
    use crate::StftPlan;

    const FRAME_LEN: usize = 400;
    const HOP_LEN: usize = 200;
    const SIGNAL_LEN: usize = 1000;
    const TOL: f32 = 5e-2;
    const INTERIOR_START: usize = FRAME_LEN;
    const INTERIOR_END: usize = SIGNAL_LEN - FRAME_LEN;

    let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
        .map(|n| (2.0 * std::f32::consts::PI * 10.0 * n as f32 / FRAME_LEN as f32).sin())
        .collect();
    let signal_f64: leto::Array1<f64> =
        leto::Array1::from(signal_f32.iter().map(|&x| x as f64).collect::<Vec<_>>());

    let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
    let cpu_plan = StftPlan::new(FRAME_LEN, HOP_LEN).expect("CPU plan");
    let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");
    let spectrum_f32: Vec<Complex32> = cpu_spectrum
        .iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect();

    let mut buffers = backend
        .make_buffers(&plan, SIGNAL_LEN)
        .expect("make_buffers non-PoT");

    backend
        .execute_inverse_with_buffers(&plan, &spectrum_f32, SIGNAL_LEN, &mut buffers)
        .expect("inverse with buffers");

    let recovered = buffers.inv_output().to_vec();
    assert_eq!(recovered.len(), SIGNAL_LEN);

    for i in INTERIOR_START..INTERIOR_END {
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
