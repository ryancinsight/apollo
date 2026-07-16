//! Value-semantic STFT GPU verification for one bounded contract.

use crate::infrastructure::transport::gpu::StftWgpuPlan;
use eunomia::Complex32;
use leto::Array1;

use super::support::backend;

#[test]
fn stft_wgpu_inverse_roundtrip_cola() {
    let Some(backend) = backend() else {
        return;
    };
    // Alternating signal, smooth, non-trivial, well inside f32 dynamic range.
    let signal_f32: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
    let signal_f64: Array1<f64> =
        Array1::from(signal_f32.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let signal_len = 8usize;

    let plan = StftWgpuPlan::new(8, 4);

    // Compute spectrum on CPU (f64) as the authoritative input.
    // frame_count = 1 + 8.div_ceil(4) = 3; spectrum_len = 3 * 8 = 24.
    let cpu_plan = crate::StftPlan::new(8, 4).expect("CPU plan");
    let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");

    // Downcast spectrum to f32 for GPU inverse.
    let gpu_spectrum: Vec<Complex32> = cpu_spectrum
        .iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect();

    let recovered = backend
        .execute_inverse(&plan, &gpu_spectrum, signal_len)
        .expect("GPU inverse STFT");

    // CPU inverse as reference for value-semantic comparison.
    let cpu_recovered = cpu_plan
        .inverse(&cpu_spectrum, signal_len)
        .expect("CPU inverse");

    assert_eq!(
        recovered.len(),
        signal_len,
        "recovered length mismatch: got {}, expected {signal_len}",
        recovered.len()
    );
    assert_eq!(
        cpu_recovered.size(),
        signal_len,
        "cpu_recovered length mismatch"
    );

    // Tolerance accounts for f64→f32 downcast of spectrum and f32 GPU arithmetic.
    const TOL: f32 = 5e-4;
    for (i, (gpu_val, cpu_val)) in recovered.iter().zip(cpu_recovered.iter()).enumerate() {
        let err = (gpu_val - *cpu_val as f32).abs();
        assert!(
            err < TOL,
            "mismatch at {i}: gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
        );
    }
}

#[test]
fn stft_wgpu_inverse_matches_cpu() {
    let Some(backend) = backend() else {
        return;
    };
    let signal_f32: Vec<f32> = vec![
        0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
    ];
    let signal_f64: Array1<f64> =
        Array1::from(signal_f32.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let signal_len = 16usize;

    let plan = StftWgpuPlan::new(8, 4);

    // frame_count = 1 + 16.div_ceil(4) = 5; spectrum_len = 5 * 8 = 40.
    let cpu_plan = crate::StftPlan::new(8, 4).expect("CPU plan");
    let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");

    let gpu_spectrum: Vec<Complex32> = cpu_spectrum
        .iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect();

    let recovered = backend
        .execute_inverse(&plan, &gpu_spectrum, signal_len)
        .expect("GPU inverse STFT");

    let cpu_recovered = cpu_plan
        .inverse(&cpu_spectrum, signal_len)
        .expect("CPU inverse");

    assert_eq!(
        recovered.len(),
        signal_len,
        "recovered length mismatch: got {}, expected {signal_len}",
        recovered.len()
    );

    const TOL: f32 = 5e-4;
    for (i, (gpu_val, cpu_val)) in recovered.iter().zip(cpu_recovered.iter()).enumerate() {
        let err = (gpu_val - *cpu_val as f32).abs();
        assert!(
            err < TOL,
            "mismatch at {i}: gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
        );
    }
}

#[test]
fn stft_wgpu_multiple_cola_sets() {
    let Some(backend) = backend() else {
        return;
    };
    let run_roundtrip = |frame_len: usize, hop_len: usize, signal: &[f32]| {
        let signal_len = signal.len();
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let cpu_plan = crate::StftPlan::new(frame_len, hop_len).expect("cpu plan");
        let signal_f64 = Array1::from(signal.iter().map(|&x| x as f64).collect::<Vec<_>>());
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("cpu forward");
        let gpu_spectrum: Vec<Complex32> = cpu_spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();
        let recovered = backend
            .execute_inverse(&plan, &gpu_spectrum, signal_len)
            .unwrap_or_else(|e| {
                panic!("GPU inverse failed for frame_len={frame_len}, hop_len={hop_len}: {e}")
            });
        let cpu_recovered = cpu_plan
            .inverse(&cpu_spectrum, signal_len)
            .expect("cpu inverse");
        assert_eq!(
            recovered.len(),
            signal_len,
            "length mismatch (frame={frame_len}, hop={hop_len})"
        );
        const TOL: f32 = 5e-3;
        for (i, (g, c)) in recovered.iter().zip(cpu_recovered.iter()).enumerate() {
            let err = (g - *c as f32).abs();
            assert!(
                err < TOL,
                "mismatch at sample {i} (frame={frame_len}, hop={hop_len}): \
                 gpu={g:.6}, cpu={c:.6}, err={err:.2e}"
            );
        }
    };

    // Case 1: frame_len=8, hop_len=4 — canonical 50% overlap, single frame.
    run_roundtrip(8, 4, &[0.5_f32, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]);

    // Case 2: frame_len=16, hop_len=8 — 50% overlap, sine wave reference signal.
    let sig16: Vec<f32> = (0..16)
        .map(|i| ((i as f32) * std::f32::consts::FRAC_PI_4).sin())
        .collect();
    run_roundtrip(16, 8, &sig16);

    // Case 3: frame_len=32, hop_len=16 — 50% overlap, cosine reference signal.
    let sig32: Vec<f32> = (0..32)
        .map(|i| ((i as f32) * std::f32::consts::PI / 8.0).cos())
        .collect();
    run_roundtrip(32, 16, &sig32);
}

#[test]
fn stft_wgpu_inverse_non_pot() {
    let Some(backend) = backend() else {
        return;
    };
    let signal = vec![
        0.5_f32, -1.0, 0.25, 0.75, -0.5, 1.5, -0.25, 0.125, 0.875, -0.625, 0.375, -0.125,
    ];
    let plan = StftWgpuPlan::new(6, 3);
    let cpu_plan = crate::StftPlan::new(6, 3).expect("CPU plan");
    let signal_f64 = Array1::from(
        signal
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>(),
    );
    let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU Chirp-Z forward");
    let spectrum = cpu_spectrum
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let actual = backend
        .execute_inverse(&plan, &spectrum, signal.len())
        .expect("GPU Chirp-Z inverse");
    let expected = cpu_plan
        .inverse(&cpu_spectrum, signal.len())
        .expect("CPU Chirp-Z inverse");

    assert_eq!(actual.len(), expected.size());
    // This preserves the established f32 inverse CPU-differential bound.
    const TOL: f32 = 5.0e-4;
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - *expected as f32).abs() < TOL,
            "Chirp-Z inverse mismatch at {index}: actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn stft_wgpu_inverse_large_frame() {
    let Some(backend) = backend() else {
        return;
    };
    const FRAME_LEN: usize = 1024;
    const HOP_LEN: usize = 512;
    const SIGNAL_LEN: usize = 8192;
    const TOL: f32 = 5e-3;

    let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
        .map(|n| (std::f32::consts::TAU * n as f32 / SIGNAL_LEN as f32).sin())
        .collect();

    let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
    let cpu_plan = crate::StftPlan::new(FRAME_LEN, HOP_LEN).expect("cpu plan");
    let signal_f64 = Array1::from(signal_f32.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("cpu forward");
    let gpu_spectrum: Vec<Complex32> = cpu_spectrum
        .iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect();

    let gpu_result = backend
        .execute_inverse(&plan, &gpu_spectrum, SIGNAL_LEN)
        .expect("GPU inverse must succeed for power-of-two frame_len");

    let cpu_reference = cpu_plan
        .inverse(&cpu_spectrum, SIGNAL_LEN)
        .expect("cpu inverse");

    assert_eq!(
        gpu_result.len(),
        SIGNAL_LEN,
        "output length must equal signal_len"
    );
    let max_err = gpu_result
        .iter()
        .zip(cpu_reference.iter())
        .map(|(g, c)| (g - *c as f32).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err <= TOL,
        "max |GPU - CPU_ref| = {max_err:.2e} exceeds TOL = {TOL:.2e} for frame_len={FRAME_LEN}"
    );
}

#[test]
fn stft_wgpu_inverse_chirpz_400() {
    let Some(backend) = backend() else {
        return;
    };
    const FRAME_LEN: usize = 400;
    const HOP_LEN: usize = 200;
    const SIGNAL_LEN: usize = 2000;
    const TOL: f32 = 5e-2;
    const INTERIOR_START: usize = FRAME_LEN;
    const INTERIOR_END: usize = SIGNAL_LEN - FRAME_LEN;

    let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
        .map(|n| (2.0 * std::f32::consts::PI * 10.0 * n as f32 / FRAME_LEN as f32).sin())
        .collect();
    let signal_f64: leto::Array1<f64> =
        leto::Array1::from(signal_f32.iter().map(|&x| x as f64).collect::<Vec<_>>());

    let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);

    let cpu_plan = crate::StftPlan::new(FRAME_LEN, HOP_LEN).expect("CPU plan");
    let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");
    let spectrum_f32: Vec<Complex32> = cpu_spectrum
        .iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect();

    let recovered = backend
        .execute_inverse(&plan, &spectrum_f32, SIGNAL_LEN)
        .expect("GPU inverse Chirp-Z");

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
