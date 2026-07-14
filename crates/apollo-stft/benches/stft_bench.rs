//! Criterion benchmarks for STFT-WGPU forward and inverse FFT-accelerated paths.
//!
//! Measures wall-clock cost (GPU dispatch + PCIe readback) for:
//!   (A) **Allocating paths** — `execute_forward` / `execute_inverse`: all GPU buffers,
//!       staging buffers, and bind groups are created on every call.
//!   (B) **Buffer-reuse paths** — `execute_forward_with_buffers` /
//!       `execute_inverse_with_buffers`: only signal/spectrum data is uploaded per call;
//!       all GPU objects (`StftGpuBuffers`) are pre-allocated once.
//!
//! Parameters cover three `(frame_len, hop_len, signal_len)` sets satisfying
//! hop = frame_len / 2 (Hann COLA condition) at small, medium, and large frame sizes.
//!
//! # Mathematical justification
//!
//! STFT forward: X[m, k] = Σ_{n=0}^{N−1} w_a[n] · x[m·hop − N/2 + n] · exp(−2πi·k·n/N)
//!   N = `frame_len` (power-of-two; Radix-2 DIT path).
//! STFT inverse WOLA: y = OLA(IDFT(X) · w_s) / Σ_m w_s²  (COLA normalisation).
//!
//! Signal is a sum of two bin-aligned sinusoids (k₁=16, k₂=64) ensuring the DFT
//! spectrum is analytically exact (zero spectral leakage); stable, non-trivial workload.

#![allow(missing_docs)]

use apollo_stft::{StftWgpuBackend, StftWgpuPlan};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Parameter sets: `(frame_len, hop_len, signal_len)`.
/// Each hop = frame_len / 2 satisfies the Hann COLA condition.
const PARAMS: &[(usize, usize, usize)] = &[(256, 128, 4096), (512, 256, 8192), (1024, 512, 16384)];

/// Acquire a WGPU backend, or return `None` if no adapter is available.
fn try_backend() -> Option<StftWgpuBackend> {
    StftWgpuBackend::try_default().ok()
}

/// Analytical signal: sum of two bin-aligned sinusoids.
///
/// `s[n] = sin(2π · k₁ · n / N) + 0.5 · sin(2π · k₂ · n / N)`, k₁=16, k₂=64.
/// Bin alignment guarantees exact DFT spectrum (zero spectral leakage).
fn analytical_signal(signal_len: usize, frame_len: usize) -> Vec<f32> {
    (0..signal_len)
        .map(|n| {
            let t = n as f32;
            (2.0 * std::f32::consts::PI * 16.0 * t / frame_len as f32).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * 64.0 * t / frame_len as f32).sin()
        })
        .collect()
}

// ── Allocating path benchmarks ──────────────────────────────────────────────

/// Benchmark the GPU forward STFT allocating path across three frame sizes.
///
/// `execute_forward` creates all GPU buffers, staging buffers, and bind groups
/// per call. Measures end-to-end cost: host→GPU upload, N·log₂N butterfly stages,
/// interleave pass, GPU→host readback.
fn bench_forward_fft(c: &mut Criterion) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device; skipping bench_forward_fft");
        return;
    };

    let mut group = c.benchmark_group("stft_forward_fft");

    for &(frame_len, hop_len, signal_len) in PARAMS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);

        group.bench_with_input(
            BenchmarkId::new("frame_len", frame_len),
            &frame_len,
            |b, _| {
                b.iter(|| {
                    backend
                        .execute_forward(black_box(&plan), black_box(&signal))
                        .expect("GPU forward STFT")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark the GPU inverse STFT allocating path across three frame sizes.
///
/// Spectrum is pre-computed once (outside the benchmark loop) so only the
/// inverse dispatch is measured: deinterleave → bitrev → butterfly×log₂N →
/// scale+window → OLA → readback.
fn bench_inverse_fft(c: &mut Criterion) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device; skipping bench_inverse_fft");
        return;
    };

    let mut group = c.benchmark_group("stft_inverse_fft");

    for &(frame_len, hop_len, signal_len) in PARAMS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);
        // Pre-compute spectrum once; benchmark measures only the inverse path.
        let spectrum = backend
            .execute_forward(&plan, &signal)
            .expect("forward pass for inverse benchmark setup");

        group.bench_with_input(
            BenchmarkId::new("frame_len", frame_len),
            &frame_len,
            |b, _| {
                b.iter(|| {
                    backend
                        .execute_inverse(
                            black_box(&plan),
                            black_box(&spectrum),
                            black_box(signal_len),
                        )
                        .expect("GPU inverse STFT")
                });
            },
        );
    }

    group.finish();
}

// ── Buffer-reuse vs allocating comparison benchmarks ───────────────────────

/// Benchmark the GPU forward STFT: allocating vs `StftGpuBuffers` reuse.
///
/// "allocating": `execute_forward` — per-call: 5–8 `device.create_buffer`,
///   4+ `device.create_bind_group`, and log₂(N) uniform-buffer allocations.
///
/// "with_buffers": `execute_forward_with_buffers` — only signal data is uploaded
///   per call via `queue.write_buffer`; all GPU objects pre-allocated.
fn bench_forward_reuse(c: &mut Criterion) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device; skipping bench_forward_reuse");
        return;
    };

    for &(frame_len, hop_len, signal_len) in PARAMS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);

        let mut group = c.benchmark_group(format!("stft_forward_reuse_fl{frame_len}"));

        // Allocating path: all GPU objects created per dispatch.
        group.bench_function(BenchmarkId::new("allocating", frame_len), |b| {
            b.iter(|| {
                backend
                    .execute_forward(black_box(&plan), black_box(&signal))
                    .expect("allocating forward")
            });
        });

        // Buffer-reuse path: GPU objects pre-allocated; only signal data uploaded.
        let mut buffers = backend
            .make_buffers(&plan, signal_len)
            .expect("make_buffers");
        group.bench_function(BenchmarkId::new("with_buffers", frame_len), |b| {
            b.iter(|| {
                backend
                    .execute_forward_with_buffers(
                        black_box(&plan),
                        black_box(&signal),
                        black_box(&mut buffers),
                    )
                    .expect("buffered forward");
                black_box(buffers.fwd_output());
            });
        });

        group.finish();
    }
}

/// Benchmark the GPU inverse STFT: allocating vs `StftGpuBuffers` reuse.
///
/// "allocating": `execute_inverse` — per-call: all GPU objects created per dispatch.
///
/// "with_buffers": `execute_inverse_with_buffers` — only spectrum data is uploaded
///   per call via `queue.write_buffer`; all GPU objects pre-allocated.
fn bench_inverse_reuse(c: &mut Criterion) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device; skipping bench_inverse_reuse");
        return;
    };

    for &(frame_len, hop_len, signal_len) in PARAMS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);
        // Pre-compute spectrum once; only the inverse path is measured.
        let spectrum = backend
            .execute_forward(&plan, &signal)
            .expect("forward for inverse benchmark setup");

        let mut group = c.benchmark_group(format!("stft_inverse_reuse_fl{frame_len}"));

        // Allocating path: all GPU objects created per dispatch.
        group.bench_function(BenchmarkId::new("allocating", frame_len), |b| {
            b.iter(|| {
                backend
                    .execute_inverse(
                        black_box(&plan),
                        black_box(&spectrum),
                        black_box(signal_len),
                    )
                    .expect("allocating inverse")
            });
        });

        // Buffer-reuse path: GPU objects pre-allocated; only spectrum data uploaded.
        let mut buffers = backend
            .make_buffers(&plan, signal_len)
            .expect("make_buffers");
        group.bench_function(BenchmarkId::new("with_buffers", frame_len), |b| {
            b.iter(|| {
                backend
                    .execute_inverse_with_buffers(
                        black_box(&plan),
                        black_box(&spectrum),
                        black_box(signal_len),
                        black_box(&mut buffers),
                    )
                    .expect("buffered inverse");
                black_box(buffers.inv_output());
            });
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_forward_fft,
    bench_inverse_fft,
    bench_forward_reuse,
    bench_inverse_reuse
);
criterion_main!(benches);
