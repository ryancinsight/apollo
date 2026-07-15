//! Native Apollo benchmarks for provider-backed STFT forward and inverse paths.
//!
//! The parameter matrix satisfies the Hann COLA relation `hop = frame / 2`.
//! Each operation retains its original allocating or reusable-buffer closure.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkSuite};
use apollo_stft::{StftWgpuBackend, StftWgpuPlan};
use std::hint::black_box;

const PARAMETERS: &[(usize, usize, usize)] =
    &[(256, 128, 4096), (512, 256, 8192), (1024, 512, 16384)];

fn try_backend() -> Option<StftWgpuBackend> {
    StftWgpuBackend::try_default().ok()
}

fn analytical_signal(signal_len: usize, frame_len: usize) -> Vec<f32> {
    (0..signal_len)
        .map(|index| {
            let time = index as f32;
            (2.0 * std::f32::consts::PI * 16.0 * time / frame_len as f32).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * 64.0 * time / frame_len as f32).sin()
        })
        .collect()
}

fn bench_forward_fft(suite: &mut BenchmarkSuite) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping STFT forward benchmarks");
        return;
    };

    for &(frame_len, hop_len, signal_len) in PARAMETERS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);
        suite.run(
            BenchmarkCase::new("stft_forward_fft", "frame_len", frame_len),
            || {
                let spectrum = backend
                    .execute_forward(black_box(&plan), black_box(&signal))
                    .expect("GPU forward STFT");
                black_box(spectrum);
            },
        );
    }
}

fn bench_inverse_fft(suite: &mut BenchmarkSuite) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping STFT inverse benchmarks");
        return;
    };

    for &(frame_len, hop_len, signal_len) in PARAMETERS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);
        let spectrum = backend
            .execute_forward(&plan, &signal)
            .expect("forward pass for inverse benchmark setup");
        suite.run(
            BenchmarkCase::new("stft_inverse_fft", "frame_len", frame_len),
            || {
                let output = backend
                    .execute_inverse(
                        black_box(&plan),
                        black_box(&spectrum),
                        black_box(signal_len),
                    )
                    .expect("GPU inverse STFT");
                black_box(output);
            },
        );
    }
}

fn bench_forward_reuse(suite: &mut BenchmarkSuite) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping STFT forward reuse benchmarks");
        return;
    };

    for &(frame_len, hop_len, signal_len) in PARAMETERS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);
        suite.run(
            BenchmarkCase::new(
                format!("stft_forward_reuse_fl{frame_len}"),
                "allocating",
                frame_len,
            ),
            || {
                let spectrum = backend
                    .execute_forward(black_box(&plan), black_box(&signal))
                    .expect("allocating forward");
                black_box(spectrum);
            },
        );

        let mut buffers = backend
            .make_buffers(&plan, signal_len)
            .expect("provider buffer allocation");
        suite.run(
            BenchmarkCase::new(
                format!("stft_forward_reuse_fl{frame_len}"),
                "with_buffers",
                frame_len,
            ),
            || {
                backend
                    .execute_forward_with_buffers(
                        black_box(&plan),
                        black_box(&signal),
                        black_box(&mut buffers),
                    )
                    .expect("buffered forward");
                black_box(buffers.fwd_output());
            },
        );
    }
}

fn bench_inverse_reuse(suite: &mut BenchmarkSuite) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping STFT inverse reuse benchmarks");
        return;
    };

    for &(frame_len, hop_len, signal_len) in PARAMETERS {
        let plan = StftWgpuPlan::new(frame_len, hop_len);
        let signal = analytical_signal(signal_len, frame_len);
        let spectrum = backend
            .execute_forward(&plan, &signal)
            .expect("forward pass for inverse benchmark setup");
        suite.run(
            BenchmarkCase::new(
                format!("stft_inverse_reuse_fl{frame_len}"),
                "allocating",
                frame_len,
            ),
            || {
                let output = backend
                    .execute_inverse(
                        black_box(&plan),
                        black_box(&spectrum),
                        black_box(signal_len),
                    )
                    .expect("allocating inverse");
                black_box(output);
            },
        );

        let mut buffers = backend
            .make_buffers(&plan, signal_len)
            .expect("provider buffer allocation");
        suite.run(
            BenchmarkCase::new(
                format!("stft_inverse_reuse_fl{frame_len}"),
                "with_buffers",
                frame_len,
            ),
            || {
                backend
                    .execute_inverse_with_buffers(
                        black_box(&plan),
                        black_box(&spectrum),
                        black_box(signal_len),
                        black_box(&mut buffers),
                    )
                    .expect("buffered inverse");
                black_box(buffers.inv_output());
            },
        );
    }
}

fn main() {
    let mut suite = BenchmarkSuite::default();
    bench_forward_fft(&mut suite);
    bench_inverse_fft(&mut suite);
    bench_forward_reuse(&mut suite);
    bench_inverse_reuse(&mut suite);
    suite.emit();
}
