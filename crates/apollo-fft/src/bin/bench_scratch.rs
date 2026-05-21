//! Scratch benchmark binary for quick performance probes.

use apollo_fft::fft_1d_complex_inplace;
use ndarray::Array1;
use num_complex::Complex64;
use std::time::Instant;

fn main() {
    let sizes = [60, 84, 90, 150];
    for &n in &sizes {
        let input: Array1<Complex64> = Array1::from_shape_fn(n, |k| {
            Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.29).cos())
        });

        // Warmup
        let mut got = input.clone();
        for _ in 0..100 {
            got.assign(&input);
            fft_1d_complex_inplace(&mut got);
        }

        // Benchmark
        let mut got = input.clone();
        let start = Instant::now();
        let iters = 100_000;
        for _ in 0..iters {
            got.assign(&input);
            fft_1d_complex_inplace(&mut got);
        }
        let elapsed = start.elapsed();
        println!(
            "Size {:3}: {:6.2} ns per iteration",
            n,
            (elapsed.as_secs_f64() * 1e9) / iters as f64
        );
    }
}
