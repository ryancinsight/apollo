//! Composite FFT bench
#![allow(missing_docs)]
use num_complex::Complex64;
use std::time::{Duration, Instant};

// Inline the forward_inplace from the mixed_radix dispatch path.
// We invoke the public API via apollo_fft.
fn time_fft<F: Fn(&mut Vec<Complex64>)>(n: usize, f: F, iters: usize) -> Duration {
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.37).sin(), (k as f64 * 0.19).cos()))
        .collect();
    // warm up
    for _ in 0..3 {
        let mut buf = input.clone();
        f(&mut buf);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut buf = input.clone();
        f(&mut buf);
    }
    t0.elapsed() / iters as u32
}

fn main() {
    use apollo_fft::application::execution::kernel::{bluestein, mixed_radix};

    // Test various 5-smooth and non-5-smooth sizes
    for &n in &[
        100usize, 125, 200, 300, 500, 625, 1000, 1250, 2000, 5000, 10000,
    ] {
        let iters = if n <= 100 {
            50000
        } else if n <= 1000 {
            5000
        } else {
            500
        };

        let composite_time = time_fft(
            n,
            |buf| mixed_radix::forward_inplace_64_with_twiddles(buf, None),
            iters,
        );
        let bluestein_time = time_fft(n, |buf| bluestein::forward_inplace_64(buf), iters);

        let speedup = bluestein_time.as_nanos() as f64 / composite_time.as_nanos() as f64;
        println!(
            "N={n:6}: composite={:>8.2}µs  bluestein={:>8.2}µs  speedup={:.1}×",
            composite_time.as_secs_f64() * 1e6,
            bluestein_time.as_secs_f64() * 1e6,
            speedup,
        );
    }
}
