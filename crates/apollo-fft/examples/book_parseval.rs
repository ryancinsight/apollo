//! Parseval's theorem and spectral energy via complex FFT.
//!
//! Parseval's theorem states that the signal energy in the time domain
//! equals the spectral energy divided by `N`:
//! `Σ|x[n]|² = (1/N) Σ|X[k]|²`
//!
//! This example verifies the theorem numerically and shows how to use
//! the complex FFT in-place API.

use apollo_fft::{fft_1d_array, fft_1d_complex_inplace, ifft_1d_complex_inplace};
use eunomia::Complex64;
use leto::Array1;
use std::f64::consts::PI;

fn energy(signal: &Array1<f64>) -> f64 {
    (0..signal.size()).map(|i| signal[[i]] * signal[[i]]).sum()
}

fn spectral_energy(spectrum: &Array1<Complex64>) -> f64 {
    let n = spectrum.size() as f64;
    (0..spectrum.size())
        .map(|i| {
            let c = spectrum[[i]];
            (c.re * c.re + c.im * c.im) / n
        })
        .sum()
}

fn main() {
    let n = 128_usize;
    let dt = 1.0 / 512.0_f64; // 512 Hz sample rate

    // ── Real signal — Parseval check ──
    let signal: Array1<f64> = Array1::from_vec(
        n,
        (0..n)
            .map(|k| (2.0 * PI * 50.0 * k as f64 * dt).sin())
            .collect(),
    )
    .expect("valid signal");

    let spectrum = fft_1d_array(&signal);
    let e_time = energy(&signal);
    let e_freq = spectral_energy(&spectrum);
    println!("real signal: time energy = {e_time:.6}, spectral energy = {e_freq:.6}");
    assert!(
        (e_time - e_freq).abs() / e_time < 1e-10,
        "Parseval violation: time={e_time:.6} freq={e_freq:.6}"
    );

    // ── Complex FFT round-trip (in-place) ──
    let mut c_data: Array1<Complex64> = Array1::from_vec(
        n,
        (0..n)
            .map(|k| Complex64::new((2.0 * PI * 50.0 * k as f64 * dt).cos(), 0.0))
            .collect(),
    )
    .expect("valid complex signal");

    // Save original for comparison.
    let c_original: Vec<Complex64> = (0..n).map(|i| c_data[[i]]).collect();

    fft_1d_complex_inplace(&mut c_data);
    ifft_1d_complex_inplace(&mut c_data);

    let max_err = (0..n)
        .map(|i| {
            let err = c_data[[i]] - c_original[i];
            (err.re * err.re + err.im * err.im).sqrt()
        })
        .fold(0.0_f64, f64::max);
    println!("complex FFT round-trip max error: {max_err:.3e}");
    assert!(max_err < 1e-11, "complex round-trip error {max_err:.3e}");

    println!("Parseval + complex FFT assertions passed");
}
