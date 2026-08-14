//! Forward and inverse real FFT round-trip through Apollo.
//!
//! A real signal of length `N` forward-transforms to a half-spectrum of
//! length `N/2 + 1` (Apollo follows the NumPy/SciPy convention).  The
//! inverse FFT reconstructs the original signal up to floating-point
//! rounding.  This example verifies the round-trip for a known analytical
//! signal and shows how to read back spectral magnitude.

use apollo_fft::{fft_1d_array, fftfreq, ifft_1d_array};
use leto::Array1;

/// Maximum absolute error tolerance for the FFT/IFFT round-trip.
const ROUND_TRIP_TOL: f64 = 1e-11;

fn main() {
    // ── Build an analytical signal: sum of two pure tones ──
    let n = 256_usize;
    let sample_rate = 1000.0_f64; // Hz
    let dt = 1.0 / sample_rate;

    let signal: Array1<f64> = Array1::from_vec(
        n,
        (0..n)
            .map(|k| {
                let t = k as f64 * dt;
                // 100 Hz + 200 Hz
                (2.0 * std::f64::consts::PI * 100.0 * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 200.0 * t).sin()
            })
            .collect(),
    )
    .expect("valid signal");

    // ── Forward FFT ──
    let spectrum = fft_1d_array(&signal);
    println!("signal length    : {}", signal.size());
    println!("spectrum length  : {}", spectrum.size()); // n/2 + 1

    // Frequency grid for the full complex spectrum; only positive half shown.
    let freqs = fftfreq(n, dt);
    println!("frequency bins   : {} (0 Hz .. Nyquist)", freqs.len());

    // ── Find the dominant bin (positive frequencies only) ──
    let half_n = n / 2 + 1;
    let magnitudes: Vec<f64> = (0..half_n)
        .map(|i| {
            let c = spectrum[[i]];
            (c.re * c.re + c.im * c.im).sqrt()
        })
        .collect();
    let (peak_bin, peak_mag) = magnitudes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let peak_freq = freqs[peak_bin];
    println!("dominant bin     : {peak_bin}  ({peak_freq:.1} Hz, magnitude {peak_mag:.4})");
    // 100 Hz should be the dominant component.
    assert!(
        (peak_freq - 100.0).abs() < 5.0,
        "expected dominant frequency near 100 Hz, got {peak_freq:.1}"
    );

    // ── Inverse FFT round-trip ──
    let reconstructed = ifft_1d_array(&spectrum);
    println!("reconstructed length: {}", reconstructed.size());

    let max_err = (0..n)
        .map(|i| (reconstructed[[i]] - signal[[i]]).abs())
        .fold(0.0_f64, f64::max);
    println!("max round-trip error: {max_err:.3e} (tolerance {ROUND_TRIP_TOL:.0e})");
    assert!(
        max_err < ROUND_TRIP_TOL,
        "round-trip error {max_err:.3e} exceeds tolerance {ROUND_TRIP_TOL:.0e}"
    );

    println!("FFT round-trip assertions passed");
}
