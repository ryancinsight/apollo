//! Dense unitary quantum Fourier transform kernel.
//!
//! Forward entry `M[k,j] = exp(2*pi*i*j*k/n) / sqrt(n)`.
//! Inverse entry `M[k,j] = exp(-2*pi*i*j*k/n) / sqrt(n)`.
//! Both maps are unitary (norm-preserving) in exact arithmetic.

use num_complex::Complex64;

/// Forward dense QFT over a contiguous amplitude vector using precomputed twiddle factors.
///
/// `twiddles[k] = exp(2*pi*i*k/n)`. The entry `twiddles[(row*col) % n]` gives
/// `exp(2*pi*i*row*col/n)` without trigonometric calls at transform time.
#[must_use]
pub fn qft_forward_dense(input: &[Complex64], twiddles: &[Complex64]) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); input.len()];
    qft_forward_dense_into(input, &mut output, twiddles);
    output
}

/// Inverse dense QFT over a contiguous amplitude vector using precomputed twiddle factors.
#[must_use]
pub fn qft_inverse_dense(input: &[Complex64], twiddles: &[Complex64]) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); input.len()];
    qft_inverse_dense_into(input, &mut output, twiddles);
    output
}

/// Forward dense QFT into caller-owned output storage.
pub fn qft_forward_dense_into(
    input: &[Complex64],
    output: &mut [Complex64],
    twiddles: &[Complex64],
) {
    qft_dense_into(input, output, twiddles, true);
}

/// Inverse dense QFT into caller-owned output storage.
pub fn qft_inverse_dense_into(
    input: &[Complex64],
    output: &mut [Complex64],
    twiddles: &[Complex64],
) {
    qft_dense_into(input, output, twiddles, false);
}

fn qft_dense_into(
    input: &[Complex64],
    output: &mut [Complex64],
    twiddles: &[Complex64],
    forward: bool,
) {
    let n = input.len();
    assert!(n > 0, "QFT length must be non-zero");
    assert_eq!(output.len(), n, "QFT output length must match input length");
    let scale = 1.0 / (n as f64).sqrt();
    for (row, slot) in output.iter_mut().enumerate() {
        let sum: Complex64 = input
            .iter()
            .enumerate()
            .map(|(col, &value)| {
                let tw = twiddles[(row * col) % n];
                let twiddle = if forward { tw } else { tw.conj() };
                value * twiddle
            })
            .sum();
        *slot = sum * scale;
    }
}
