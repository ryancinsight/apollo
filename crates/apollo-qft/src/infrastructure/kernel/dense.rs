//! Dense unitary quantum Fourier transform kernel.
//!
//! Forward entry `M[k,j] = exp(2*pi*i*j*k/n) / sqrt(n)`.
//! Inverse entry `M[k,j] = exp(-2*pi*i*j*k/n) / sqrt(n)`.
//! Both maps are unitary (norm-preserving) in exact arithmetic.

use moirai::ParallelSliceMut;
use num_complex::Complex64;

/// Below this operation count, serial loops avoid parallel scheduling overhead.
const QFT_PAR_OP_THRESHOLD: usize = 16_384;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QftDirection {
    Forward,
    Inverse,
}

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
    qft_dense_into(input, output, twiddles, QftDirection::Forward);
}

/// Inverse dense QFT into caller-owned output storage.
pub fn qft_inverse_dense_into(
    input: &[Complex64],
    output: &mut [Complex64],
    twiddles: &[Complex64],
) {
    qft_dense_into(input, output, twiddles, QftDirection::Inverse);
}

fn qft_dense_into(
    input: &[Complex64],
    output: &mut [Complex64],
    twiddles: &[Complex64],
    direction: QftDirection,
) {
    let n = input.len();
    assert!(n > 0, "QFT length must be non-zero");
    assert_eq!(output.len(), n, "QFT output length must match input length");
    let scale = 1.0 / (n as f64).sqrt();
    let work_items = n.saturating_mul(n);
    if work_items >= QFT_PAR_OP_THRESHOLD {
        output.par_mut().enumerate(|row, slot| {
            *slot = qft_row(input, twiddles, row, direction, scale);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(row, slot)| {
            *slot = qft_row(input, twiddles, row, direction, scale);
        });
    }
}

fn qft_row(
    input: &[Complex64],
    twiddles: &[Complex64],
    row: usize,
    direction: QftDirection,
    scale: f64,
) -> Complex64 {
    let n = input.len();
    let sum: Complex64 = input
        .iter()
        .enumerate()
        .map(|(col, &value)| {
            let tw = twiddles[(row * col) % n];
            let twiddle = match direction {
                QftDirection::Forward => tw,
                QftDirection::Inverse => tw.conj(),
            };
            value * twiddle
        })
        .sum();
    sum * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn twiddles(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let angle = std::f64::consts::TAU * k as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect()
    }

    fn input(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|index| Complex64::new((index as f64 * 0.17).sin(), (index as f64 * 0.23).cos()))
            .collect()
    }

    #[test]
    fn moirai_parallel_forward_matches_row_formula_at_threshold() {
        let n = 128;
        let input = input(n);
        let twiddles = twiddles(n);
        let actual = qft_forward_dense(&input, &twiddles);
        let scale = 1.0 / (n as f64).sqrt();

        for (row, value) in actual.iter().enumerate() {
            let expected = qft_row(&input, &twiddles, row, QftDirection::Forward, scale);
            assert_eq!(value.re.to_bits(), expected.re.to_bits());
            assert_eq!(value.im.to_bits(), expected.im.to_bits());
        }
    }

    #[test]
    fn moirai_parallel_inverse_matches_row_formula_at_threshold() {
        let n = 128;
        let input = input(n);
        let twiddles = twiddles(n);
        let actual = qft_inverse_dense(&input, &twiddles);
        let scale = 1.0 / (n as f64).sqrt();

        for (row, value) in actual.iter().enumerate() {
            let expected = qft_row(&input, &twiddles, row, QftDirection::Inverse, scale);
            assert_eq!(value.re.to_bits(), expected.re.to_bits());
            assert_eq!(value.im.to_bits(), expected.im.to_bits());
        }
    }
}
