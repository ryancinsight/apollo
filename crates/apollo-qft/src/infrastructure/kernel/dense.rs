//! Dense unitary quantum Fourier transform kernel.
//!
//! Forward entry `M[k,j] = exp(2*pi*i*j*k/n) / sqrt(n)`.
//! Inverse entry `M[k,j] = exp(-2*pi*i*j*k/n) / sqrt(n)`.
//! Both maps are unitary (norm-preserving) in exact arithmetic.

use mnemosyne::scratch::ScratchPool;
use moirai::ParallelSliceMut;
use num_complex::Complex64;

/// Below this operation count, serial loops avoid parallel scheduling overhead.
const QFT_PAR_OP_THRESHOLD: usize = 16_384;

thread_local! {
    static QFT_TWIDDLE_LANE_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

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
        let input_lanes = interleaved_lanes(input);
        output.par_mut().enumerate(|row, slot| {
            *slot = qft_row_hermes(&input_lanes, twiddles, row, direction, scale);
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

fn qft_row_hermes(
    input_lanes: &[f64],
    twiddles: &[Complex64],
    row: usize,
    direction: QftDirection,
    scale: f64,
) -> Complex64 {
    QFT_TWIDDLE_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(input_lanes.len(), |twiddle_lanes| {
            fill_twiddle_lanes(twiddle_lanes, twiddles, row, direction);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                input_lanes,
                twiddle_lanes,
            )
            .expect("QFT Hermes dot uses equal-length interleaved complex lane slices");
            Complex64::new(re * scale, im * scale)
        })
    })
}

fn interleaved_lanes(values: &[Complex64]) -> Vec<f64> {
    let mut lanes = Vec::with_capacity(values.len() * 2);
    for value in values {
        lanes.push(value.re);
        lanes.push(value.im);
    }
    lanes
}

fn fill_twiddle_lanes(
    lanes: &mut [f64],
    twiddles: &[Complex64],
    row: usize,
    direction: QftDirection,
) {
    let n = twiddles.len();
    for (col, lane_pair) in lanes.chunks_exact_mut(2).enumerate() {
        let tw = twiddles[(row * col) % n];
        let twiddle = match direction {
            QftDirection::Forward => tw,
            QftDirection::Inverse => tw.conj(),
        };
        lane_pair[0] = twiddle.re;
        lane_pair[1] = twiddle.im;
    }
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

    #[test]
    fn hermes_forward_row_matches_scalar_formula_at_threshold() {
        let n = 128;
        let input = input(n);
        let twiddles = twiddles(n);
        let input_lanes = interleaved_lanes(&input);
        let scale = 1.0 / (n as f64).sqrt();

        for row in [0usize, 1, 17, 64, 127] {
            let actual = qft_row_hermes(&input_lanes, &twiddles, row, QftDirection::Forward, scale);
            let expected = qft_row(&input, &twiddles, row, QftDirection::Forward, scale);
            assert_eq!(actual.re.to_bits(), expected.re.to_bits(), "row={row}");
            assert_eq!(actual.im.to_bits(), expected.im.to_bits(), "row={row}");
        }
    }

    #[test]
    fn hermes_inverse_row_matches_scalar_formula_at_threshold() {
        let n = 128;
        let input = input(n);
        let twiddles = twiddles(n);
        let input_lanes = interleaved_lanes(&input);
        let scale = 1.0 / (n as f64).sqrt();

        for row in [0usize, 1, 17, 64, 127] {
            let actual = qft_row_hermes(&input_lanes, &twiddles, row, QftDirection::Inverse, scale);
            let expected = qft_row(&input, &twiddles, row, QftDirection::Inverse, scale);
            assert_eq!(actual.re.to_bits(), expected.re.to_bits(), "row={row}");
            assert_eq!(actual.im.to_bits(), expected.im.to_bits(), "row={row}");
        }
    }

    #[test]
    fn twiddle_lanes_match_direction_formula() {
        let n = 8;
        let twiddles = twiddles(n);
        let mut lanes = vec![0.0; n * 2];

        fill_twiddle_lanes(&mut lanes, &twiddles, 3, QftDirection::Inverse);

        for (col, lane_pair) in lanes.chunks_exact(2).enumerate() {
            let expected = twiddles[(3 * col) % n].conj();
            assert_eq!(lane_pair[0].to_bits(), expected.re.to_bits(), "col={col}");
            assert_eq!(lane_pair[1].to_bits(), expected.im.to_bits(), "col={col}");
        }
    }
}
