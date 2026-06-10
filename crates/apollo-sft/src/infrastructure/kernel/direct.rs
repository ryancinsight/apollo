//! Direct DFT kernel retained as a mathematical reference.
//!
//! **Note:** This function is O(N^2) and is no longer called in production
//! code. Forward and inverse sparse execution use Apollo FFT kernels
//! (`O(N log N)`) instead. This module is preserved as a ground-truth reference
//! for verification cross-checks.

#[cfg(test)]
use mnemosyne::scratch::ScratchPool;
#[cfg(test)]
use num_complex::Complex64;

/// Below this reference-row length, scalar accumulation avoids Hermes dispatch and scratch setup.
#[cfg(test)]
const DFT_HERMES_ROW_LEN_THRESHOLD: usize = 256;

#[cfg(test)]
thread_local! {
    static DFT_TWIDDLE_LANE_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Execute the dense DFT or inverse DFT over a complex slice.
///
/// # Mathematical contract
///
/// Forward:
/// X_k = sum_n x_n exp(-2*pi*i*k*n/N).
///
/// Inverse:
/// x_n = (1/N) sum_k X_k exp(2*pi*i*k*n/N).
#[must_use]
#[cfg(test)]
pub(crate) fn dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    let mut output = vec![Complex64::new(0.0, 0.0); n];
    if n == 0 {
        return output;
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let tau = std::f64::consts::TAU;
    let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
    let input_lanes = (n >= DFT_HERMES_ROW_LEN_THRESHOLD).then(|| interleaved_lanes(input));

    for (k, out) in output.iter_mut().enumerate() {
        let sum = match &input_lanes {
            Some(lanes) => dft_row_hermes(lanes, k, n, sign, tau),
            None => dft_row_scalar(input, k, sign, tau),
        };
        *out = sum * scale;
    }

    output
}

#[cfg(test)]
fn dft_row_scalar(input: &[Complex64], k: usize, sign: f64, tau: f64) -> Complex64 {
    let n = input.len();
    input
        .iter()
        .enumerate()
        .map(|(n_idx, &x_n)| {
            let angle = sign * tau * (k as f64) * (n_idx as f64) / (n as f64);
            let twiddle = Complex64::new(angle.cos(), angle.sin());
            x_n * twiddle
        })
        .sum()
}

#[cfg(test)]
fn dft_row_hermes(input_lanes: &[f64], k: usize, n: usize, sign: f64, tau: f64) -> Complex64 {
    DFT_TWIDDLE_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(input_lanes.len(), |twiddle_lanes| {
            fill_twiddle_lanes(twiddle_lanes, k, n, sign, tau);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                input_lanes,
                twiddle_lanes,
            )
            .expect("SFT reference DFT Hermes dot uses equal-length interleaved lanes");
            Complex64::new(re, im)
        })
    })
}

#[cfg(test)]
fn interleaved_lanes(values: &[Complex64]) -> Vec<f64> {
    let mut lanes = Vec::with_capacity(values.len() * 2);
    for value in values {
        lanes.push(value.re);
        lanes.push(value.im);
    }
    lanes
}

#[cfg(test)]
fn fill_twiddle_lanes(lanes: &mut [f64], k: usize, n: usize, sign: f64, tau: f64) {
    for (n_idx, lane_pair) in lanes.chunks_exact_mut(2).enumerate() {
        let angle = sign * tau * (k as f64) * (n_idx as f64) / (n as f64);
        lane_pair[0] = angle.cos();
        lane_pair[1] = angle.sin();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermes_dft_row_matches_scalar_formula_at_threshold() {
        let input = (0..DFT_HERMES_ROW_LEN_THRESHOLD)
            .map(|index| {
                Complex64::new(
                    (index as f64 * 0.013).sin() + 0.25,
                    (index as f64 * 0.017).cos() - 0.5,
                )
            })
            .collect::<Vec<_>>();
        let lanes = interleaved_lanes(&input);
        let tau = std::f64::consts::TAU;

        for (k, sign) in [(0, -1.0), (7, -1.0), (13, 1.0), (255, 1.0)] {
            let expected = dft_row_scalar(&input, k, sign, tau);
            let actual = dft_row_hermes(&lanes, k, input.len(), sign, tau);
            assert!((actual.re - expected.re).abs() < 1.0e-10);
            assert!((actual.im - expected.im).abs() < 1.0e-10);
        }
    }

    #[test]
    fn twiddle_lanes_match_dft_formula() {
        let n = DFT_HERMES_ROW_LEN_THRESHOLD;
        let k = 11;
        let sign = -1.0;
        let tau = std::f64::consts::TAU;
        let mut lanes = vec![f64::NAN; n * 2];

        fill_twiddle_lanes(&mut lanes, k, n, sign, tau);

        for (n_idx, lane_pair) in lanes.chunks_exact(2).enumerate() {
            let angle = sign * tau * (k as f64) * (n_idx as f64) / (n as f64);
            assert_eq!(lane_pair[0].to_bits(), angle.cos().to_bits());
            assert_eq!(lane_pair[1].to_bits(), angle.sin().to_bits());
        }
    }
}
