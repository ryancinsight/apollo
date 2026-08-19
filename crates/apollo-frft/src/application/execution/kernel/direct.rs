use eunomia::Complex64;
use mnemosyne::scratch::ScratchPool;
use moirai::ParallelSliceMut;
use std::f64::consts::PI;

/// Below this O(N²) operation count, serial loops avoid parallel scheduling overhead.
const FRFT_PAR_OP_THRESHOLD: usize = 16_384;

thread_local! {
    static DIRECT_WEIGHT_LANE_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Evaluate the direct $O(N^2)$ fractional Fourier kernel on a centered grid into a user-provided buffer.
#[allow(clippy::similar_names)]
pub fn direct_frft_forward_into(
    input: &[Complex64],
    output: &mut [Complex64],
    order: f64,
    cot: f64,
    csc: f64,
    scale: Complex64,
) {
    let n = input.len();
    if n == 0 {
        return;
    }
    assert_eq!(n, output.len(), "FrFT output buffer length mismatch");

    let reduced = order.rem_euclid(4.0);
    if (reduced - 0.0).abs() < 1.0e-12 {
        output.copy_from_slice(input);
        return;
    }
    if (reduced - 2.0).abs() < 1.0e-12 {
        for (i, &val) in input.iter().enumerate() {
            output[n - 1 - i] = val;
        }
        return;
    }

    let sign = if (reduced - 3.0).abs() < 1.0e-12 {
        1.0
    } else {
        -1.0
    };

    let center = (n as f64 - 1.0) * 0.5;
    let work_items = n.saturating_mul(n);

    if (reduced - 1.0).abs() < 1.0e-12 || (reduced - 3.0).abs() < 1.0e-12 {
        let scale = 1.0 / (n as f64).sqrt();
        if work_items >= FRFT_PAR_OP_THRESHOLD {
            let input_lanes = interleaved_lanes(input);
            output.par_mut().enumerate(|k, out| {
                *out = centered_dft_row_hermes(input_lanes, n, center, sign, scale, k);
            });
        } else {
            output.iter_mut().enumerate().for_each(|(k, out)| {
                *out = centered_dft_row(input, n, center, sign, scale, k);
            });
        }
        return;
    }

    if work_items >= FRFT_PAR_OP_THRESHOLD {
        let input_lanes = interleaved_lanes(input);
        output.par_mut().enumerate(|k, out| {
            *out = fractional_row_hermes(input_lanes, n, center, cot, csc, scale, k);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            *out = fractional_row(input, n, center, cot, csc, scale, k);
        });
    }
}

#[inline]
fn centered_dft_row(
    input: &[Complex64],
    n: usize,
    center: f64,
    sign: f64,
    scale: f64,
    k: usize,
) -> Complex64 {
    let u = k as f64 - center;
    let sum = input
        .iter()
        .enumerate()
        .map(|(j, &value)| {
            let x = j as f64 - center;
            let angle = sign * 2.0 * PI * (x * u) / n as f64;
            value * Complex64::new(angle.cos(), angle.sin())
        })
        .sum::<Complex64>();
    sum * scale
}

#[inline]
fn centered_dft_row_hermes(
    input_lanes: &[f64],
    n: usize,
    center: f64,
    sign: f64,
    scale: f64,
    k: usize,
) -> Complex64 {
    let u = k as f64 - center;
    DIRECT_WEIGHT_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(input_lanes.len(), |weight_lanes| {
            fill_centered_dft_weight_lanes(weight_lanes, n, center, sign, u);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                input_lanes,
                weight_lanes,
            )
            .expect("FRFT centered DFT Hermes dot uses equal-length interleaved lanes");
            Complex64::new(re * scale, im * scale)
        })
    })
}

#[inline]
fn fractional_row(
    input: &[Complex64],
    n: usize,
    center: f64,
    cot: f64,
    csc: f64,
    scale: Complex64,
    k: usize,
) -> Complex64 {
    let u = k as f64 - center;
    let sum = input
        .iter()
        .enumerate()
        .map(|(j, &value)| {
            let x = j as f64 - center;
            let phase = PI * ((x * x + u * u) * cot - 2.0 * x * u * csc) / n as f64;
            value * Complex64::new(phase.cos(), phase.sin())
        })
        .sum::<Complex64>();
    sum * scale
}

#[inline]
fn fractional_row_hermes(
    input_lanes: &[f64],
    n: usize,
    center: f64,
    cot: f64,
    csc: f64,
    scale: Complex64,
    k: usize,
) -> Complex64 {
    let u = k as f64 - center;
    DIRECT_WEIGHT_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(input_lanes.len(), |weight_lanes| {
            fill_fractional_weight_lanes(weight_lanes, n, center, cot, csc, u);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                input_lanes,
                weight_lanes,
            )
            .expect("FRFT fractional Hermes dot uses equal-length interleaved lanes");
            Complex64::new(re, im) * scale
        })
    })
}

#[inline]
fn interleaved_lanes(input: &[Complex64]) -> &[f64] {
    bytemuck::cast_slice(input)
}

fn fill_centered_dft_weight_lanes(lanes: &mut [f64], n: usize, center: f64, sign: f64, u: f64) {
    for (j, lane_pair) in lanes.chunks_exact_mut(2).enumerate() {
        let x = j as f64 - center;
        let angle = sign * 2.0 * PI * (x * u) / n as f64;
        lane_pair[0] = angle.cos();
        lane_pair[1] = angle.sin();
    }
}

fn fill_fractional_weight_lanes(
    lanes: &mut [f64],
    n: usize,
    center: f64,
    cot: f64,
    csc: f64,
    u: f64,
) {
    for (j, lane_pair) in lanes.chunks_exact_mut(2).enumerate() {
        let x = j as f64 - center;
        let phase = PI * ((x * x + u * u) * cot - 2.0 * x * u * csc) / n as f64;
        lane_pair[0] = phase.cos();
        lane_pair[1] = phase.sin();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_abs_diff_eq;

    fn signal(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|index| Complex64::new((index as f64 * 0.17).sin(), (index as f64 * 0.23).cos()))
            .collect()
    }

    #[test]
    fn moirai_parallel_fractional_rows_match_serial_formula_at_threshold() {
        let n = 128;
        let input = signal(n);
        let order = 0.75_f64;
        let alpha = order * std::f64::consts::FRAC_PI_2;
        let cot = alpha.cos() / alpha.sin();
        let csc = 1.0 / alpha.sin();
        let scale = (Complex64::new(1.0, 0.0) - Complex64::i() * cot).sqrt() / (n as f64).sqrt();
        let center = (n as f64 - 1.0) * 0.5;
        let mut actual = vec![Complex64::new(0.0, 0.0); n];

        direct_frft_forward_into(&input, &mut actual, order, cot, csc, scale);

        for (row, value) in actual.iter().enumerate() {
            let expected = fractional_row(&input, n, center, cot, csc, scale, row);
            assert_abs_diff_eq!(value.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(value.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn moirai_parallel_centered_dft_rows_match_serial_formula_at_threshold() {
        let n = 128;
        let input = signal(n);
        let center = (n as f64 - 1.0) * 0.5;
        let scale = 1.0 / (n as f64).sqrt();
        let mut actual = vec![Complex64::new(0.0, 0.0); n];

        direct_frft_forward_into(&input, &mut actual, 1.0, 0.0, 0.0, Complex64::new(1.0, 0.0));

        for (row, value) in actual.iter().enumerate() {
            let expected = centered_dft_row(&input, n, center, -1.0, scale, row);
            assert_abs_diff_eq!(value.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(value.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn hermes_fractional_row_matches_scalar_formula_at_threshold() {
        let n = 128;
        let input = signal(n);
        let input_lanes = interleaved_lanes(&input);
        let order = 0.75_f64;
        let alpha = order * std::f64::consts::FRAC_PI_2;
        let cot = alpha.cos() / alpha.sin();
        let csc = 1.0 / alpha.sin();
        let scale = (Complex64::new(1.0, 0.0) - Complex64::i() * cot).sqrt() / (n as f64).sqrt();
        let center = (n as f64 - 1.0) * 0.5;

        for k in [0_usize, 1, 17, 64, 127] {
            let actual = fractional_row_hermes(input_lanes, n, center, cot, csc, scale, k);
            let expected = fractional_row(&input, n, center, cot, csc, scale, k);
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn hermes_centered_dft_row_matches_scalar_formula_at_threshold() {
        let n = 128;
        let input = signal(n);
        let input_lanes = interleaved_lanes(&input);
        let center = (n as f64 - 1.0) * 0.5;
        let scale = 1.0 / (n as f64).sqrt();

        for k in [0_usize, 1, 17, 64, 127] {
            let actual = centered_dft_row_hermes(input_lanes, n, center, -1.0, scale, k);
            let expected = centered_dft_row(&input, n, center, -1.0, scale, k);
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn direct_weight_lanes_match_scalar_phasors() {
        let n = 8;
        let center = (n as f64 - 1.0) * 0.5;
        let u = 2.0_f64 - center;
        let mut centered = vec![0.0; n * 2];
        fill_centered_dft_weight_lanes(&mut centered, n, center, -1.0, u);
        for (j, lane_pair) in centered.chunks_exact(2).enumerate() {
            let x = j as f64 - center;
            let angle = -2.0 * PI * (x * u) / n as f64;
            assert_eq!(lane_pair[0].to_bits(), angle.cos().to_bits());
            assert_eq!(lane_pair[1].to_bits(), angle.sin().to_bits());
        }

        let mut fractional = vec![0.0; n * 2];
        fill_fractional_weight_lanes(&mut fractional, n, center, 0.25, 1.25, u);
        for (j, lane_pair) in fractional.chunks_exact(2).enumerate() {
            let x = j as f64 - center;
            let phase = PI * ((x * x + u * u) * 0.25 - 2.0 * x * u * 1.25) / n as f64;
            assert_eq!(lane_pair[0].to_bits(), phase.cos().to_bits());
            assert_eq!(lane_pair[1].to_bits(), phase.sin().to_bits());
        }
    }
}
