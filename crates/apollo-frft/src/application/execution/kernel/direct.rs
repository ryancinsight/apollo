use moirai::ParallelSliceMut;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Below this O(N²) operation count, serial loops avoid parallel scheduling overhead.
const FRFT_PAR_OP_THRESHOLD: usize = 16_384;

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
            output.par_mut().enumerate(|k, out| {
                *out = centered_dft_row(input, n, center, sign, scale, k);
            });
        } else {
            output.iter_mut().enumerate().for_each(|(k, out)| {
                *out = centered_dft_row(input, n, center, sign, scale, k);
            });
        }
        return;
    }

    if work_items >= FRFT_PAR_OP_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            *out = fractional_row(input, n, center, cot, csc, scale, k);
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

#[cfg(test)]
mod tests {
    use super::*;

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
            assert_eq!(value.re.to_bits(), expected.re.to_bits());
            assert_eq!(value.im.to_bits(), expected.im.to_bits());
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
            assert_eq!(value.re.to_bits(), expected.re.to_bits());
            assert_eq!(value.im.to_bits(), expected.im.to_bits());
        }
    }
}
