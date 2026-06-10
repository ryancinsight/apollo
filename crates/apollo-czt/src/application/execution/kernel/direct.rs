use crate::domain::contracts::error::CztError;
use moirai::ParallelSliceMut;
use ndarray::Array1;
use num_complex::Complex64;

/// Below this O(NM) operation count, serial loops avoid scheduling overhead.
const DIRECT_PAR_OP_THRESHOLD: usize = 16_384;

/// Direct execution of the Chirp Z-Transform logic.
/// Evaluates sequentially against $O(NM)$ limits.
pub fn czt_direct_forward(
    input: &Array1<Complex64>,
    output_len: usize,
    a: Complex64,
    w: Complex64,
) -> Result<Array1<Complex64>, CztError> {
    let mut output = Array1::zeros(output_len);
    let input = input
        .as_slice()
        .expect("CZT direct input must be contiguous");
    let output_slice = output
        .as_slice_mut()
        .expect("CZT direct output must be contiguous");
    let work_items = input.len().saturating_mul(output_len);
    if work_items >= DIRECT_PAR_OP_THRESHOLD {
        output_slice.par_mut().enumerate(|k, slot| {
            *slot = czt_direct_bin(input, k, a, w);
        });
    } else {
        output_slice.iter_mut().enumerate().for_each(|(k, slot)| {
            *slot = czt_direct_bin(input, k, a, w);
        });
    }
    Ok(output)
}

#[inline]
fn czt_direct_bin(input: &[Complex64], k: usize, a: Complex64, w: Complex64) -> Complex64 {
    let z_k = a * w.powf(-(k as f64));
    let mut sum = Complex64::new(0.0, 0.0);
    let mut z_pow = Complex64::new(1.0, 0.0);
    let z_inv = Complex64::new(1.0, 0.0) / z_k;
    for value in input {
        sum += *value * z_pow;
        z_pow *= z_inv;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn moirai_parallel_direct_rows_match_bin_formula_at_threshold() {
        let input_len = 128usize;
        let output_len = DIRECT_PAR_OP_THRESHOLD / input_len;
        let input = Array1::from_shape_fn(input_len, |index| {
            Complex64::new((index as f64 * 0.125).sin(), (index as f64 * 0.03125).cos())
        });
        let a = Complex64::from_polar(1.0, 0.125);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / 257.0);

        let actual = czt_direct_forward(&input, output_len, a, w).expect("direct czt");
        let input = input.as_slice().expect("contiguous input");

        for (k, actual) in actual.iter().enumerate() {
            let expected = czt_direct_bin(input, k, a, w);
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }
}
