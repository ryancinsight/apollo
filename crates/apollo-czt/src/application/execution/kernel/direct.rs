use crate::domain::contracts::error::CztError;
use mnemosyne::scratch::ScratchPool;
use moirai::ParallelSliceMut;
use ndarray::Array1;
use num_complex::Complex64;

/// Below this O(NM) operation count, serial loops avoid scheduling overhead.
const DIRECT_PAR_OP_THRESHOLD: usize = 16_384;

thread_local! {
    static DIRECT_WEIGHT_LANE_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

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
    czt_direct_forward_slice_into(input, output_slice, a, w)?;
    Ok(output)
}

/// Direct CZT evaluation over contiguous Complex64 slices.
pub(crate) fn czt_direct_forward_slice_into(
    input: &[Complex64],
    output: &mut [Complex64],
    a: Complex64,
    w: Complex64,
) -> Result<(), CztError> {
    if input.is_empty() || output.is_empty() {
        return Err(CztError::EmptyLength);
    }
    let work_items = input.len().saturating_mul(output.len());
    if work_items >= DIRECT_PAR_OP_THRESHOLD {
        let input_lanes = interleaved_lanes(input);
        output.par_mut().enumerate(|k, slot| {
            *slot = czt_direct_bin_hermes(input_lanes, k, a, w);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, slot)| {
            *slot = czt_direct_bin(input, k, a, w);
        });
    }
    Ok(())
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

fn czt_direct_bin_hermes(input_lanes: &[f64], k: usize, a: Complex64, w: Complex64) -> Complex64 {
    let z_k = a * w.powf(-(k as f64));
    let z_inv = Complex64::new(1.0, 0.0) / z_k;
    DIRECT_WEIGHT_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(input_lanes.len(), |weight_lanes| {
            fill_power_lanes(weight_lanes, z_inv);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                input_lanes,
                weight_lanes,
            )
            .expect("CZT Hermes dot uses equal-length interleaved complex lane slices");
            Complex64::new(re, im)
        })
    })
}

#[inline]
fn interleaved_lanes(values: &[Complex64]) -> &[f64] {
    bytemuck::cast_slice(values)
}

fn fill_power_lanes(lanes: &mut [f64], z_inv: Complex64) {
    let mut z_pow = Complex64::new(1.0, 0.0);
    for lane_pair in lanes.chunks_exact_mut(2) {
        lane_pair[0] = z_pow.re;
        lane_pair[1] = z_pow.im;
        z_pow *= z_inv;
    }
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

    #[test]
    fn hermes_direct_row_matches_scalar_formula_at_threshold() {
        let input_len = 128usize;
        let input = (0..input_len)
            .map(|index| {
                Complex64::new((index as f64 * 0.125).sin(), (index as f64 * 0.03125).cos())
            })
            .collect::<Vec<_>>();
        let input_lanes = interleaved_lanes(&input);
        let a = Complex64::from_polar(1.0, 0.125);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / 257.0);

        for k in [0usize, 1, 17, 64, 127] {
            let actual = czt_direct_bin_hermes(input_lanes, k, a, w);
            let expected = czt_direct_bin(&input, k, a, w);
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn power_lanes_match_geometric_progression() {
        let z_inv = Complex64::from_polar(1.0, 0.25);
        let mut lanes = vec![0.0; 16];

        fill_power_lanes(&mut lanes, z_inv);

        let mut expected = Complex64::new(1.0, 0.0);
        for lane_pair in lanes.chunks_exact(2) {
            assert_eq!(lane_pair[0].to_bits(), expected.re.to_bits());
            assert_eq!(lane_pair[1].to_bits(), expected.im.to_bits());
            expected *= z_inv;
        }
    }
}
