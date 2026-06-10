//! Direct real-valued DHT kernel.

use crate::domain::contracts::error::{DhtError, DhtResult};
use moirai::ParallelSliceMut;

/// Below this length the serial path avoids parallel task-spawn overhead.
/// The threshold is a conservative empirical heuristic; a benchmark-derived value would replace this.
const PAR_THRESHOLD: usize = 256;

/// Below this row length, direct scalar accumulation avoids Hartley-row scratch setup.
const HERMES_DOT_LEN_THRESHOLD: usize = 1_024;

thread_local! {
    static HARTLEY_ROW_SCRATCH: mnemosyne::scratch::ScratchPool<f64> = const { mnemosyne::scratch::ScratchPool::new() };
}

/// Compute the Hartley cas (cosine+sine) coefficient: cas(theta) = cos(theta) + sin(theta).
#[must_use]
pub fn hartley_cas(theta: f64) -> f64 {
    theta.cos() + theta.sin()
}

/// Compute one unnormalized DHT pass from `input` into `output`.
///
/// The kernel implements `H[k] = sum_n x[n] cas(2 pi k n / N)` and uses only
/// real-valued storage. The same function is valid for the inverse pass; the
/// caller applies the Hartley normalization `1 / N`.
/// The DHT satisfies DHT(DHT(x)) = N*x, where the factor N arises from the
/// circular convolution theorem for the Hartley transform.
pub fn transform_real(input: &[f64], output: &mut [f64]) -> DhtResult<()> {
    let len = input.len();
    if len == 0 {
        return Err(DhtError::EmptySignal);
    }
    if output.len() != len {
        return Err(DhtError::LengthMismatch);
    }

    let factor = std::f64::consts::TAU / len as f64;
    if len >= PAR_THRESHOLD {
        output.par_mut().enumerate(|k, value| {
            *value = coefficient_with_hermes(input, factor, k);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, value)| {
            *value = coefficient(input, factor, k);
        });
    }
    Ok(())
}

fn coefficient_with_hermes(input: &[f64], factor: f64, k: usize) -> f64 {
    if input.len() < HERMES_DOT_LEN_THRESHOLD {
        return coefficient(input, factor, k);
    }

    HARTLEY_ROW_SCRATCH.with(|pool| {
        pool.with_scratch(input.len(), |row| {
            fill_hartley_row(row, factor, k);
            hermes_simd::dot::<f64>(input, row)
                .expect("Hermes DHT coefficient dot uses equal-length finite row buffers")
        })
    })
}

fn fill_hartley_row(row: &mut [f64], factor: f64, k: usize) {
    for (n, value) in row.iter_mut().enumerate() {
        *value = hartley_cas(factor * k as f64 * n as f64);
    }
}

/// Compute the Hartley coefficient H[k] = sum_n x[n] cas(factor * k * n).
fn coefficient(input: &[f64], factor: f64, k: usize) -> f64 {
    input
        .iter()
        .enumerate()
        .map(|(n, sample)| sample * hartley_cas(factor * k as f64 * n as f64))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use hermes_simd::PreferredArch;

    #[test]
    fn hermes_coefficient_matches_scalar_formula_at_threshold() {
        let len = HERMES_DOT_LEN_THRESHOLD;
        let input: Vec<f64> = (0..len)
            .map(|index| (index as f64 * 0.03125).sin() + (index as f64 * 0.0078125).cos())
            .collect();
        let factor = std::f64::consts::TAU / len as f64;

        for &k in &[0usize, 1, 7, 127, 511] {
            let actual = coefficient_with_hermes(&input, factor, k);
            let expected = coefficient(&input, factor, k);
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-9);
        }
    }

    #[test]
    fn hermes_hartley_row_matches_cas_formula() {
        let len = HERMES_DOT_LEN_THRESHOLD;
        let factor = std::f64::consts::TAU / len as f64;
        let k = 19usize;
        let mut row = vec![0.0; len];

        fill_hartley_row(&mut row, factor, k);

        for (n, actual) in row.iter().enumerate() {
            let expected = hartley_cas(factor * k as f64 * n as f64);
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn preferred_arch_marker_is_zero_sized() {
        assert_eq!(core::mem::size_of::<PreferredArch>(), 0);
    }
}
