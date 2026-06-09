//! Direct real-valued DHT kernel.

use crate::domain::contracts::error::{DhtError, DhtResult};
use moirai::ParallelSliceMut;

/// Below this length the serial path avoids parallel task-spawn overhead.
/// The threshold is a conservative empirical heuristic; a benchmark-derived value would replace this.
const PAR_THRESHOLD: usize = 256;

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
            *value = coefficient(input, factor, k);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, value)| {
            *value = coefficient(input, factor, k);
        });
    }
    Ok(())
}

/// Compute the Hartley coefficient H[k] = sum_n x[n] cas(factor * k * n).
fn coefficient(input: &[f64], factor: f64, k: usize) -> f64 {
    input
        .iter()
        .enumerate()
        .map(|(n, sample)| sample * hartley_cas(factor * k as f64 * n as f64))
        .sum()
}
