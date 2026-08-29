//! Whether a transform length has a correct route through the plan builders.
//!
//! [`FftPlan1D::new`](crate::FftPlan1D::new) is infallible, and for some
//! lengths the transform is not merely unavailable but wrong: 109 lengths in
//! `2..=8200` either assert during planning or execution, or return results
//! with 100% relative error
//! (`ATLAS-APOLLO-COMPOSITE-RADIX-WRONG-ANSWERS-2026-08-28`). A caller whose
//! length is chosen by *its* caller — a grid width, a user-supplied period —
//! needs to know that before planning, so it can take another route.
//!
//! # Why this probes instead of predicting
//!
//! The affected set has no arithmetic characterization to shadow. It is not
//! "divisible by the square of a prime above the radix set": `23² = 529`
//! routes correctly while `19² = 361` does not, `19·23 = 437` is correct while
//! `6·437 = 2622` is not, and the prime `1153` and all its multiples fail. A
//! predicate written against the strategy chain would therefore be both wrong
//! and silently drifting. So [`supports_length`] runs the transform once and
//! checks it against a closed-form oracle, which is exact for every failure
//! mode by construction and needs no knowledge of the routing at all. When the
//! tracking item lands a general route, this starts answering `true`
//! everywhere with no change here.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::domain::metadata::shape::Shape1D;
use crate::FftPlan1D;
use eunomia::Complex64;

/// Probe verdicts, keyed by length. A verdict costs one `O(n log n)`
/// transform, so it is paid once per length per process.
static VERDICTS: LazyLock<Mutex<HashMap<usize, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Report whether a 1-D complex transform of length `n` computes the DFT.
///
/// `false` means the length is one of the routes tracked by
/// `ATLAS-APOLLO-COMPOSITE-RADIX-WRONG-ANSWERS-2026-08-28`: planning or
/// executing at this length either asserts or returns a wrong answer, so the
/// caller must use another method. `true` means a transform at this length was
/// executed and agreed with the analytic DFT of a single complex exponential
/// to within the accumulated rounding bound.
///
/// The first call for a given `n` runs one transform of that length; the
/// verdict is then cached for the life of the process, so repeated calls are a
/// map lookup.
///
/// # Diagnostics
///
/// Probing an unroutable length runs code that asserts, so apollo's own
/// assertion message — which names the length and the tracking item — reaches
/// stderr once per length. That message is the reason for the `false`. This
/// function deliberately does not install a panic hook to suppress it: doing so
/// would race any other thread panicking in the same window.
///
/// # Examples
///
/// ```
/// assert!(apollo_fft::supports_length(1024));
/// assert!(apollo_fft::supports_length(360));
/// ```
#[must_use]
pub fn supports_length(n: usize) -> bool {
    if n <= 1 {
        // The identity strategy; nothing to route and nothing to get wrong.
        return true;
    }
    if let Some(&verdict) = VERDICTS.lock().expect("routing verdicts").get(&n) {
        return verdict;
    }
    let verdict = probe(n);
    VERDICTS
        .lock()
        .expect("routing verdicts")
        .insert(n, verdict);
    verdict
}

/// Transform one complex exponential and compare against its analytic DFT.
///
/// For `x[j] = exp(2πi j / n)` the DFT under the forward convention
/// `X[k] = Σ_j x[j] exp(-2πi jk/n)` is `n` at exactly one bin and zero at every
/// other, so both the peak magnitude and the total leakage are known in closed
/// form. Parseval's identity `Σ_k |X[k]|² = n Σ_j |x[j]|²` is checked on a
/// second, spectrally dense input, which no permutation error can satisfy by
/// accident.
fn probe(n: usize) -> bool {
    std::panic::catch_unwind(|| {
        let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("probe length is non-zero"));

        // Rounding budget. A radix-based FFT of length n accumulates at most
        // O(ε log₂ n) relative error, and here ‖x‖∞ = 1 gives ‖X‖∞ = n, so the
        // absolute error is bounded by a small multiple of n·ε·log₂ n. With
        // log₂ n < 64 for any representable length, 64·n·ε covers it with room
        // to spare. The defect being detected produces 100% relative error, so
        // the verdict is insensitive to the exact multiple.
        let tolerance = 64.0 * n as f64 * f64::EPSILON;

        let mut tone: Vec<Complex64> = (0..n)
            .map(|j| {
                // j mod n keeps the angle argument exact for large n.
                let angle = 2.0 * std::f64::consts::PI * (j % n) as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();
        plan.forward_complex_slice_inplace(&mut tone);
        let (peak_bin, peak) = tone.iter().enumerate().fold((0, 0.0f64), |best, (i, v)| {
            let magnitude = v.norm();
            if magnitude > best.1 {
                (i, magnitude)
            } else {
                best
            }
        });
        let leakage = tone
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != peak_bin)
            .map(|(_, v)| v.norm())
            .fold(0.0f64, f64::max);
        if (peak - n as f64).abs() > tolerance || leakage > tolerance {
            return false;
        }

        let dense: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i % 101) as f64 / 101.0 - 0.5, (i % 97) as f64 / 97.0 - 0.5))
            .collect();
        let energy_in: f64 = dense.iter().map(|v| v.norm_sqr()).sum();
        let mut spectrum = dense;
        plan.forward_complex_slice_inplace(&mut spectrum);
        let energy_out: f64 = spectrum.iter().map(|v| v.norm_sqr()).sum();
        let expected = n as f64 * energy_in;
        (energy_out - expected).abs() <= tolerance * expected
    })
    .unwrap_or(false)
}

#[cfg(test)]
mod routing_tests;
