//! The retangle inverts the untangle.

use super::{retangle_real_half, untangle_real_half};
use eunomia::Complex;

/// `2 x 64u` covering both directions.
///
/// Each direction forms a bin from two inputs of magnitude at most `A`: a
/// twiddle carried at most seven recurrence steps from its seed (each a complex
/// product, `sqrt(5)u` without FMA, so under `24u` in all), one complex product
/// with it, and three roundings of sums and halvings — under `32u·A` per bin,
/// counted twice for the two inputs. The inverse sees the forward's error
/// through a map of gain at most one per input and adds its own.
const BOUND_FACTOR: f64 = 128.0;

/// Every packed length up to 64 and a sample above, odd ones included: the
/// untangle maps `M` complex samples onto `M + 1` bins with real ends for any
/// `M`, so the retangle must invert it without the split's multiple-of-four
/// restriction. f64 here; the public round trips in `tests/real_half_api`
/// exercise the kernel at every shipped plan precision.
#[test]
fn retangle_inverts_untangle_for_any_packed_spectrum() {
    for m in (1..=64).chain([96, 100, 255, 256, 512]) {
        let n = 2 * m;
        let packed: Vec<Complex<f64>> = (0..m)
            .map(|k| {
                let x = k as f64;
                Complex::new((0.37 * x).sin() + 0.2, (1.3 * x).cos() - 0.1)
            })
            .collect();
        let mut bins = packed.clone();
        bins.push(Complex::default());

        untangle_real_half(&mut bins, n);
        retangle_real_half(&mut bins, n);

        let scale = packed
            .iter()
            .map(|z| z.re.abs().max(z.im.abs()))
            .fold(0.0_f64, f64::max);
        let bound = BOUND_FACTOR * (f64::EPSILON / 2.0) * scale;
        for (k, (got, want)) in bins.iter().zip(&packed).enumerate() {
            assert!(
                (got.re - want.re).abs() <= bound && (got.im - want.im).abs() <= bound,
                "M={m} sample {k}: {got:?} against {want:?} (bound {bound:.2e})"
            );
        }
    }
}
