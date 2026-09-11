//! The retangle inverts the untangle, and the twiddle table is the recurrence.

use super::{retangle_real_half, split_twiddle_table, split_twiddles, untangle_real_half};
use eunomia::Complex;

/// `2 x 64u` covering both directions.
///
/// Each direction forms a bin from two inputs of magnitude at most `A`: a
/// twiddle carried at most seven recurrence steps from its seed (each a complex
/// product, `sqrt(5)u` without FMA, so under `24u` in all), one complex product
/// with it, and three roundings of sums and halvings — under `32u·A` per bin,
/// counted twice for the two inputs. The inverse sees the forward's error
/// through a map of gain at most one per input and adds its own. A table
/// twiddle carries one rounding instead of the recurrence's, so it sits inside
/// the same bound.
const BOUND_FACTOR: f64 = 128.0;

/// Seven recurrence steps from a seed, each a complex product under
/// `sqrt(5)u`, plus the seed's own rounding: under `24u` per component.
const TWIDDLE_BOUND_FACTOR: f64 = 24.0;

fn packed(m: usize) -> Vec<Complex<f64>> {
    (0..m)
        .map(|k| {
            let x = k as f64;
            Complex::new((0.37 * x).sin() + 0.2, (1.3 * x).cos() - 0.1)
        })
        .collect()
}

/// Every packed length up to 64 and a sample above, odd ones included: the
/// untangle maps `M` complex samples onto `M + 1` bins with real ends for any
/// `M`, so the retangle must invert it without the split's multiple-of-four
/// restriction — from the recurrence and from the table alike. f64 here; the
/// public round trips in `tests/real_half_api` exercise the kernel at every
/// shipped plan precision.
#[test]
fn retangle_inverts_untangle_for_either_twiddle_source() {
    for m in (1..=64).chain([96, 100, 255, 256, 512]) {
        let n = 2 * m;
        let packed = packed(m);
        let scale = packed
            .iter()
            .map(|z| z.re.abs().max(z.im.abs()))
            .fold(0.0_f64, f64::max);
        let bound = BOUND_FACTOR * (f64::EPSILON / 2.0) * scale;
        let table = split_twiddle_table::<f64>(n);

        for (source, from_table) in [("recurrence", false), ("table", true)] {
            let mut bins = packed.clone();
            bins.push(Complex::default());
            if from_table {
                untangle_real_half(&mut bins, n, table.iter().copied());
                retangle_real_half(&mut bins, n, table.iter().copied());
            } else {
                untangle_real_half(&mut bins, n, split_twiddles(n));
                retangle_real_half(&mut bins, n, split_twiddles(n));
            }
            for (k, (got, want)) in bins.iter().zip(&packed).enumerate() {
                assert!(
                    (got.re - want.re).abs() <= bound && (got.im - want.im).abs() <= bound,
                    "M={m} {source} sample {k}: {got:?} against {want:?} (bound {bound:.2e})"
                );
            }
        }
    }
}

/// The table holds exactly the twiddles the recurrence yields, to the
/// recurrence's own error: same count, same `W_N^k`.
#[test]
fn twiddle_table_is_the_recurrence() {
    for n in (2..=128).step_by(2).chain([200, 1000, 1024]) {
        let table = split_twiddle_table::<f64>(n);
        let recurrence: Vec<Complex<f64>> = split_twiddles(n).collect();
        assert_eq!(table.len(), recurrence.len(), "N={n}: twiddle count");
        assert_eq!(
            table.len(),
            (n / 2).div_ceil(2).saturating_sub(1),
            "N={n}: k = 1..ceil(M/2)"
        );
        let bound = TWIDDLE_BOUND_FACTOR * (f64::EPSILON / 2.0);
        for (k, (a, b)) in table.iter().zip(&recurrence).enumerate() {
            assert!(
                (a.re - b.re).abs() <= bound && (a.im - b.im).abs() <= bound,
                "N={n} k={}: table {a:?} against recurrence {b:?}",
                k + 1
            );
        }
    }
}
