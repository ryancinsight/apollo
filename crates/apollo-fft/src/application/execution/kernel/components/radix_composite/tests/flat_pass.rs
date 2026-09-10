//! The generic flat passes against the scalar formula, on the dispatched
//! register width, in both precisions.

use super::super::CompositeCache;
use eunomia::Complex;
use std::f64::consts::TAU;

/// A radix-2 pass over several `(prev_len, g_count)` shapes, with and
/// without a pointwise spectrum, against the scalar butterfly.
///
/// Each output is `a0 ± a1 · t`: the multiply carries two roundings and the
/// add one, against a scalar reference with its own three, so the distance
/// is within `6 u (|a0| + |a1 t|)`; the bound is `8 u` of that magnitude.
fn radix_2_pass_matches_the_scalar_formula<T>(unit_roundoff: f64)
where
    T: CompositeCache + eunomia::FloatElement + Into<f64>,
{
    let mut ran = 0;
    for (prev_len, g_count) in [
        (1usize, 64usize),
        (2, 32),
        (3, 21),
        (5, 13),
        (8, 8),
        (13, 5),
        (32, 2),
        (64, 1),
    ] {
        let stage_chunk = 2 * prev_len;
        let n = g_count * stage_chunk;
        let src: Vec<Complex<T>> = (0..n)
            .map(|i| {
                let x = i as f64;
                Complex::new(
                    T::from_f64((0.013 * x).sin() + 0.5),
                    T::from_f64(0.25 * (0.029 * x).cos() - 0.125),
                )
            })
            .collect();
        let tw: Vec<Complex<T>> = (0..prev_len)
            .map(|j| {
                let (s, c) = (-TAU * j as f64 / stage_chunk as f64).sin_cos();
                Complex::new(T::from_f64(c), T::from_f64(s))
            })
            .collect();
        let factors: Vec<Complex<T>> = (0..n)
            .map(|i| Complex::new(T::from_f64(1.0 + 0.001 * i as f64), T::from_f64(-0.5)))
            .collect();
        for pointwise in [None, Some(factors.as_slice())] {
            let mut dst = vec![Complex::new(T::from_f64(0.0), T::from_f64(0.0)); n];
            if !T::try_flat_pass_r2::<false>(
                &src,
                &mut dst,
                prev_len,
                g_count,
                stage_chunk,
                &tw,
                pointwise,
            ) {
                continue;
            }
            ran += 1;
            let stride = g_count * prev_len;
            for g in 0..g_count {
                for j in 0..prev_len {
                    let a0 = src[g * prev_len + j];
                    let v = src[stride + g * prev_len + j];
                    let t = tw[j];
                    let (vr, vi, tr, ti): (f64, f64, f64, f64) =
                        (v.re.into(), v.im.into(), t.re.into(), t.im.into());
                    let a1 = (vr * tr - vi * ti, vr * ti + vi * tr);
                    let (r0, i0): (f64, f64) = (a0.re.into(), a0.im.into());
                    let magnitude = r0.hypot(i0) + a1.0.hypot(a1.1);
                    let mut expected = [(r0 + a1.0, i0 + a1.1), (r0 - a1.0, i0 - a1.1)];
                    let mut bound = 8.0 * unit_roundoff * magnitude;
                    if let Some(factors) = pointwise {
                        for (k, e) in expected.iter_mut().enumerate() {
                            let p = factors[g * stage_chunk + j + k * prev_len];
                            let (pr, pi): (f64, f64) = (p.re.into(), p.im.into());
                            *e = (e.0 * pr - e.1 * pi, e.0 * pi + e.1 * pr);
                        }
                        bound *= 4.0;
                    }
                    for (k, e) in expected.iter().enumerate() {
                        let got = dst[g * stage_chunk + j + k * prev_len];
                        let (gr, gi): (f64, f64) = (got.re.into(), got.im.into());
                        let error = (gr - e.0).hypot(gi - e.1);
                        assert!(
                            error <= bound,
                            "prev_len={prev_len} g_count={g_count} pointwise={} group {g} j={j} output {k}: {error:e} > {bound:e}",
                            pointwise.is_some()
                        );
                    }
                }
            }
        }
    }
    assert!(
        ran >= 14,
        "the dispatched width ran {ran} of 16 pass shapes"
    );
}

#[test]
fn radix_2_pass_matches_the_scalar_formula_in_both_precisions() {
    radix_2_pass_matches_the_scalar_formula::<f32>(f64::from(f32::EPSILON) / 2.0);
    radix_2_pass_matches_the_scalar_formula::<f64>(f64::EPSILON / 2.0);
}

/// A radix-4 pass over several shapes in both directions against the
/// scalar butterfly on twiddled arms.
///
/// Each output sums four twiddled arms: two multiply roundings and two add
/// levels per arm, within `4 u` of each `|a_k|` (the twiddles are unit), and
/// the scalar reference carries the same; the bound is `12 u` of the arms'
/// magnitude.
fn radix_4_pass_matches_the_scalar_formula<T, const INVERSE: bool>(unit_roundoff: f64)
where
    T: CompositeCache + eunomia::FloatElement + Into<f64>,
{
    let sign = if INVERSE { 1.0 } else { -1.0 };
    let mut ran = 0;
    for (prev_len, g_count) in [
        (1usize, 64usize),
        (1, 65),
        (2, 32),
        (3, 21),
        (5, 13),
        (8, 8),
        (13, 5),
        (32, 2),
    ] {
        let stage_chunk = 4 * prev_len;
        let n = g_count * stage_chunk;
        let src: Vec<Complex<T>> = (0..n)
            .map(|i| {
                let x = i as f64;
                Complex::new(
                    T::from_f64((0.017 * x).sin() - 0.25),
                    T::from_f64(0.5 * (0.023 * x).cos() + 0.125),
                )
            })
            .collect();
        let tw: Vec<Complex<T>> = (1..4)
            .flat_map(|k| {
                (0..prev_len).map(move |j| {
                    let (s, c) = (sign * TAU * (k * j) as f64 / stage_chunk as f64).sin_cos();
                    Complex::new(T::from_f64(c), T::from_f64(s))
                })
            })
            .collect();
        let mut dst = vec![Complex::new(T::from_f64(0.0), T::from_f64(0.0)); n];
        if !T::try_flat_pass_r4::<INVERSE>(
            &src,
            &mut dst,
            prev_len,
            g_count,
            stage_chunk,
            &tw,
            None,
        ) {
            continue;
        }
        ran += 1;
        let stride = g_count * prev_len;
        for g in 0..g_count {
            for j in 0..prev_len {
                let arm = |k: usize| -> (f64, f64) {
                    let a = src[k * stride + g * prev_len + j];
                    let (ar, ai): (f64, f64) = (a.re.into(), a.im.into());
                    if k == 0 {
                        return (ar, ai);
                    }
                    let t = tw[(k - 1) * prev_len + j];
                    let (tr, ti): (f64, f64) = (t.re.into(), t.im.into());
                    (ar * tr - ai * ti, ar * ti + ai * tr)
                };
                let (a0, a1, a2, a3) = (arm(0), arm(1), arm(2), arm(3));
                let magnitude = [a0, a1, a2, a3].iter().map(|a| a.0.hypot(a.1)).sum::<f64>();
                let t0 = (a0.0 + a2.0, a0.1 + a2.1);
                let t1 = (a0.0 - a2.0, a0.1 - a2.1);
                let t2 = (a1.0 + a3.0, a1.1 + a3.1);
                let t3 = (a1.0 - a3.0, a1.1 - a3.1);
                let it3 = if INVERSE {
                    (-t3.1, t3.0)
                } else {
                    (t3.1, -t3.0)
                };
                let expected = [
                    (t0.0 + t2.0, t0.1 + t2.1),
                    (t1.0 + it3.0, t1.1 + it3.1),
                    (t0.0 - t2.0, t0.1 - t2.1),
                    (t1.0 - it3.0, t1.1 - it3.1),
                ];
                let bound = 12.0 * unit_roundoff * magnitude;
                for (k, e) in expected.iter().enumerate() {
                    let got = dst[g * stage_chunk + j + k * prev_len];
                    let (gr, gi): (f64, f64) = (got.re.into(), got.im.into());
                    let error = (gr - e.0).hypot(gi - e.1);
                    assert!(
                        error <= bound,
                        "inverse={INVERSE} prev_len={prev_len} g_count={g_count} group {g} j={j} arm {k}: {error:e} > {bound:e}"
                    );
                }
            }
        }
    }
    assert!(ran >= 6, "the dispatched width ran {ran} of 8 pass shapes");
}

#[test]
fn radix_4_pass_matches_the_scalar_formula_in_both_precisions_and_directions() {
    radix_4_pass_matches_the_scalar_formula::<f32, false>(f64::from(f32::EPSILON) / 2.0);
    radix_4_pass_matches_the_scalar_formula::<f32, true>(f64::from(f32::EPSILON) / 2.0);
    radix_4_pass_matches_the_scalar_formula::<f64, false>(f64::EPSILON / 2.0);
    radix_4_pass_matches_the_scalar_formula::<f64, true>(f64::EPSILON / 2.0);
}
