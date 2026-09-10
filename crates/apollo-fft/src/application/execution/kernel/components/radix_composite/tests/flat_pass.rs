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
        (2, 33),
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
        "the dispatched width ran {ran} of 18 pass shapes"
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
        (2, 33),
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
    assert!(ran >= 7, "the dispatched width ran {ran} of 9 pass shapes");
}

#[test]
fn radix_4_pass_matches_the_scalar_formula_in_both_precisions_and_directions() {
    radix_4_pass_matches_the_scalar_formula::<f32, false>(f64::from(f32::EPSILON) / 2.0);
    radix_4_pass_matches_the_scalar_formula::<f32, true>(f64::from(f32::EPSILON) / 2.0);
    radix_4_pass_matches_the_scalar_formula::<f64, false>(f64::EPSILON / 2.0);
    radix_4_pass_matches_the_scalar_formula::<f64, true>(f64::EPSILON / 2.0);
}

/// A radix-3 pass over several shapes in both directions against the
/// scalar butterfly on twiddled arms; the bound as for radix 4, with one
/// scaled term more.
fn radix_3_pass_matches_the_scalar_formula<T, const INVERSE: bool>(unit_roundoff: f64)
where
    T: CompositeCache + eunomia::FloatElement + Into<f64>,
{
    let sign = if INVERSE { 1.0 } else { -1.0 };
    let s = 3f64.sqrt() / 2.0;
    let mut ran = 0;
    for (prev_len, g_count) in [
        (1usize, 64usize),
        (1, 65),
        (2, 32),
        (2, 33),
        (3, 21),
        (5, 13),
        (8, 8),
        (13, 5),
        (32, 2),
    ] {
        let stage_chunk = 3 * prev_len;
        let n = g_count * stage_chunk;
        let src: Vec<Complex<T>> = (0..n)
            .map(|i| {
                let x = i as f64;
                Complex::new(
                    T::from_f64((0.011 * x).cos() - 0.125),
                    T::from_f64(0.75 * (0.037 * x).sin() + 0.25),
                )
            })
            .collect();
        let tw: Vec<Complex<T>> = (1..3)
            .flat_map(|k| {
                (0..prev_len).map(move |j| {
                    let (si, co) = (sign * TAU * (k * j) as f64 / stage_chunk as f64).sin_cos();
                    Complex::new(T::from_f64(co), T::from_f64(si))
                })
            })
            .collect();
        let mut dst = vec![Complex::new(T::from_f64(0.0), T::from_f64(0.0)); n];
        if !T::try_flat_pass_r3::<INVERSE>(
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
                let (a0, a1, a2) = (arm(0), arm(1), arm(2));
                let magnitude = [a0, a1, a2].iter().map(|a| a.0.hypot(a.1)).sum::<f64>();
                let sum = (a1.0 + a2.0, a1.1 + a2.1);
                let diff = (a1.0 - a2.0, a1.1 - a2.1);
                let m0 = (a0.0 - 0.5 * sum.0, a0.1 - 0.5 * sum.1);
                let m1 = if INVERSE {
                    (-diff.1 * s, diff.0 * s)
                } else {
                    (diff.1 * s, -diff.0 * s)
                };
                let expected = [
                    (a0.0 + sum.0, a0.1 + sum.1),
                    (m0.0 + m1.0, m0.1 + m1.1),
                    (m0.0 - m1.0, m0.1 - m1.1),
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
    assert!(ran >= 7, "the dispatched width ran {ran} of 9 pass shapes");
}

#[test]
fn radix_3_pass_matches_the_scalar_formula_in_both_precisions_and_directions() {
    radix_3_pass_matches_the_scalar_formula::<f32, false>(f64::from(f32::EPSILON) / 2.0);
    radix_3_pass_matches_the_scalar_formula::<f32, true>(f64::from(f32::EPSILON) / 2.0);
    radix_3_pass_matches_the_scalar_formula::<f64, false>(f64::EPSILON / 2.0);
    radix_3_pass_matches_the_scalar_formula::<f64, true>(f64::EPSILON / 2.0);
}

/// A radix-5 pass over several shapes in both directions against the
/// scalar butterfly on twiddled arms; the sums carry 4 scaled terms, so the
/// bound is `20 u` of the arms' magnitude.
fn radix_5_pass_matches_the_scalar_formula<T, const INVERSE: bool>(unit_roundoff: f64)
where
    T: CompositeCache + eunomia::FloatElement + Into<f64>,
{
    let sign = if INVERSE { 1.0 } else { -1.0 };
    let mut ran = 0;
    for (prev_len, g_count) in [
        (1usize, 64usize),
        (1, 65),
        (2, 32),
        (2, 33),
        (3, 21),
        (5, 13),
        (8, 8),
        (13, 5),
        (32, 2),
    ] {
        let stage_chunk = 5 * prev_len;
        let n = g_count * stage_chunk;
        let src: Vec<Complex<T>> = (0..n)
            .map(|i| {
                let x = i as f64;
                Complex::new(
                    T::from_f64((0.019 * x).sin() + 0.375),
                    T::from_f64(0.5 * (0.041 * x).cos() - 0.25),
                )
            })
            .collect();
        let tw: Vec<Complex<T>> = (1..5)
            .flat_map(|k| {
                (0..prev_len).map(move |j| {
                    let (si, co) = (sign * TAU * (k * j) as f64 / stage_chunk as f64).sin_cos();
                    Complex::new(T::from_f64(co), T::from_f64(si))
                })
            })
            .collect();
        let mut dst = vec![Complex::new(T::from_f64(0.0), T::from_f64(0.0)); n];
        if !T::try_flat_pass_r5::<INVERSE>(
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
                let (a0, a1, a2, a3, a4) = (arm(0), arm(1), arm(2), arm(3), arm(4));
                let magnitude = [a0, a1, a2, a3, a4]
                    .iter()
                    .map(|a| a.0.hypot(a.1))
                    .sum::<f64>();
                let (c1, c2) = ((TAU / 5.0).cos(), (2.0 * TAU / 5.0).cos());
                let (s1, s2) = (sign * (TAU / 5.0).sin(), sign * (2.0 * TAU / 5.0).sin());
                let add = |p: (f64, f64), q: (f64, f64)| (p.0 + q.0, p.1 + q.1);
                let sub = |p: (f64, f64), q: (f64, f64)| (p.0 - q.0, p.1 - q.1);
                let sc = |p: (f64, f64), k: f64| (p.0 * k, p.1 * k);
                let (t1, t2, t3, t4) = (add(a1, a4), sub(a1, a4), add(a2, a3), sub(a2, a3));
                let m1 = add(sc(t1, c1), sc(t3, c2));
                let m2 = add(sc(t1, c2), sc(t3, c1));
                let q3 = add(sc(t2, s1), sc(t4, s2));
                let q4 = sub(sc(t2, s2), sc(t4, s1));
                let iq3 = (-q3.1, q3.0);
                let iq4 = (-q4.1, q4.0);
                let a1c = add(a0, m1);
                let a2c = add(a0, m2);
                let expected = [
                    add(a0, add(t1, t3)),
                    add(a1c, iq3),
                    add(a2c, iq4),
                    sub(a2c, iq4),
                    sub(a1c, iq3),
                ];

                let bound = 20.0 * unit_roundoff * magnitude;
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
    assert!(ran >= 7, "the dispatched width ran {ran} of 9 pass shapes");
}

#[test]
fn radix_5_pass_matches_the_scalar_formula_in_both_precisions_and_directions() {
    radix_5_pass_matches_the_scalar_formula::<f32, false>(f64::from(f32::EPSILON) / 2.0);
    radix_5_pass_matches_the_scalar_formula::<f32, true>(f64::from(f32::EPSILON) / 2.0);
    radix_5_pass_matches_the_scalar_formula::<f64, false>(f64::EPSILON / 2.0);
    radix_5_pass_matches_the_scalar_formula::<f64, true>(f64::EPSILON / 2.0);
}

/// A radix-7 pass over several shapes in both directions against the
/// scalar butterfly on twiddled arms; the sums carry 6 scaled terms, so the
/// bound is `28 u` of the arms' magnitude.
fn radix_7_pass_matches_the_scalar_formula<T, const INVERSE: bool>(unit_roundoff: f64)
where
    T: CompositeCache + eunomia::FloatElement + Into<f64>,
{
    let sign = if INVERSE { 1.0 } else { -1.0 };
    let mut ran = 0;
    for (prev_len, g_count) in [
        (1usize, 64usize),
        (1, 65),
        (2, 32),
        (2, 33),
        (3, 21),
        (5, 13),
        (8, 8),
        (13, 5),
        (32, 2),
    ] {
        let stage_chunk = 7 * prev_len;
        let n = g_count * stage_chunk;
        let src: Vec<Complex<T>> = (0..n)
            .map(|i| {
                let x = i as f64;
                Complex::new(
                    T::from_f64((0.019 * x).sin() + 0.375),
                    T::from_f64(0.5 * (0.041 * x).cos() - 0.25),
                )
            })
            .collect();
        let tw: Vec<Complex<T>> = (1..7)
            .flat_map(|k| {
                (0..prev_len).map(move |j| {
                    let (si, co) = (sign * TAU * (k * j) as f64 / stage_chunk as f64).sin_cos();
                    Complex::new(T::from_f64(co), T::from_f64(si))
                })
            })
            .collect();
        let mut dst = vec![Complex::new(T::from_f64(0.0), T::from_f64(0.0)); n];
        if !T::try_flat_pass_r7::<INVERSE>(
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
                let a: [(f64, f64); 7] = core::array::from_fn(arm);
                let magnitude = a.iter().map(|x| x.0.hypot(x.1)).sum::<f64>();
                let add = |p: (f64, f64), q: (f64, f64)| (p.0 + q.0, p.1 + q.1);
                let sub = |p: (f64, f64), q: (f64, f64)| (p.0 - q.0, p.1 - q.1);
                let sc = |p: (f64, f64), k: f64| (p.0 * k, p.1 * k);
                let turn = |v: (f64, f64)| if sign > 0.0 { (-v.1, v.0) } else { (v.1, -v.0) };
                let (c1, c2, c3) = (
                    (TAU / 7.0).cos(),
                    (2.0 * TAU / 7.0).cos(),
                    (3.0 * TAU / 7.0).cos(),
                );
                let (s1, s2, s3) = (
                    (TAU / 7.0).sin(),
                    (2.0 * TAU / 7.0).sin(),
                    (3.0 * TAU / 7.0).sin(),
                );
                let (xr1, xr2, xr3) = (add(a[1], a[6]), add(a[2], a[5]), add(a[3], a[4]));
                let (xi1, xi2, xi3) = (
                    turn(sub(a[1], a[6])),
                    turn(sub(a[2], a[5])),
                    turn(sub(a[3], a[4])),
                );
                let lin = |p: f64, q: f64, r: f64| {
                    add(a[0], add(sc(xr1, p), add(sc(xr2, q), sc(xr3, r))))
                };
                let (re1, re2, re3) = (lin(c1, c2, c3), lin(c2, c3, c1), lin(c3, c1, c2));
                let d1 = add(sc(xi1, s1), add(sc(xi2, s2), sc(xi3, s3)));
                let d2 = sub(sub(sc(xi1, s2), sc(xi2, s3)), sc(xi3, s1));
                let d3 = add(sub(sc(xi1, s3), sc(xi2, s1)), sc(xi3, s2));
                let expected = [
                    add(a[0], add(xr1, add(xr2, xr3))),
                    add(re1, d1),
                    add(re2, d2),
                    add(re3, d3),
                    sub(re3, d3),
                    sub(re2, d2),
                    sub(re1, d1),
                ];

                let bound = 28.0 * unit_roundoff * magnitude;
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
    assert!(ran >= 7, "the dispatched width ran {ran} of 9 pass shapes");
}

#[test]
fn radix_7_pass_matches_the_scalar_formula_in_both_precisions_and_directions() {
    radix_7_pass_matches_the_scalar_formula::<f32, false>(f64::from(f32::EPSILON) / 2.0);
    radix_7_pass_matches_the_scalar_formula::<f32, true>(f64::from(f32::EPSILON) / 2.0);
    radix_7_pass_matches_the_scalar_formula::<f64, false>(f64::EPSILON / 2.0);
    radix_7_pass_matches_the_scalar_formula::<f64, true>(f64::EPSILON / 2.0);
}
