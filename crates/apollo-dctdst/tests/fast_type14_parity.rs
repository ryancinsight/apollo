//! The Type-I and Type-IV fast paths must compute what the direct kernels do.
//!
//! The direct kernels are the specification: each evaluates its defining sum
//! term by term. The fast paths reach the same values through an FFT of a
//! symmetric or half-shifted extension, which shares no code with them. That
//! makes the direct kernel a genuine oracle rather than a second opinion from
//! the same source.
//!
//! # Tolerance
//!
//! Derived from the accumulation, not fitted to the observed error.
//!
//! The direct kernel is the weaker of the two paths and sets the bound. It
//! sums `N` terms sequentially, each at most `2 * max|x|` (Type I carries the
//! factor of two), so its partial sums reach `O(N * max|x|)` and sequential
//! accumulation of `N` such terms carries relative error `O(N * eps)`. The
//! absolute bound is therefore `O(N^2 * eps * max|x|)`. The FFT path is the
//! stronger one — `O(log M * eps)` over `M ~ 2N` — so it does not enter.
//!
//! An earlier version of this file bounded the error at `N * eps`, which is
//! the *relative* growth mistaken for the absolute one. It rejected correct
//! results: DST-IV at `N = 360` differs by 1.7e-12 and DST-I at `N = 48` by
//! 1.3e-13, both a small multiple over that bound and both far under this one.
//!
//! The bound separates cleanly from what the test exists to catch. A sign or
//! index error in an extension puts the error at the scale of the output
//! itself — `O(N * max|x|)`, order 1e2 at these lengths — which is ten or more
//! orders above `N^2 * eps * max|x|`.

use apollo_dctdst::infrastructure::kernel::{direct, fast};

/// `N^2 * eps * max|x|`, with a small constant for the sum of both paths.
fn tolerance(n: usize, signal: &[f64]) -> f64 {
    let peak = signal.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
    4.0 * (n * n) as f64 * f64::EPSILON * peak
}

/// Deterministic and asymmetric: a symmetric signal can mask a sign error in
/// an odd extension, and a constant one masks index shifts entirely.
fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (5.0 * t).sin() + 0.4 * (17.0 * t).cos() - 0.3 * t + 0.1
        })
        .collect()
}

/// Lengths spanning the routing families: powers of two, smooth composites,
/// primes, and `361 = 19²` with its neighbours — the length whose `2N = 722`
/// FFT returned relative error 0.997 while the composite-radix defect was
/// live, and the reason this work was blocked.
const LENGTHS: [usize; 12] = [16, 17, 24, 31, 32, 48, 64, 100, 127, 360, 361, 362];

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

#[test]
fn dct1_fast_matches_the_direct_kernel() {
    for n in LENGTHS {
        let x = signal(n);
        let (mut fast_out, mut direct_out) = (vec![0.0; n], vec![0.0; n]);
        fast::dct1_fast(&x, &mut fast_out);
        direct::dct1(&x, &mut direct_out);
        let err = max_abs_diff(&fast_out, &direct_out);
        assert!(
            err <= tolerance(n, &x),
            "DCT-I at n = {n}: fast path differs from direct by {err:e}, bound {:e}",
            tolerance(n, &x)
        );
    }
}

#[test]
fn dst1_fast_matches_the_direct_kernel() {
    for n in LENGTHS {
        let x = signal(n);
        let (mut fast_out, mut direct_out) = (vec![0.0; n], vec![0.0; n]);
        fast::dst1_fast(&x, &mut fast_out);
        direct::dst1(&x, &mut direct_out);
        let err = max_abs_diff(&fast_out, &direct_out);
        assert!(
            err <= tolerance(n, &x),
            "DST-I at n = {n}: fast path differs from direct by {err:e}, bound {:e}",
            tolerance(n, &x)
        );
    }
}

#[test]
fn dct4_fast_matches_the_direct_kernel() {
    for n in LENGTHS {
        let x = signal(n);
        let (mut fast_out, mut direct_out) = (vec![0.0; n], vec![0.0; n]);
        fast::dct4_fast(&x, &mut fast_out);
        direct::dct4(&x, &mut direct_out);
        let err = max_abs_diff(&fast_out, &direct_out);
        assert!(
            err <= tolerance(n, &x),
            "DCT-IV at n = {n}: fast path differs from direct by {err:e}, bound {:e}",
            tolerance(n, &x)
        );
    }
}

#[test]
fn dst4_fast_matches_the_direct_kernel() {
    for n in LENGTHS {
        let x = signal(n);
        let (mut fast_out, mut direct_out) = (vec![0.0; n], vec![0.0; n]);
        fast::dst4_fast(&x, &mut fast_out);
        direct::dst4(&x, &mut direct_out);
        let err = max_abs_diff(&fast_out, &direct_out);
        assert!(
            err <= tolerance(n, &x),
            "DST-IV at n = {n}: fast path differs from direct by {err:e}, bound {:e}",
            tolerance(n, &x)
        );
    }
}

/// The shared kernel must agree with the two single-output kernels exactly.
///
/// It runs the same FFT and reads both projections from one post-twiddled
/// value, so any difference means the pair and the singles have drifted apart
/// — the failure mode a shared kernel exists to prevent and can also cause.
#[test]
fn dct4_dst4_pair_matches_the_single_output_kernels() {
    for n in LENGTHS {
        let x = signal(n);
        let (mut pair_dct, mut pair_dst) = (vec![0.0; n], vec![0.0; n]);
        fast::dct4_dst4_fast(&x, &mut pair_dct, &mut pair_dst);

        let (mut single_dct, mut single_dst) = (vec![0.0; n], vec![0.0; n]);
        fast::dct4_fast(&x, &mut single_dct);
        fast::dst4_fast(&x, &mut single_dst);

        assert_eq!(pair_dct, single_dct, "DCT-IV pair vs single at n = {n}");
        assert_eq!(pair_dst, single_dst, "DST-IV pair vs single at n = {n}");
    }
}

/// DCT-IV and DST-IV are involutions up to `2/N`, an identity the direct
/// kernels satisfy by construction and the fast paths must inherit.
///
/// This is an independent oracle rather than a second differential check: it
/// tests the transform against its own algebra, so it would catch an error
/// present in both the fast and direct paths.
#[test]
fn type4_transforms_are_self_inverse_up_to_scale() {
    for n in LENGTHS {
        let x = signal(n);
        for (name, transform) in [
            ("DCT-IV", fast::dct4_fast as fn(&[f64], &mut [f64])),
            ("DST-IV", fast::dst4_fast as fn(&[f64], &mut [f64])),
        ] {
            let mut once = vec![0.0; n];
            transform(&x, &mut once);
            let mut twice = vec![0.0; n];
            transform(&once, &mut twice);
            let scale = 2.0 / n as f64;
            let err = twice
                .iter()
                .zip(&x)
                .map(|(y, x)| (y * scale - x).abs())
                .fold(0.0f64, f64::max);
            assert!(
                err <= tolerance(n, &x),
                "{name} at n = {n}: applying it twice missed the input by {err:e}"
            );
        }
    }
}
