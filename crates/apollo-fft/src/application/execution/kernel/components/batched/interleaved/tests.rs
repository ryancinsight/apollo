//! Correctness for the interleaved in-place four-step.
//!
//! Same oracle set as the planar sibling — a direct DFT is the analytical
//! authority — plus a differential against the planar kernel itself, since
//! the two must remain interchangeable behind one gate.

use super::four_step_interleaved;
use crate::application::execution::kernel::components::batched;
use eunomia::{Complex32, Complex64};
use std::f64::consts::TAU;

fn dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (t, v) in input.iter().enumerate() {
                let (s, c) = (sign * TAU * ((k * t) % n) as f64 / n as f64).sin_cos();
                re += v.re * c - v.im * s;
                im += v.re * s + v.im * c;
            }
            Complex64::new(re, im)
        })
        .collect()
}

fn signal(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

/// Bound from the `O(log N · u)` forward-error result with `|X_k| <= ||x||_1`.
fn tolerance(n: usize, input: &[Complex64]) -> f64 {
    let l1: f64 = input.iter().map(|v| v.re.hypot(v.im)).sum();
    let stages = f64::from(u32::try_from(n.trailing_zeros()).expect("fits u32"));
    16.0 * stages * (f64::EPSILON / 2.0) * l1
}

fn worst(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.re - y.re).hypot(x.im - y.im))
        .fold(0.0f64, f64::max)
}

#[test]
fn forward_matches_the_direct_transform() {
    for k in [2u32, 4, 6, 8, 10, 12] {
        let n = 1usize << k;
        let src = signal(n);
        let expected = dft(&src, false);
        let mut data = src.clone();
        four_step_interleaved::<f64, false>(&mut data);
        let (err, bound) = (worst(&data, &expected), tolerance(n, &src));
        assert!(
            err <= bound,
            "N={n}: forward differs by {err:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn inverse_matches_the_direct_transform() {
    for k in [2u32, 4, 6, 8, 10] {
        let n = 1usize << k;
        let src = signal(n);
        let expected = dft(&src, true);
        let mut data = src.clone();
        four_step_interleaved::<f64, true>(&mut data);
        let (err, bound) = (worst(&data, &expected), tolerance(n, &src));
        assert!(
            err <= bound,
            "N={n}: inverse differs by {err:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn forward_then_inverse_recovers_the_input() {
    for k in [4u32, 6, 8, 10, 12] {
        let n = 1usize << k;
        let src = signal(n);
        let mut data = src.clone();
        four_step_interleaved::<f64, false>(&mut data);
        four_step_interleaved::<f64, true>(&mut data);
        // The unnormalized round trip scales by N.
        let bound = tolerance(n, &src) * n as f64;
        let err = data
            .iter()
            .zip(src.iter())
            .map(|(a, b)| (a.re - b.re * n as f64).hypot(a.im - b.im * n as f64))
            .fold(0.0f64, f64::max);
        assert!(
            err <= bound,
            "N={n}: round trip differs by {err:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn f32_forward_matches_the_direct_transform() {
    for k in [2u32, 4, 6, 8] {
        let n = 1usize << k;
        let src64 = signal(n);
        let mut data: Vec<Complex32> = src64
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        let expected = dft(&src64, false);
        four_step_interleaved::<f32, false>(&mut data);

        let l1: f64 = src64.iter().map(|v| v.re.hypot(v.im)).sum();
        let stages = f64::from(k);
        let bound = 16.0 * stages * f64::from(f32::EPSILON / 2.0) * l1;
        let err = data
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (f64::from(a.re) - b.re).hypot(f64::from(a.im) - b.im))
            .fold(0.0f64, f64::max);
        assert!(
            err <= bound,
            "N={n} f32: differs by {err:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn matches_the_planar_kernel_within_rounding() {
    for k in [4u32, 8, 12, 14] {
        let n = 1usize << k;
        let src = signal(n);
        let mut ours = src.clone();
        four_step_interleaved::<f64, false>(&mut ours);

        let mut theirs = src.clone();
        let mut scratch = vec![Complex64::default(); batched::scratch_len(n)];
        batched::four_step_batched::<f64, false>(&mut theirs, &mut scratch);

        // Different evaluation shapes (interleaved FMA order against planar),
        // each inside the stage bound, so the differential carries both.
        let bound = 2.0 * tolerance(n, &src);
        let err = worst(&ours, &theirs);
        assert!(
            err <= bound,
            "N={n}: kernels differ by {err:.3e} > {bound:.3e}"
        );
    }
}
