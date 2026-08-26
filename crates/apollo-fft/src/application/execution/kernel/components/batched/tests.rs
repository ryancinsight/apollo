//! Unit coverage for the batched four-step components.
//!
//! The transpose and the batched stage set are verified separately from the
//! assembled transform, so a failure localizes.

use super::{four_step_batched, scratch_len, twiddle_transpose, BatchedPlanCache};
use eunomia::Complex64;
use std::f64::consts::TAU;

#[test]
fn fused_twiddle_transpose_matches_the_separate_passes() {
    // Reference: the two passes the fused function replaced — an elementwise
    // multiply at the pre-transpose index, then a plain transpose. Synthetic
    // twiddle values (not roots of unity) so an index mix-up cannot cancel.
    for m in [1usize, 2, 4, 8, 16, 33, 64] {
        let re0: Vec<f64> = (0..m * m).map(|i| 0.5 + i as f64).collect();
        let im0: Vec<f64> = (0..m * m).map(|i| 0.25 - i as f64 * 0.5).collect();
        let tw: Vec<Complex64> = (0..m * m)
            .map(|i| Complex64::new(1.0 + 0.01 * i as f64, 0.02 * i as f64 - 0.3))
            .collect();

        let mut expected_re = vec![0.0f64; m * m];
        let mut expected_im = vec![0.0f64; m * m];
        for r in 0..m {
            for c in 0..m {
                let p = r * m + c;
                let t = tw[p];
                expected_re[c * m + r] = re0[p] * t.re - im0[p] * t.im;
                expected_im[c * m + r] = re0[p] * t.im + im0[p] * t.re;
            }
        }

        // Unpadded and padded strides run the same logical operation; both
        // must match the reference, and the padded run must never touch pad.
        for pad in [0usize, 8] {
            let stride = m + pad;
            let sentinel = f64::NAN;
            let mut re = vec![sentinel; m * stride];
            let mut im = vec![sentinel; m * stride];
            for r in 0..m {
                re[r * stride..r * stride + m].copy_from_slice(&re0[r * m..(r + 1) * m]);
                im[r * stride..r * stride + m].copy_from_slice(&im0[r * m..(r + 1) * m]);
            }
            twiddle_transpose(&mut re, &mut im, &tw, m, stride);
            for r in 0..m {
                for c in 0..m {
                    assert_eq!(
                        re[r * stride + c],
                        expected_re[r * m + c],
                        "m={m} pad={pad}: real ({r},{c}) differs"
                    );
                    assert_eq!(
                        im[r * stride + c],
                        expected_im[r * m + c],
                        "m={m} pad={pad}: imaginary ({r},{c}) differs"
                    );
                }
                for c in m..stride {
                    assert!(
                        re[r * stride + c].is_nan() && im[r * stride + c].is_nan(),
                        "m={m} pad={pad}: pad column {c} of row {r} was written"
                    );
                }
            }
        }
    }
}

/// Direct DFT, the analytical oracle for the assembled transform.
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

#[test]
fn forward_matches_the_direct_transform() {
    // Even powers only: the four-step gate admits square splits.
    for k in [2u32, 4, 6, 8, 10, 12] {
        let n = 1usize << k;
        let src = signal(n);
        let expected = dft(&src, false);
        let mut data = src.clone();
        let mut scratch = vec![Complex64::default(); scratch_len(n)];
        four_step_batched::<f64, false>(&mut data, &mut scratch);

        let bound = tolerance(n, &src);
        let worst = data
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
            .fold(0.0f64, f64::max);
        assert!(
            worst <= bound,
            "N={n}: forward differs by {worst:.3e} > {bound:.3e}"
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
        let mut scratch = vec![Complex64::default(); scratch_len(n)];
        four_step_batched::<f64, true>(&mut data, &mut scratch);

        let bound = tolerance(n, &src);
        let worst = data
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
            .fold(0.0f64, f64::max);
        assert!(
            worst <= bound,
            "N={n}: inverse differs by {worst:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn forward_then_inverse_recovers_the_input() {
    for k in [4u32, 6, 8, 10, 12] {
        let n = 1usize << k;
        let src = signal(n);
        let mut data = src.clone();
        let mut scratch = vec![Complex64::default(); scratch_len(n)];
        four_step_batched::<f64, false>(&mut data, &mut scratch);
        four_step_batched::<f64, true>(&mut data, &mut scratch);

        // The unnormalized round trip scales by N.
        let bound = tolerance(n, &src) * n as f64;
        let worst = data
            .iter()
            .zip(src.iter())
            .map(|(a, b)| (a.re - b.re * n as f64).hypot(a.im - b.im * n as f64))
            .fold(0.0f64, f64::max);
        assert!(
            worst <= bound,
            "N={n}: round trip differs by {worst:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn f32_forward_matches_the_direct_transform() {
    use eunomia::Complex32;

    for k in [4u32, 6, 8] {
        let n = 1usize << k;
        let src64 = signal(n);
        let src: Vec<Complex32> = src64
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        let expected = dft(&src64, false);
        let mut data = src.clone();
        let mut scratch = vec![Complex32::default(); scratch_len(n)];
        four_step_batched::<f32, false>(&mut data, &mut scratch);

        let l1: f64 = src64.iter().map(|v| v.re.hypot(v.im)).sum();
        let stages = f64::from(u32::try_from(n.trailing_zeros()).expect("fits u32"));
        let bound = 16.0 * stages * f64::from(f32::EPSILON / 2.0) * l1;
        let worst = data
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| {
                f64::from(a.re)
                    .hypot(0.0)
                    .mul_add(0.0, (f64::from(a.re) - b.re).hypot(f64::from(a.im) - b.im))
            })
            .fold(0.0f64, f64::max);
        assert!(
            worst <= bound,
            "N={n} f32: differs by {worst:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn plans_are_cached_per_length_and_direction() {
    let a = <f64 as BatchedPlanCache>::cached_plan::<false>(64);
    let b = <f64 as BatchedPlanCache>::cached_plan::<false>(64);
    assert!(
        std::sync::Arc::ptr_eq(&a, &b),
        "a repeated request must reuse the cached plan rather than rebuild it"
    );
    let inv = <f64 as BatchedPlanCache>::cached_plan::<true>(64);
    assert!(
        !std::sync::Arc::ptr_eq(&a, &inv),
        "forward and inverse plans carry conjugate twiddles and must not share"
    );
}
