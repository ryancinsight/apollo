//! Unit coverage for the batched four-step components.
//!
//! The transpose and the batched stage set are verified separately from the
//! assembled transform, so a failure localizes.

use super::{
    four_step_batched, scratch_len, transpose_planes, transpose_planes_into, BatchedPlanCache,
    LaneOrder, PlaneView,
};
use eunomia::{Complex, Complex32, Complex64};
use std::f64::consts::TAU;

#[test]
fn transpose_is_its_own_inverse_and_never_touches_the_pad() {
    for m in [1usize, 2, 4, 8, 16, 33, 64] {
        // The dispatched order, and the AVX-512 orders whatever the host,
        // where the order is no involution; a group must fit the row.
        let mut orders = vec![
            (0usize, LaneOrder::IDENTITY),
            (8, LaneOrder::IDENTITY),
            (8, LaneOrder::for_batch::<f64>(m)),
        ];
        if m % 8 == 0 {
            orders.push((8, LaneOrder::from_geometry(8, 2)));
        }
        if m % 16 == 0 {
            orders.push((8, LaneOrder::from_geometry(16, 4)));
        }
        for (pad, order) in orders {
            let stride = m + pad;
            let sentinel = f64::NAN;
            let re0: Vec<f64> = (0..m * m).map(|i| 0.5 + i as f64).collect();
            let im0: Vec<f64> = (0..m * m).map(|i| 0.25 - i as f64 * 0.5).collect();
            let mut re = vec![sentinel; m * stride];
            let mut im = vec![sentinel; m * stride];
            for r in 0..m {
                re[r * stride..r * stride + m].copy_from_slice(&re0[r * m..(r + 1) * m]);
                im[r * stride..r * stride + m].copy_from_slice(&im0[r * m..(r + 1) * m]);
            }
            transpose_planes(&mut re, &mut im, m, stride, order);
            for r in 0..m {
                for c in 0..m {
                    // Plane cell (r, c) holds memory (r, order(c)), the
                    // transpose of memory (order(c), r), which sat at plane
                    // cell (order(c), plane(r)).
                    let from = order.column(c) * m + order.plane(r);
                    assert_eq!(
                        re[r * stride + c],
                        re0[from],
                        "m={m} pad={pad} re ({r},{c})"
                    );
                    assert_eq!(
                        im[r * stride + c],
                        im0[from],
                        "m={m} pad={pad} im ({r},{c})"
                    );
                }
                for c in m..stride {
                    assert!(
                        re[r * stride + c].is_nan() && im[r * stride + c].is_nan(),
                        "m={m} pad={pad}: pad column {c} of row {r} was written"
                    );
                }
            }
            transpose_planes(&mut re, &mut im, m, stride, order);
            for r in 0..m {
                assert_eq!(&re[r * stride..r * stride + m], &re0[r * m..(r + 1) * m]);
            }
        }
    }
}

/// The two-level fold tables must reproduce the twiddle matrix entry for
/// entry within the roundings the factoring adds.
///
/// Each factor is one direct evaluation within one ulp (`EPSILON`) of
/// exact, as is the matrix entry it is held against; the product's two
/// roundings per part add `2 EPSILON`, so the difference is within
/// `5 EPSILON` (measured worst `4.03`); the bound is `6 EPSILON`. Rows are
/// natural, columns in plane order, as the fold indexes them.
#[test]
fn fold_tables_reproduce_the_twiddle_matrix_within_their_roundings() {
    use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
    let (n, m) = (256usize, 16usize);
    let fold = <f64 as BatchedPlanCache>::cached_four_step_fold::<false>(n, m, m);
    let interleaved = <f64 as MixedRadixScalar>::cached_four_step_twiddles::<false>(n, m, m);
    let order = LaneOrder::for_batch::<f64>(m);
    let lanes = fold.lanes;
    assert_eq!(
        lanes, m,
        "below the compact bound the fine row is the whole row"
    );
    let bound = 6.0 * f64::EPSILON;
    for row in 0..m {
        for col in 0..m {
            let k = order.column(col);
            let (fine, coarse) = (row * lanes + k % lanes, row * (m / lanes) + k / lanes);
            let entry = Complex64::new(fold.fine_re[fine], fold.fine_im[fine])
                * Complex64::new(fold.coarse_re[coarse], fold.coarse_im[coarse]);
            let exact = interleaved[row * m + col];
            let err = (entry.re - exact.re).hypot(entry.im - exact.im);
            assert!(err <= bound, "({row},{col}): {err:.3e} > {bound:.3e}");
        }
    }
    let again = <f64 as BatchedPlanCache>::cached_four_step_fold::<false>(n, m, m);
    assert!(
        std::sync::Arc::ptr_eq(&fold, &again),
        "the fold tables must cache"
    );
}

/// Above the compact bound the fine table is one lane group wide and the
/// coarse table carries the rest; the same entry-for-entry agreement holds
/// there, checked on the smallest compact length.
#[test]
fn compact_fold_tables_reproduce_the_twiddle_matrix_within_their_roundings() {
    use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
    let (n, m) = (1usize << 18, 512usize);
    let order = LaneOrder::for_batch::<f64>(m);
    let fold = super::FourStepFold::<f64>::new::<false>(n, m, m, order);
    let interleaved = <f64 as MixedRadixScalar>::cached_four_step_twiddles::<false>(n, m, m);
    let lanes = fold.lanes;
    assert_eq!(lanes, order.lanes());
    assert!(lanes < m, "the compact table has more than one lane group");
    let bound = 6.0 * f64::EPSILON;
    for row in 0..m {
        for col in 0..m {
            let k = order.column(col);
            let (fine, coarse) = (row * lanes + k % lanes, row * (m / lanes) + k / lanes);
            let entry = Complex64::new(fold.fine_re[fine], fold.fine_im[fine])
                * Complex64::new(fold.coarse_re[coarse], fold.coarse_im[coarse]);
            let exact = interleaved[row * m + col];
            let err = (entry.re - exact.re).hypot(entry.im - exact.im);
            assert!(err <= bound, "({row},{col}): {err:.3e} > {bound:.3e}");
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

/// The odd-power split route runs both stage sets without seams, and a
/// seam that reads its staged source whenever stage 2 is present, rather
/// than when a source exists, escaped the impulse and round-trip checks:
/// the output was a consistent permutation, which an impulse cannot see and
/// an inverse undoes. The split lengths are therefore held to the direct
/// transform, 2048 with two sweeps per half and 8192, the Bluestein padding
/// of the prime squares that first exposed it.
#[test]
fn odd_lengths_match_the_direct_transform() {
    for n in [512usize, 2048, 8192] {
        let input = signal(n);
        let mut data = input.clone();
        let mut scratch = vec![Complex64::default(); scratch_len(n)];
        four_step_batched::<f64, false>(&mut data, &mut scratch);
        let expected = dft(&input, false);
        let err = data
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
            .fold(0.0_f64, f64::max);
        let bound = tolerance(n, &input);
        assert!(err <= bound, "n={n}: {err:.3e} > {bound:.3e}");
    }
}

#[test]
fn f32_n32768_public_plan_matches_an_impulse_and_round_trip() {
    let n = 32_768usize;
    let source_index = 13usize;
    let source = Complex32::new(0.75, -0.25);
    let mut actual = vec![Complex32::default(); n];
    actual[source_index] = source;
    let plan = crate::FftPlan1D::<f32>::new(
        crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
    );

    plan.forward_complex_slice_inplace(&mut actual);

    let stages = n.trailing_zeros() as f32;
    // Each radix level contributes at most one complex twiddle multiply and
    // two butterfly additions. Bounding each complex operation by sixteen
    // roundoffs, then doubling for twiddle construction and comparison,
    // gives 64 * log2(N) * epsilon * |source|.
    let forward_bound = 64.0 * stages * f32::EPSILON * source.norm();
    let forward_error = actual
        .iter()
        .enumerate()
        .map(|(frequency, got)| {
            let phase =
                -core::f32::consts::TAU * ((frequency * source_index) % n) as f32 / n as f32;
            let (sin, cos) = phase.sin_cos();
            let expected = Complex32::new(
                source.re.mul_add(cos, -(source.im * sin)),
                source.re.mul_add(sin, source.im * cos),
            );
            (*got - expected).norm()
        })
        .fold(0.0_f32, f32::max);
    assert!(
        forward_error <= forward_bound,
        "N={n} f32 impulse differs by {forward_error:.3e} > {forward_bound:.3e}"
    );

    plan.inverse_complex_slice_inplace(&mut actual);

    // Forward and inverse each satisfy the bound above; normalization adds
    // one exactly representable power-of-two scale at this length.
    let round_trip_bound = 128.0 * stages * f32::EPSILON * source.norm();
    let round_trip_error = actual
        .iter()
        .enumerate()
        .map(|(index, got)| {
            let expected = if index == source_index {
                source
            } else {
                Complex32::default()
            };
            (*got - expected).norm()
        })
        .fold(0.0_f32, f32::max);
    assert!(
        round_trip_error <= round_trip_bound,
        "N={n} f32 round trip differs by {round_trip_error:.3e} > {round_trip_bound:.3e}"
    );
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

#[test]
fn batched_plans_and_planes_are_shared_across_threads() {
    // `FourStepFold` owns two-level tables, `m (F + m / F)` pairs, and the
    // caches
    // are keyed per thread; if a thread that misses builds its own table
    // instead of taking the shared one, that storage exists once per thread
    // that touches the length and nothing evicts it.
    //
    // Both handles are spawned before either is joined, and both `Arc`s stay
    // live across the comparison: a per-thread cache dies with its thread, so
    // comparing raw addresses of dropped allocations could match by reuse.
    // `Arc::ptr_eq` on two live handles cannot.
    const LEN: usize = 1 << 12;
    const HALF: usize = 1 << 6;

    let plan_handles: Vec<_> = (0..2)
        .map(|_| std::thread::spawn(|| <f64 as BatchedPlanCache>::cached_plan::<false>(LEN)))
        .collect();
    let plans: Vec<_> = plan_handles
        .into_iter()
        .map(|handle| handle.join().expect("plan builder thread must not panic"))
        .collect();
    assert!(
        std::sync::Arc::ptr_eq(&plans[0], &plans[1]),
        "each thread built its own {LEN}-point batched plan"
    );

    let planes_handles: Vec<_> = (0..2)
        .map(|_| {
            std::thread::spawn(|| {
                <f64 as BatchedPlanCache>::cached_four_step_fold::<false>(LEN, HALF, HALF)
            })
        })
        .collect();
    let planes: Vec<_> = planes_handles
        .into_iter()
        .map(|handle| handle.join().expect("planes builder thread must not panic"))
        .collect();
    assert!(
        std::sync::Arc::ptr_eq(&planes[0], &planes[1]),
        "each thread built its own {LEN}-point four-step planes, duplicating          {} bytes of plane storage per thread",
        2 * HALF * HALF * core::mem::size_of::<f64>()
    );
}

/// Differential check at the lengths the planar domain gained: RustFFT is an
/// independent implementation whose forward error carries the same
/// `O(log N · u)` bound, so the distance between the two is at most twice
/// [`tolerance`] scaled to the precision under test.
fn large_lengths_agree_with_rustfft<F>(unit_roundoff: f64)
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<Complex = Complex<F>>
        + crate::application::orchestration::cache::plans::PlanCacheProvider<PlanScalar = F>
        + eunomia::FloatElement
        + rustfft::FftNum,
{
    for k in [16u32, 18] {
        let n = 1usize << k;
        assert!(
            super::planar_applies(n),
            "n = {n} is inside the planar domain"
        );
        let source = signal(n);
        let input: Vec<Complex<F>> = source
            .iter()
            .map(|z| {
                Complex::new(
                    <F as eunomia::FloatElement>::from_f64(z.re),
                    <F as eunomia::FloatElement>::from_f64(z.im),
                )
            })
            .collect();

        let mut actual = input.clone();
        crate::FftPlan1D::<F>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        )
        .forward_complex_slice_inplace(&mut actual);

        let mut expected: Vec<rustfft::num_complex::Complex<F>> = input
            .iter()
            .map(|z| rustfft::num_complex::Complex::new(z.re, z.im))
            .collect();
        rustfft::FftPlanner::<F>::new()
            .plan_fft_forward(n)
            .process(&mut expected);

        let l1: f64 = source.iter().map(|v| v.re.hypot(v.im)).sum();
        let stages = f64::from(k);
        let bound = 2.0 * 16.0 * stages * unit_roundoff * l1;
        for (bin, (a, e)) in actual.iter().zip(&expected).enumerate() {
            let error = (Complex64::new(a.re.to_f64(), a.im.to_f64())
                - Complex64::new(e.re.to_f64(), e.im.to_f64()))
            .norm();
            assert!(error <= bound, "n={n}, bin={bin}: {error:e} > {bound:e}");
        }
    }
}

#[test]
fn large_planar_lengths_agree_with_rustfft_in_both_precisions() {
    large_lengths_agree_with_rustfft::<f32>(f64::from(f32::EPSILON) / 2.0);
    large_lengths_agree_with_rustfft::<f64>(f64::EPSILON / 2.0);
}

/// The rectangular transpose lands every cell where the in-place rule
/// puts it, on the dispatched width and the scalar reference alike.
fn rectangular_transpose_matches_the_reference<T>()
where
    T: BatchedPlanCache<Complex = Complex<T>>
        + eunomia::FloatElement
        + PartialEq
        + core::fmt::Debug,
{
    // 24 rows: three eight-lane tiles, so the unpaired tile block runs.
    for (rows, cols) in [(16usize, 32usize), (32, 64), (24, 40)] {
        let (src_stride, dst_stride) = (cols + 8, rows + 8);
        let order = LaneOrder::for_batch::<T>(rows);
        let src_re: Vec<T> = (0..rows * src_stride)
            .map(|i| T::from_f64(i as f64 + 0.25))
            .collect();
        let src_im: Vec<T> = (0..rows * src_stride)
            .map(|i| T::from_f64(-(i as f64) - 0.5))
            .collect();
        let sentinel = T::from_f64(-1.0);
        let mut expected_re = vec![sentinel; cols * dst_stride];
        let mut expected_im = vec![sentinel; cols * dst_stride];
        transpose_planes_into(
            PlaneView {
                re: &src_re,
                im: &src_im,
                rows,
                cols,
                stride: src_stride,
            },
            &mut expected_re,
            &mut expected_im,
            dst_stride,
            order,
        );
        for r in 0..rows {
            for c in 0..cols {
                let to = order.column(c) * dst_stride + order.plane(r);
                assert_eq!(expected_re[to], src_re[r * src_stride + c]);
            }
        }
        let mut actual_re = vec![sentinel; cols * dst_stride];
        let mut actual_im = vec![sentinel; cols * dst_stride];
        let handled = hermes_simd::vectorize(super::boundary::TransposePlanesInto {
            src_re: &src_re,
            src_im: &src_im,
            rows,
            cols,
            src_stride,
            dst_re: &mut actual_re,
            dst_im: &mut actual_im,
            dst_stride,
        });
        assert!(handled, "a two-lane or wider backend handles {rows}x{cols}");
        assert_eq!(actual_re, expected_re, "{rows}x{cols} re");
        assert_eq!(actual_im, expected_im, "{rows}x{cols} im");
    }
}

#[test]
fn rectangular_transpose_matches_the_reference_in_both_precisions() {
    rectangular_transpose_matches_the_reference::<f32>();
    rectangular_transpose_matches_the_reference::<f64>();
}

#[test]
fn rectangular_fold_table_reproduces_the_twiddle_matrix() {
    // 2048 = 32 × 64: the second set's planes are 64 rows of 32 columns,
    // below the compact bound, so the fine table is the whole row.
    let (n, rows, cols) = (2048usize, 64usize, 32usize);
    let order = LaneOrder::for_batch::<f64>(cols);
    let fold = super::FourStepFold::<f64>::new::<false>(n, rows, cols, order);
    assert_eq!(fold.lanes, cols);
    for p in 0..rows {
        for k in 0..cols {
            let angle = -TAU * ((p * order.column(k)) % n) as f64 / n as f64;
            let (sin, cos) = angle.sin_cos();
            let got = Complex64::new(fold.fine_re[p * cols + k], fold.fine_im[p * cols + k]);
            let error = (got - Complex64::new(cos, sin)).norm();
            assert!(error <= 6.0 * f64::EPSILON, "({p},{k}): {error:e}");
        }
    }
}
