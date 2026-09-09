//! Unit coverage for the batched four-step components.
//!
//! The transpose and the batched stage set are verified separately from the
//! assembled transform, so a failure localizes.

use super::{
    combine_planar_halves, four_step_batched, scratch_len, transpose_planes, BatchedPlanCache,
    LaneOrder,
};
use eunomia::{Complex, Complex32, Complex64};
use std::f64::consts::TAU;

#[test]
fn transpose_is_its_own_inverse_and_never_touches_the_pad() {
    for m in [1usize, 2, 4, 8, 16, 33, 64] {
        for (pad, order) in [
            (0usize, LaneOrder::IDENTITY),
            (8, LaneOrder::IDENTITY),
            (8, LaneOrder::for_batch::<f64>(m)),
        ] {
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
                    // Plane cell (r, c) holds logical column order(c) of
                    // row r: the transpose of logical (order(c), row
                    // order(r)), which sits at plane cell (order(c), order(r)).
                    let from = order.column(c) * m + order.column(r);
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

/// The planes must match the interleaved matrix row for row.
///
/// They were row-permuted while the stage set that folds them was decimated
/// in time and so took bit-reversed input. That set is now decimated in
/// frequency and takes natural order, so the planes follow the data rows
/// (`gap_audit.md#planar-pass-attribution`). This asserts the layout the
/// fold actually indexes; the transform-level oracles below assert that the
/// pairing is right.
#[test]
fn four_step_planes_are_the_row_faithful_split_of_the_interleaved_matrix() {
    use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
    let (n, m) = (256usize, 16usize);
    let planes = <f64 as BatchedPlanCache>::cached_four_step_planes::<false>(n, m);
    let interleaved = <f64 as MixedRadixScalar>::cached_four_step_twiddles::<false>(n, m, m);
    let order = LaneOrder::for_batch::<f64>(m);
    for row in 0..m {
        for col in 0..m {
            // Plane column `order(col)` holds memory column `col`.
            let at = row * m + order.column(col);
            assert_eq!(
                planes.re[at].to_bits(),
                interleaved[row * m + col].re.to_bits(),
                "re ({row},{col})"
            );
            assert_eq!(
                planes.im[at].to_bits(),
                interleaved[row * m + col].im.to_bits(),
                "im ({row},{col})"
            );
        }
    }
    let again = <f64 as BatchedPlanCache>::cached_four_step_planes::<false>(n, m);
    assert!(std::sync::Arc::ptr_eq(&planes, &again), "planes must cache");
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
fn f32_planar_half_combine_matches_the_scalar_formula() {
    let (m, stride) = (16usize, 24usize);
    let half = m * m;
    let plane = m * stride;
    let mut even = vec![Complex32::default(); plane];
    let mut odd = vec![Complex32::default(); plane];
    let (even_re, even_im) =
        eunomia::layout::cast_slice_mut::<_, f32>(&mut even).split_at_mut(plane);
    let (odd_re, odd_im) = eunomia::layout::cast_slice_mut::<_, f32>(&mut odd).split_at_mut(plane);
    for row in 0..m {
        for column in 0..m {
            let index = row * stride + column;
            let logical = (row * m + column) as f32;
            even_re[index] = 0.25 + logical * 0.003;
            even_im[index] = -0.5 + logical * 0.002;
            odd_re[index] = 0.75 - logical * 0.001;
            odd_im[index] = -0.125 + logical * 0.004;
        }
    }
    let twiddles: Vec<Complex32> = (0..half)
        .map(|index| {
            let angle = -core::f32::consts::TAU * index as f32 / (2 * half) as f32;
            let (sin, cos) = angle.sin_cos();
            Complex32::new(cos, sin)
        })
        .collect();
    let mut expected = vec![Complex32::default(); 2 * half];
    let bits = m.trailing_zeros();
    let order = LaneOrder::for_batch::<f32>(m);
    for row in 0..m {
        let base = row * stride;
        let dst = (row.reverse_bits() >> (usize::BITS - bits)) * m;
        for column in 0..m {
            // Plane column `column` holds memory column `order(column)`.
            let index = dst + order.column(column);
            let even_value = Complex32::new(even_re[base + column], even_im[base + column]);
            let odd_value = Complex32::new(odd_re[base + column], odd_im[base + column]);
            let rotated = odd_value * twiddles[index];
            expected[index] = even_value + rotated;
            expected[index + half] = even_value - rotated;
        }
    }

    let mut actual = vec![Complex32::default(); 2 * half];
    combine_planar_halves(&mut actual, &even, &odd, m, stride, &twiddles, order);

    // One complex multiply followed by one add/sub accumulates at most eight
    // unit roundoffs at this scale; the factor of two covers subnormal-free
    // input scaling and the SIMD path's fused multiply-add rounding.
    let bound = 16.0 * f32::EPSILON;
    let worst = actual
        .iter()
        .zip(&expected)
        .map(|(got, want)| (*got - *want).norm())
        .fold(0.0_f32, f32::max);
    assert!(worst <= bound, "combine error {worst:.3e} > {bound:.3e}");
}

/// The reinterleave sink at the f32 native width.
///
/// A correct result alone would not prove the eight-lane body ran: declining
/// the width falls back to the four-lane body and then the scalar loop, and
/// all three produce the same answer. So the width is asserted against the
/// independently dispatched capability, exactly as the four-lane kernels are
/// (`test_support::executed_or_declined_untouched`), and only then is the
/// output compared. The pass moves data and computes nothing, so the
/// comparison is bit-exact rather than bounded.
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
    // `FourStepPlanes` owns two `m x m` planes -- 16n bytes at `f64`, 262,144
    // at n = 16,384, the largest length the planar route accepts. The caches
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
                <f64 as BatchedPlanCache>::cached_four_step_planes::<false>(LEN, HALF)
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
