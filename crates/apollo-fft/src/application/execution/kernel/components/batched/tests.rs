//! Unit coverage for the batched four-step components.
//!
//! The transpose and the batched stage set are verified separately from the
//! assembled transform, so a failure localizes.

use super::{
    combine_planar_halves, four_step_batched, scratch_len, transpose_planes, BatchedPlanCache,
};
use eunomia::{Complex32, Complex64};
use std::f64::consts::TAU;

#[test]
fn transpose_is_its_own_inverse_and_never_touches_the_pad() {
    for m in [1usize, 2, 4, 8, 16, 33, 64] {
        for pad in [0usize, 8] {
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
            transpose_planes(&mut re, &mut im, m, stride);
            for r in 0..m {
                for c in 0..m {
                    assert_eq!(
                        re[r * stride + c],
                        re0[c * m + r],
                        "m={m} pad={pad} re ({r},{c})"
                    );
                    assert_eq!(
                        im[r * stride + c],
                        im0[c * m + r],
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
            transpose_planes(&mut re, &mut im, m, stride);
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
    for row in 0..m {
        for col in 0..m {
            assert_eq!(
                planes.re[row * m + col].to_bits(),
                interleaved[row * m + col].re.to_bits(),
                "re ({row},{col})"
            );
            assert_eq!(
                planes.im[row * m + col].to_bits(),
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
    let (even_re, even_im) = bytemuck::cast_slice_mut::<_, f32>(&mut even).split_at_mut(plane);
    let (odd_re, odd_im) = bytemuck::cast_slice_mut::<_, f32>(&mut odd).split_at_mut(plane);
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
    for row in 0..m {
        let base = row * stride;
        let dst = (row.reverse_bits() >> (usize::BITS - bits)) * m;
        for column in 0..m {
            let index = dst + column;
            let even_value = Complex32::new(even_re[base + column], even_im[base + column]);
            let odd_value = Complex32::new(odd_re[base + column], odd_im[base + column]);
            let rotated = odd_value * twiddles[index];
            expected[index] = even_value + rotated;
            expected[index + half] = even_value - rotated;
        }
    }

    let mut actual = vec![Complex32::default(); 2 * half];
    combine_planar_halves(&mut actual, &even, &odd, m, stride, &twiddles);

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
fn f32_reinterleave_takes_the_native_width_and_matches_the_scalar_sink() {
    // `plane_geometry(256)`: sixteen live columns on a stride of `m + ROW_PAD`.
    let (m, stride) = (16usize, 24usize);
    let plane = m * stride;
    let mut re = vec![0.0f32; plane];
    let mut im = vec![0.0f32; plane];
    // The pad keeps its sentinel: reading it would land a value the scalar
    // reference never writes, so a pad touch fails the comparison below.
    for row in 0..m {
        for column in 0..m {
            let index = row * stride + column;
            let logical = (row * m + column) as f32;
            re[index] = 0.25 + logical * 0.003;
            im[index] = -0.5 + logical * 0.002;
        }
    }
    for slot in re.iter_mut().chain(im.iter_mut()) {
        if *slot == 0.0 {
            *slot = f32::from_bits(0x7f80_0001);
        }
    }

    let bits = m.trailing_zeros();
    let mut expected = vec![Complex32::default(); m * m];
    for row in 0..m {
        let src = row * stride;
        let dst = (row.reverse_bits() >> (usize::BITS - bits)) * m;
        for column in 0..m {
            expected[dst + column] = Complex32::new(re[src + column], im[src + column]);
        }
    }

    let mut actual = vec![Complex32::default(); m * m];
    let handled = hermes_simd::vectorize_lanes::<8, f32, _>(super::boundary::InterleaveRows {
        re: &re,
        im: &im,
        data: bytemuck::cast_slice_mut(&mut actual),
        m,
        stride,
    });
    let available = super::super::lane_capability::native_lanes_supported::<8, f32>();
    assert_eq!(
        handled,
        available.then_some(true),
        "the sink must take exactly the eight f32 lanes the dispatcher reports"
    );
    if !available {
        return;
    }
    for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            (got.re.to_bits(), got.im.to_bits()),
            (want.re.to_bits(), want.im.to_bits()),
            "eight-lane sink differs from the scalar sink at output {index}"
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

#[test]
fn batched_plans_and_planes_are_shared_across_threads() {
    // `BatchedPlan` owns `tw: Vec<(T, T)>` of `len - 1` entries and
    // `FourStepPlanes` owns two `Box<[T]>` planes, so each is O(16n) bytes for
    // `f64`. The caches are keyed per thread; if a thread that misses builds
    // its own table instead of taking the shared one, retention multiplies by
    // the worker count of whatever executor drives the transform -- 24 on this
    // host, so roughly 200 MB for one length at n = 262,144. Pointer identity
    // is the direct evidence: one allocation, or one per thread.
    const LEN: usize = 1 << 12;
    const HALF: usize = 1 << 6;

    let plan_addresses: Vec<usize> = (0..2)
        .map(|_| {
            std::thread::spawn(|| {
                std::sync::Arc::as_ptr(&<f64 as BatchedPlanCache>::cached_plan::<false>(LEN))
                    as usize
            })
        })
        .map(|handle| handle.join().expect("plan builder thread must not panic"))
        .collect();
    assert_eq!(
        plan_addresses[0],
        plan_addresses[1],
        "each thread built its own {LEN}-point batched plan, duplicating \
         {} bytes of twiddles per thread",
        (LEN - 1) * core::mem::size_of::<(f64, f64)>()
    );

    let planes_addresses: Vec<usize> = (0..2)
        .map(|_| {
            std::thread::spawn(|| {
                std::sync::Arc::as_ptr(
                    &<f64 as BatchedPlanCache>::cached_four_step_planes::<false>(LEN, HALF),
                ) as usize
            })
        })
        .map(|handle| handle.join().expect("planes builder thread must not panic"))
        .collect();
    assert_eq!(
        planes_addresses[0], planes_addresses[1],
        "each thread built its own {LEN}-point four-step planes"
    );
}
