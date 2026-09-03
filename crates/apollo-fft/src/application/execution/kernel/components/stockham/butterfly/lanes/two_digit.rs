//! The two-digit stages: the widths where one register straddles two Stockham
//! digits.
//!
//! A digit that owns two `k` has its inputs in two-sample runs, so at the
//! eight-lane f32 width a register spans two digits and the gather that builds
//! one input index across both is a 128-bit half interleave. The four-lane f64
//! width takes the same kernels with one digit per register and no combine at
//! all. Both stage shapes — the `groups == 4` pair and the `groups == 8`
//! triple — share that packing and the twiddle register it needs, so they
//! share a module (ADR 0045, ninth slice).

use core::ops::{Add, Mul, Sub};

use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

use super::super::stage::stage_triple_scalar_one_impl;

/// Builds `[w_j, w_j, w_{j+1}, w_{j+1}]` — each digit's twiddle repeated
/// for its two `k`.
///
/// Both halves come from broadcasts: a splat fills a register with one
/// twiddle, and the half blend keeps each digit in its own half, so the
/// register is built in registers rather than through a stack buffer, with
/// no cross-lane permute. The four-lane width holds one digit and needs the
/// splat alone.
#[expect(
    clippy::inline_always,
    reason = "an out-of-line call inside the stage loop clobbers every live vector register and leaves the dispatcher's target-feature scope"
)]
#[inline(always)]
pub(super) fn digit_pair_twiddles<T, A>(
    _simd: Simd<T, A>,
    twiddles: &[Complex<T>],
    j: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar + eunomia::layout::Pod,
    A: SimdArch + SimdKernel<T>,
    Complex<T>: eunomia::layout::Pod,
{
    let low = ComplexReg::<T, A>::splat(twiddles[j]);
    if A::LANE_COUNT / 4 <= 1 {
        return low;
    }
    let high = ComplexReg::<T, A>::splat(twiddles[j + 1]);
    // Each broadcast already holds its digit in both halves, so the two
    // digits combine with the in-lane blend rather than the cross-lane
    // gather `interleave_halves` performs.
    ComplexReg::from_interleaved(low.into_interleaved().blend_halves(high.into_interleaved()))
}

/// The pair stage at `groups == 4` (`half_groups == 2`): every digit owns
/// two `k`, so its eight inputs are adjacent and two digits share a
/// register — the two-sample runs of one input index come together with
/// one half interleave per register pair.
struct PairStageQuarterGroupsTwo<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    radix: usize,
    first_twiddles: &'a [Complex<T>],
    second_twiddles: &'a [Complex<T>],
}

impl<T> LaneKernel<T> for PairStageQuarterGroupsTwo<'_, T>
where
    T: LaneScalar + eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    type Output = ();

    #[inline]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let Self {
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
        } = self;
        let n = src.len();
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let digits = per_register / 2;
        debug_assert!(
            (1..=2).contains(&digits),
            "invariant: two-digit kernels serve four- and eight-lane registers"
        );
        // Every store offset is a multiple of the stage stride
        // (`quarter_n` for the pair, `eighth_n` for the triple), so the
        // chunked stores address the intended run only when that stride
        // is a whole number of registers: exactly when `radix` is a
        // multiple of the digits a register holds.
        let vector_end = if radix % digits == 0 { radix } else { 0 };

        if vector_end > 0 {
            // A register holds `digits` consecutive digits × two `k`; the
            // `8 · digits` consecutive inputs `src[8j..]` load as four
            // registers per digit pair, and one half interleave per pair
            // gathers each input index across the two digits. Outputs are
            // contiguous over `2j` and `quarter_n = 2 · radix` is a register
            // multiple once the loop runs.
            let src_view = simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(src));
            let mut dst_view = simd.view_mut(eunomia::layout::cast_slice_mut::<Complex<T>, T>(dst));
            for j in (0..vector_end).step_by(digits) {
                let base = (8 * j) / per_register;
                let load = |c: usize| Vector::from_view_chunk(&src_view, base + c);
                let (x0, x1, x2, x3) = if digits == 1 {
                    (load(0), load(1), load(2), load(3))
                } else {
                    let (x0, x1) = load(0).interleave_halves(load(2));
                    let (x2, x3) = load(1).interleave_halves(load(3));
                    (x0, x1, x2, x3)
                };
                let w1 = digit_pair_twiddles(simd, first_twiddles, j);
                let w2 = digit_pair_twiddles(simd, second_twiddles, j);
                let w3 = digit_pair_twiddles(simd, second_twiddles, j + radix);
                let x0 = ComplexReg::from_interleaved(x0);
                let x1 = ComplexReg::from_interleaved(x1);
                let x2 = ComplexReg::from_interleaved(x2) * w1;
                let x3 = ComplexReg::from_interleaved(x3) * w1;
                let a0 = x0 + x2;
                let a1 = x1 + x3;
                let b0 = x0 - x2;
                let b1 = x1 - x3;
                let c0 = a1 * w2;
                let c1 = b1 * w3;
                let mut store = |offset: usize, value: ComplexReg<T, A>| {
                    value
                        .into_interleaved()
                        .store_to_view_chunk(&mut dst_view, offset / per_register);
                };
                store(2 * j, a0 + c0);
                store(2 * j + half_n, a0 - c0);
                store(2 * j + quarter_n, b0 + c1);
                store(2 * j + half_n + quarter_n, b0 - c1);
            }
        }

        for j in vector_end..radix {
            for k in 0..2 {
                let x0 = src[8 * j + k];
                let x1 = src[8 * j + 2 + k];
                let x2 = src[8 * j + 4 + k] * first_twiddles[j];
                let x3 = src[8 * j + 6 + k] * first_twiddles[j];
                let a0 = x0 + x2;
                let a1 = x1 + x3;
                let b0 = x0 - x2;
                let b1 = x1 - x3;
                let c0 = a1 * second_twiddles[j];
                let c1 = b1 * second_twiddles[j + radix];
                dst[2 * j + k] = a0 + c0;
                dst[2 * j + k + half_n] = a0 - c0;
                dst[2 * j + k + quarter_n] = b0 + c1;
                dst[2 * j + k + half_n + quarter_n] = b0 - c1;
            }
        }
    }
}

/// [`stage_pair_lanes`] for `groups == 4` (`src.len() == 8 * radix`) at the
/// widths where two digits share a register.
#[inline]
pub(crate) fn stage_pair_quarter_groups_two_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    radix: usize,
    first_twiddles: &[Complex<T>],
    second_twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(PairStageQuarterGroupsTwo {
        src,
        dst,
        radix,
        first_twiddles,
        second_twiddles,
    })
    .is_some()
}

/// The triple stage at `groups == 8` (`quarter_groups == 2`): every digit
/// owns two `k`, so its sixteen inputs are adjacent and two digits share a
/// register at the eight-lane width.
struct TripleStageGroupsEight<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    radix: usize,
    first_twiddles: &'a [Complex<T>],
    second_twiddles: &'a [Complex<T>],
    third_twiddles: &'a [Complex<T>],
}

impl<T> LaneKernel<T> for TripleStageGroupsEight<'_, T>
where
    T: LaneScalar + eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    type Output = ();

    #[inline]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let Self {
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
        } = self;
        let n = src.len();
        let eighth_n = n >> 3;
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let digits = per_register / 2;
        debug_assert!(
            (1..=2).contains(&digits),
            "invariant: two-digit kernels serve four- and eight-lane registers"
        );
        // Every store offset is a multiple of the stage stride
        // (`quarter_n` for the pair, `eighth_n` for the triple), so the
        // chunked stores address the intended run only when that stride
        // is a whole number of registers: exactly when `radix` is a
        // multiple of the digits a register holds.
        let vector_end = if radix % digits == 0 { radix } else { 0 };

        if vector_end > 0 {
            // `16 · digits` consecutive inputs load as eight registers per
            // digit pair; one half interleave per register pair gathers each
            // input index across the two digits. Outputs are contiguous over
            // `2j`, and `eighth_n = 2 · radix` is a register multiple once the
            // loop runs.
            let src_view = simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(src));
            let mut dst_view = simd.view_mut(eunomia::layout::cast_slice_mut::<Complex<T>, T>(dst));
            for j in (0..vector_end).step_by(digits) {
                let base = (16 * j) / per_register;
                let load = |c: usize| Vector::from_view_chunk(&src_view, base + c);
                let x: [ComplexReg<T, A>; 8] = if digits == 1 {
                    core::array::from_fn(|i| ComplexReg::from_interleaved(load(i)))
                } else {
                    let mut x = [ComplexReg::<T, A>::zero(); 8];
                    for t in 0..4 {
                        let (even, odd) = load(t).interleave_halves(load(t + 4));
                        x[2 * t] = ComplexReg::from_interleaved(even);
                        x[2 * t + 1] = ComplexReg::from_interleaved(odd);
                    }
                    x
                };
                let w1 = digit_pair_twiddles(simd, first_twiddles, j);
                let w2a = digit_pair_twiddles(simd, second_twiddles, j);
                let w2b = digit_pair_twiddles(simd, second_twiddles, j + radix);
                let w3a = digit_pair_twiddles(simd, third_twiddles, j);
                let w3b = digit_pair_twiddles(simd, third_twiddles, j + radix);
                let w3c = digit_pair_twiddles(simd, third_twiddles, j + 2 * radix);
                let w3d = digit_pair_twiddles(simd, third_twiddles, j + 3 * radix);

                let x4 = x[4] * w1;
                let x5 = x[5] * w1;
                let x6 = x[6] * w1;
                let x7 = x[7] * w1;
                let s0 = x[0] + x4;
                let s1 = x[1] + x5;
                let s2 = x[2] + x6;
                let s3 = x[3] + x7;
                let d0 = x[0] - x4;
                let d1 = x[1] - x5;
                let d2 = x[2] - x6;
                let d3 = x[3] - x7;
                let t2 = s2 * w2a;
                let t3 = s3 * w2a;
                let p0 = s0 + t2;
                let p4 = s0 - t2;
                let p1 = s1 + t3;
                let p5 = s1 - t3;
                let u2 = d2 * w2b;
                let u3 = d3 * w2b;
                let p2 = d0 + u2;
                let p6 = d0 - u2;
                let p3 = d1 + u3;
                let p7 = d1 - u3;
                let q0 = p1 * w3a;
                let q1 = p3 * w3b;
                let q2 = p5 * w3c;
                let q3 = p7 * w3d;

                let mut store = |offset: usize, value: ComplexReg<T, A>| {
                    value
                        .into_interleaved()
                        .store_to_view_chunk(&mut dst_view, offset / per_register);
                };
                let out = 2 * j;
                store(out, p0 + q0);
                store(out + half_n, p0 - q0);
                store(out + quarter_n, p4 + q2);
                store(out + half_n + quarter_n, p4 - q2);
                store(out + eighth_n, p2 + q1);
                store(out + half_n + eighth_n, p2 - q1);
                store(out + quarter_n + eighth_n, p6 + q3);
                store(out + half_n + quarter_n + eighth_n, p6 - q3);
            }
        }

        for j in vector_end..radix {
            for k in 0..2 {
                stage_triple_scalar_one_impl(
                    src,
                    dst,
                    16 * j,
                    2 * j,
                    2,
                    eighth_n,
                    quarter_n,
                    half_n,
                    k,
                    first_twiddles[j],
                    second_twiddles[j],
                    second_twiddles[j + radix],
                    third_twiddles[j],
                    third_twiddles[j + radix],
                    third_twiddles[j + 2 * radix],
                    third_twiddles[j + 3 * radix],
                );
            }
        }
    }
}

/// [`stage_triple_lanes`] for `groups == 8` (`src.len() == 16 * radix`) at
/// the widths where a digit's two `k` or two digits share a register.
#[inline]
pub(crate) fn stage_triple_groups_eight_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    radix: usize,
    first_twiddles: &[Complex<T>],
    second_twiddles: &[Complex<T>],
    third_twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(TripleStageGroupsEight {
        src,
        dst,
        radix,
        first_twiddles,
        second_twiddles,
        third_twiddles,
    })
    .is_some()
}
