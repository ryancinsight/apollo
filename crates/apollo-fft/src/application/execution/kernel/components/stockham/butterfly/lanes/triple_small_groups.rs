//! The triple stage at `groups == 4`, where every digit owns one `k`.
//!
//! A register of consecutive digits reads its sixteen inputs as eight
//! consecutive registers and splits them into the stride-8 subsequences; the
//! twiddles and outputs are contiguous. Its two-digit sibling — the same stage
//! at `groups == 8`, where a register straddles two digits — lives in
//! [`super::two_digit`] (ADR 0045, eighth and ninth slices).

use core::ops::{Add, Mul, Sub};

use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

use super::super::stage::stage_triple_scalar_one_impl;

/// The triple stage at `groups == 4` (`quarter_groups == 1`): every digit `j`
/// owns one `k`, so its eight inputs are the adjacent samples `src[8j..8j + 8]`
/// and its eight outputs the strided `dst[j + m · n/8]`.
struct TripleStageQuarterGroupsOne<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    radix: usize,
    first_twiddles: &'a [Complex<T>],
    second_twiddles: &'a [Complex<T>],
    third_twiddles: &'a [Complex<T>],
}

/// Splits eight registers of consecutive samples into the eight stride-8
/// subsequences: a 4-way split of each half, then a 2-way split across the
/// halves, so output `i` holds the samples congruent to `i` modulo 8.
#[inline]
fn deinterleave_pairs8<T, A>(v: [Vector<T, A>; 8]) -> [Vector<T, A>; 8]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let (a0, a1, a2, a3) = v[0].deinterleave_pairs4(v[1], v[2], v[3]);
    let (b0, b1, b2, b3) = v[4].deinterleave_pairs4(v[5], v[6], v[7]);
    let (x0, x4) = a0.deinterleave_pairs(b0);
    let (x1, x5) = a1.deinterleave_pairs(b1);
    let (x2, x6) = a2.deinterleave_pairs(b2);
    let (x3, x7) = a3.deinterleave_pairs(b3);
    [x0, x1, x2, x3, x4, x5, x6, x7]
}

impl<T> LaneKernel<T> for TripleStageQuarterGroupsOne<'_, T>
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
        let vector_end = radix & !(per_register - 1);

        if vector_end > 0 {
            // A register of `per_register` consecutive `j` reads the
            // `8 · per_register` consecutive inputs `src[8j..]` as eight
            // registers and splits them into `x0..x7`; twiddles and outputs
            // are contiguous in `j`, and `eighth_n = radix >= per_register`
            // keeps every store aligned.
            let src_view = simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(src));
            let first_view =
                simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(first_twiddles));
            let second_view = simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(
                second_twiddles,
            ));
            let third_view =
                simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(third_twiddles));
            let mut dst_view = simd.view_mut(eunomia::layout::cast_slice_mut::<Complex<T>, T>(dst));
            for j in (0..vector_end).step_by(per_register) {
                let base = (8 * j) / per_register;
                let x = deinterleave_pairs8(core::array::from_fn(|i| {
                    Vector::from_view_chunk(&src_view, base + i)
                }))
                .map(ComplexReg::from_interleaved);
                let chunk = j / per_register;
                let tw = |view: &_, offset: usize| {
                    ComplexReg::<T, A>::from_interleaved(Vector::from_view_chunk(
                        view,
                        (j + offset) / per_register,
                    ))
                };
                let w1 = ComplexReg::from_interleaved(Vector::from_view_chunk(&first_view, chunk));
                let w2a = tw(&second_view, 0);
                let w2b = tw(&second_view, radix);
                let w3a = tw(&third_view, 0);
                let w3b = tw(&third_view, radix);
                let w3c = tw(&third_view, 2 * radix);
                let w3d = tw(&third_view, 3 * radix);

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
                store(j, p0 + q0);
                store(j + half_n, p0 - q0);
                store(j + quarter_n, p4 + q2);
                store(j + half_n + quarter_n, p4 - q2);
                store(j + eighth_n, p2 + q1);
                store(j + half_n + eighth_n, p2 - q1);
                store(j + quarter_n + eighth_n, p6 + q3);
                store(j + half_n + quarter_n + eighth_n, p6 - q3);
            }
        }

        for j in vector_end..radix {
            stage_triple_scalar_one_impl(
                src,
                dst,
                8 * j,
                j,
                1,
                eighth_n,
                quarter_n,
                half_n,
                0,
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

/// [`stage_triple_lanes`] for `groups == 4` (`src.len() == 8 * radix`), where
/// each digit's inputs are adjacent and the general kernel's per-group loop
/// would run entirely in its scalar tail.
#[inline]
pub(crate) fn stage_triple_quarter_groups_one_lanes<T, const LANES: usize>(
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
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(TripleStageQuarterGroupsOne {
        src,
        dst,
        radix,
        first_twiddles,
        second_twiddles,
        third_twiddles,
    })
    .is_some()
}
