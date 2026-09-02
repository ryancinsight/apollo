//! The fused triple stage as a lane kernel: three radix-2 passes in one
//! traversal, eight outputs per `k` from `{w1, w2a, w2b, w3a, w3b, w3c, w3d}`.

use core::ops::{Add, Mul, Sub};

use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

use super::super::stage::{stage_triple_scalar_one_impl, stage_triple_scalar_one_j0_impl};
use super::pair::digit_pair_twiddles;

/// The general triple stage: `radix >= 2`, `src.len() / (2 * radix)` groups.
struct TripleStage<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    radix: usize,
    first_twiddles: &'a [Complex<T>],
    second_twiddles: &'a [Complex<T>],
    third_twiddles: &'a [Complex<T>],
}

/// The `radix == 1` triple stage: the second-pass twiddle and the third pass's
/// middle twiddle are exact quarter turns, so they rotate instead of multiply.
struct TripleStageRadixOne<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    second_twiddles: &'a [Complex<T>],
    third_twiddles: &'a [Complex<T>],
}

/// Multiplies by `+i` or `-i`, selected once per stage from the twiddle's sign.
#[inline]
fn quarter_turn<T, A>(v: ComplexReg<T, A>, positive: bool) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    if positive {
        v.mul_i()
    } else {
        v.mul_neg_i()
    }
}

impl<T> LaneKernel<T> for TripleStage<'_, T>
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
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
        let groups = n / (radix << 1);
        let quarter_groups = groups >> 2;
        let eighth_n = n >> 3;
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let vector_end = quarter_groups & !(per_register - 1);
        let src_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(src));

        for j in 0..radix {
            let w1 = first_twiddles[j];
            let w2a = second_twiddles[j];
            let w2b = second_twiddles[j + radix];
            let w3a = third_twiddles[j];
            let w3b = third_twiddles[j + radix];
            let w3c = third_twiddles[j + 2 * radix];
            let w3d = third_twiddles[j + 3 * radix];
            let src_base = j * groups * 2;
            let dst_base = j * quarter_groups;

            if vector_end > 0 {
                // The vector loop runs only when `quarter_groups >= per_register`;
                // both are powers of two, so every offset below — multiples of
                // `quarter_groups` plus a `k` stepped by `per_register` — is a
                // whole register index.
                let w1v = ComplexReg::<T, A>::splat(w1);
                let w2av = ComplexReg::<T, A>::splat(w2a);
                let w2bv = ComplexReg::<T, A>::splat(w2b);
                let w3av = ComplexReg::<T, A>::splat(w3a);
                let w3bv = ComplexReg::<T, A>::splat(w3b);
                let w3cv = ComplexReg::<T, A>::splat(w3c);
                let w3dv = ComplexReg::<T, A>::splat(w3d);
                let load = |offset: usize| {
                    ComplexReg::from_interleaved(Vector::from_view_chunk(
                        &src_view,
                        offset / per_register,
                    ))
                };
                let mut dst_view = simd.view_mut(bytemuck::cast_slice_mut::<Complex<T>, T>(dst));
                for k in (0..vector_end).step_by(per_register) {
                    let x0 = load(src_base + k);
                    let x2 = load(src_base + 2 * quarter_groups + k);
                    let x4 = load(src_base + groups + k) * w1v;
                    let x6 = load(src_base + groups + 2 * quarter_groups + k) * w1v;
                    let s0 = x0 + x4;
                    let s2 = x2 + x6;
                    let d0 = x0 - x4;
                    let d2 = x2 - x6;
                    let t2 = s2 * w2av;
                    let p0 = s0 + t2;
                    let p4 = s0 - t2;

                    let x1 = load(src_base + quarter_groups + k);
                    let x3 = load(src_base + 3 * quarter_groups + k);
                    let x5 = load(src_base + groups + quarter_groups + k) * w1v;
                    let x7 = load(src_base + groups + 3 * quarter_groups + k) * w1v;
                    let s1 = x1 + x5;
                    let s3 = x3 + x7;
                    let d1 = x1 - x5;
                    let d3 = x3 - x7;
                    let t3 = s3 * w2av;
                    let p1 = s1 + t3;
                    let p5 = s1 - t3;

                    let mut store = |offset: usize, value: ComplexReg<T, A>| {
                        value
                            .into_interleaved()
                            .store_to_view_chunk(&mut dst_view, offset / per_register);
                    };
                    let out = dst_base + k;
                    let q0 = p1 * w3av;
                    let q2 = p5 * w3cv;
                    store(out, p0 + q0);
                    store(half_n + out, p0 - q0);
                    store(quarter_n + out, p4 + q2);
                    store(half_n + quarter_n + out, p4 - q2);

                    let u2 = d2 * w2bv;
                    let u3 = d3 * w2bv;
                    let p2 = d0 + u2;
                    let p3 = d1 + u3;
                    let p6 = d0 - u2;
                    let p7 = d1 - u3;
                    let q1 = p3 * w3bv;
                    let q3 = p7 * w3dv;
                    store(eighth_n + out, p2 + q1);
                    store(half_n + eighth_n + out, p2 - q1);
                    store(quarter_n + eighth_n + out, p6 + q3);
                    store(half_n + quarter_n + eighth_n + out, p6 - q3);
                }
            }

            for k in vector_end..quarter_groups {
                stage_triple_scalar_one_impl(
                    src,
                    dst,
                    src_base,
                    dst_base,
                    quarter_groups,
                    eighth_n,
                    quarter_n,
                    half_n,
                    k,
                    w1,
                    w2a,
                    w2b,
                    w3a,
                    w3b,
                    w3c,
                    w3d,
                );
            }
        }
    }
}

impl<T> LaneKernel<T> for TripleStageRadixOne<'_, T>
where
    T: LaneScalar + bytemuck::Pod + Default + PartialOrd,
    Complex<T>: bytemuck::Pod
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
            second_twiddles,
            third_twiddles,
        } = self;
        let n = src.len();
        let eighth_n = n >> 3;
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let vector_end = eighth_n & !(per_register - 1);
        let w2b = second_twiddles[1];
        let w3b = third_twiddles[1];
        let w3c = third_twiddles[2];
        let w3d = third_twiddles[3];

        if vector_end > 0 {
            // `eighth_n >= per_register`, both powers of two, so every multiple
            // of `eighth_n` plus a stepped `k` is a whole register index. The
            // quarter turns `w2b` and `w3c` are `±i` exactly; only their sign
            // selects the rotation.
            let second_positive = w2b.im > T::default();
            let third_positive = w3c.im > T::default();
            let w3bv = ComplexReg::<T, A>::splat(w3b);
            let w3dv = ComplexReg::<T, A>::splat(w3d);
            let src_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(src));
            let mut dst_view = simd.view_mut(bytemuck::cast_slice_mut::<Complex<T>, T>(dst));
            let load = |offset: usize| {
                ComplexReg::from_interleaved(Vector::from_view_chunk(
                    &src_view,
                    offset / per_register,
                ))
            };
            for k in (0..vector_end).step_by(per_register) {
                let x0 = load(k);
                let x2 = load(2 * eighth_n + k);
                let x4 = load(4 * eighth_n + k);
                let x6 = load(6 * eighth_n + k);
                let s0 = x0 + x4;
                let s2 = x2 + x6;
                let d0 = x0 - x4;
                let d2 = x2 - x6;
                let u2 = quarter_turn(d2, second_positive);
                let p0 = s0 + s2;
                let p2 = d0 + u2;
                let p4 = s0 - s2;
                let p6 = d0 - u2;

                let x1 = load(eighth_n + k);
                let x3 = load(3 * eighth_n + k);
                let x5 = load(5 * eighth_n + k);
                let x7 = load(7 * eighth_n + k);
                let s1 = x1 + x5;
                let s3 = x3 + x7;
                let d1 = x1 - x5;
                let d3 = x3 - x7;
                let u3 = quarter_turn(d3, second_positive);
                let p1 = s1 + s3;
                let p3 = d1 + u3;
                let p5 = s1 - s3;
                let p7 = d1 - u3;

                let mut store = |offset: usize, value: ComplexReg<T, A>| {
                    value
                        .into_interleaved()
                        .store_to_view_chunk(&mut dst_view, offset / per_register);
                };
                let q2 = quarter_turn(p5, third_positive);
                store(k, p0 + p1);
                store(half_n + k, p0 - p1);
                store(quarter_n + k, p4 + q2);
                store(half_n + quarter_n + k, p4 - q2);

                let q1 = p3 * w3bv;
                let q3 = p7 * w3dv;
                store(eighth_n + k, p2 + q1);
                store(half_n + eighth_n + k, p2 - q1);
                store(quarter_n + eighth_n + k, p6 + q3);
                store(half_n + quarter_n + eighth_n + k, p6 - q3);
            }
        }

        for k in vector_end..eighth_n {
            stage_triple_scalar_one_j0_impl(
                src, dst, 0, 0, eighth_n, eighth_n, quarter_n, half_n, k, w2b, w3b, w3c, w3d,
            );
        }
    }
}

/// Runs the general triple stage (`radix >= 2`) at `LANES` scalar lanes per
/// register on the hardware backend of that width.
///
/// Returns `false`, having touched nothing, when this host has no hardware
/// backend at `LANES`; the caller then takes its scalar route.
#[inline]
pub(crate) fn stage_triple_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    radix: usize,
    first_twiddles: &[Complex<T>],
    second_twiddles: &[Complex<T>],
    third_twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(TripleStage {
        src,
        dst,
        radix,
        first_twiddles,
        second_twiddles,
        third_twiddles,
    })
    .is_some()
}

/// [`stage_triple_lanes`] for `radix == 1`, where `second_twiddles[1]` and
/// `third_twiddles[2]` are exact quarter turns.
#[inline]
pub(crate) fn stage_triple_radix_one_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    second_twiddles: &[Complex<T>],
    third_twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + bytemuck::Pod + Default + PartialOrd,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(TripleStageRadixOne {
        src,
        dst,
        second_twiddles,
        third_twiddles,
    })
    .is_some()
}

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
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
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
            let src_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(src));
            let first_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(first_twiddles));
            let second_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(second_twiddles));
            let third_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(third_twiddles));
            let mut dst_view = simd.view_mut(bytemuck::cast_slice_mut::<Complex<T>, T>(dst));
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
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
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
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
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
            let src_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(src));
            let mut dst_view = simd.view_mut(bytemuck::cast_slice_mut::<Complex<T>, T>(dst));
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
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
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
