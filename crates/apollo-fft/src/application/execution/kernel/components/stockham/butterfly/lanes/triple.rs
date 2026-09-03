//! The fused triple stage as a lane kernel: three radix-2 passes in one
//! traversal, eight outputs per `k` from `{w1, w2a, w2b, w3a, w3b, w3c, w3d}`.

use core::ops::{Add, Mul, Sub};

use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

use super::super::stage::{stage_triple_scalar_one_impl, stage_triple_scalar_one_j0_impl};

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
        let groups = n / (radix << 1);
        let quarter_groups = groups >> 2;
        let eighth_n = n >> 3;
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let vector_end = quarter_groups & !(per_register - 1);
        let src_view = simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(src));

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
                let mut dst_view =
                    simd.view_mut(eunomia::layout::cast_slice_mut::<Complex<T>, T>(dst));
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
    T: LaneScalar + eunomia::layout::Pod + Default + PartialOrd,
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
            let src_view = simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(src));
            let mut dst_view = simd.view_mut(eunomia::layout::cast_slice_mut::<Complex<T>, T>(dst));
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
    T: LaneScalar + eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod
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
    T: LaneScalar + eunomia::layout::Pod + Default + PartialOrd,
    Complex<T>: eunomia::layout::Pod
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
