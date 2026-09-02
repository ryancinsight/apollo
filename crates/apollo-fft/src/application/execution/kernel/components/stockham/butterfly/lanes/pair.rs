//! The fused pair stage as a lane kernel: two radix-2 passes in one
//! traversal, four outputs per `k` from `{w_j, w_j', w_{j+r}'}`.

use core::ops::{Add, Mul, Sub};

use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

/// The general pair stage: `radix >= 2`, `src.len() / (2 * radix)` groups.
struct PairStage<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    radix: usize,
    first_twiddles: &'a [Complex<T>],
    second_twiddles: &'a [Complex<T>],
}

/// The `radix == 1` pair stage, which needs only the second pass's twiddle.
struct PairStageRadixOne<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    second_twiddles: &'a [Complex<T>],
}

impl<T> LaneKernel<T> for PairStage<'_, T>
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
        } = self;
        let n = src.len();
        let groups = n / (radix << 1);
        let half_groups = groups >> 1;
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let vector_end = half_groups & !(per_register - 1);
        let src_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(src));

        for j in 0..radix {
            let w1 = first_twiddles[j];
            let w2 = second_twiddles[j];
            let w3 = second_twiddles[j + radix];
            let src_base = j * groups * 2;
            let dst_base = j * half_groups;

            if vector_end > 0 {
                // The vector loop runs only when `half_groups >= per_register`;
                // both are powers of two, so every offset below — multiples of
                // `half_groups` plus a `k` stepped by `per_register` — is a
                // whole register index.
                let w1v = ComplexReg::<T, A>::splat(w1);
                let w2v = ComplexReg::<T, A>::splat(w2);
                let w3v = ComplexReg::<T, A>::splat(w3);
                let load = |offset: usize| {
                    ComplexReg::from_interleaved(Vector::from_view_chunk(
                        &src_view,
                        offset / per_register,
                    ))
                };
                let mut dst_view = simd.view_mut(bytemuck::cast_slice_mut::<Complex<T>, T>(dst));
                for k in (0..vector_end).step_by(per_register) {
                    let x0 = load(src_base + k);
                    let x1 = load(src_base + half_groups + k);
                    let x2 = load(src_base + groups + k) * w1v;
                    let x3 = load(src_base + groups + half_groups + k) * w1v;
                    let a0 = x0 + x2;
                    let a1 = x1 + x3;
                    let b0 = x0 - x2;
                    let b1 = x1 - x3;
                    let c0 = a1 * w2v;
                    let c1 = b1 * w3v;
                    let mut store = |offset: usize, value: ComplexReg<T, A>| {
                        value
                            .into_interleaved()
                            .store_to_view_chunk(&mut dst_view, offset / per_register);
                    };
                    store(dst_base + k, a0 + c0);
                    store(dst_base + half_n + k, a0 - c0);
                    store(dst_base + quarter_n + k, b0 + c1);
                    store(dst_base + half_n + quarter_n + k, b0 - c1);
                }
            }

            for k in vector_end..half_groups {
                let x0 = src[src_base + k];
                let x1 = src[src_base + half_groups + k];
                let x2 = src[src_base + groups + k] * w1;
                let x3 = src[src_base + groups + half_groups + k] * w1;
                let a0 = x0 + x2;
                let a1 = x1 + x3;
                let b0 = x0 - x2;
                let b1 = x1 - x3;
                let c0 = a1 * w2;
                let c1 = b1 * w3;
                dst[dst_base + k] = a0 + c0;
                dst[dst_base + half_n + k] = a0 - c0;
                dst[dst_base + quarter_n + k] = b0 + c1;
                dst[dst_base + half_n + quarter_n + k] = b0 - c1;
            }
        }
    }
}

impl<T> LaneKernel<T> for PairStageRadixOne<'_, T>
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
            second_twiddles,
        } = self;
        let n = src.len();
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let vector_end = quarter_n & !(per_register - 1);
        let w3 = second_twiddles[1];

        if vector_end > 0 {
            // As in `PairStage`: `quarter_n >= per_register`, both powers of
            // two, so `quarter_n`, `half_n`, and each `k` are register indices.
            let w3v = ComplexReg::<T, A>::splat(w3);
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
                let x1 = load(quarter_n + k);
                let x2 = load(half_n + k);
                let x3 = load(half_n + quarter_n + k);
                let a0 = x0 + x2;
                let a1 = x1 + x3;
                let b0 = x0 - x2;
                let b1 = x1 - x3;
                let c1 = b1 * w3v;
                let mut store = |offset: usize, value: ComplexReg<T, A>| {
                    value
                        .into_interleaved()
                        .store_to_view_chunk(&mut dst_view, offset / per_register);
                };
                store(k, a0 + a1);
                store(half_n + k, a0 - a1);
                store(quarter_n + k, b0 + c1);
                store(half_n + quarter_n + k, b0 - c1);
            }
        }

        for k in vector_end..quarter_n {
            let x0 = src[k];
            let x1 = src[quarter_n + k];
            let x2 = src[half_n + k];
            let x3 = src[half_n + quarter_n + k];
            let a0 = x0 + x2;
            let a1 = x1 + x3;
            let b0 = x0 - x2;
            let b1 = x1 - x3;
            let c1 = b1 * w3;
            dst[k] = a0 + a1;
            dst[half_n + k] = a0 - a1;
            dst[quarter_n + k] = b0 + c1;
            dst[half_n + quarter_n + k] = b0 - c1;
        }
    }
}

/// Runs the general pair stage (`radix >= 2`) at `LANES` scalar lanes per
/// register on the hardware backend of that width.
///
/// Returns `false`, having touched nothing, when this host has no hardware
/// backend at `LANES`; the caller then takes its scalar route.
#[inline]
pub(crate) fn stage_pair_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    radix: usize,
    first_twiddles: &[Complex<T>],
    second_twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(PairStage {
        src,
        dst,
        radix,
        first_twiddles,
        second_twiddles,
    })
    .is_some()
}

/// [`stage_pair_lanes`] for `radix == 1`, where only `second_twiddles[1]`
/// enters the recurrence.
#[inline]
pub(crate) fn stage_pair_radix_one_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    second_twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(PairStageRadixOne {
        src,
        dst,
        second_twiddles,
    })
    .is_some()
}

/// The pair stage at `groups == 2`: every Stockham digit `j` owns one `k`, so
/// its four inputs are the adjacent samples `src[4j..4j + 4]` and its four
/// outputs the strided `dst[j + {0, n/4, n/2, 3n/4}]`.
struct PairStageGroupsTwo<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    radix: usize,
    first_twiddles: &'a [Complex<T>],
    second_twiddles: &'a [Complex<T>],
}

impl<T> LaneKernel<T> for PairStageGroupsTwo<'_, T>
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
        } = self;
        let n = src.len();
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let vector_end = radix & !(per_register - 1);

        if vector_end > 0 {
            // A register of `per_register` consecutive `j` reads the
            // `4 · per_register` consecutive inputs `src[4j..]` as four
            // registers and splits them into the stride-4 subsequences
            // `x0..x3`; its twiddles and outputs are contiguous in `j`, and
            // `quarter_n = radix >= per_register` keeps every store aligned.
            let src_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(src));
            let first_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(first_twiddles));
            let second_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(second_twiddles));
            let mut dst_view = simd.view_mut(bytemuck::cast_slice_mut::<Complex<T>, T>(dst));
            for j in (0..vector_end).step_by(per_register) {
                let base = (4 * j) / per_register;
                let (x0, x1, x2, x3) = Vector::from_view_chunk(&src_view, base)
                    .deinterleave_pairs4(
                        Vector::from_view_chunk(&src_view, base + 1),
                        Vector::from_view_chunk(&src_view, base + 2),
                        Vector::from_view_chunk(&src_view, base + 3),
                    );
                let chunk = j / per_register;
                let w1 = ComplexReg::from_interleaved(Vector::from_view_chunk(&first_view, chunk));
                let w2 = ComplexReg::from_interleaved(Vector::from_view_chunk(&second_view, chunk));
                let w3 = ComplexReg::from_interleaved(Vector::from_view_chunk(
                    &second_view,
                    (j + radix) / per_register,
                ));
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
                store(j, a0 + c0);
                store(j + half_n, a0 - c0);
                store(j + quarter_n, b0 + c1);
                store(j + half_n + quarter_n, b0 - c1);
            }
        }

        for j in vector_end..radix {
            let x0 = src[4 * j];
            let x1 = src[4 * j + 1];
            let x2 = src[4 * j + 2] * first_twiddles[j];
            let x3 = src[4 * j + 3] * first_twiddles[j];
            let a0 = x0 + x2;
            let a1 = x1 + x3;
            let b0 = x0 - x2;
            let b1 = x1 - x3;
            let c0 = a1 * second_twiddles[j];
            let c1 = b1 * second_twiddles[j + radix];
            dst[j] = a0 + c0;
            dst[j + half_n] = a0 - c0;
            dst[j + quarter_n] = b0 + c1;
            dst[j + half_n + quarter_n] = b0 - c1;
        }
    }
}

/// [`stage_pair_lanes`] for `groups == 2` (`src.len() == 4 * radix`), where
/// each digit's inputs are adjacent and the general kernel's per-group loop
/// would run entirely in its scalar tail.
#[inline]
pub(crate) fn stage_pair_groups_two_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    radix: usize,
    first_twiddles: &[Complex<T>],
    second_twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(PairStageGroupsTwo {
        src,
        dst,
        radix,
        first_twiddles,
        second_twiddles,
    })
    .is_some()
}
