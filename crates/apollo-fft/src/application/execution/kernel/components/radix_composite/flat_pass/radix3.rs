//! The flat radix-3 Stockham pass.
//!
//! Butterfly on the twiddled arms `a0..a2`, with `s = √3 / 2`:
//!
//! ```text
//!   sum = a1 + a2,  diff = a1 - a2
//!   b0  = a0 + sum
//!   m0  = a0 - sum / 2                (one fused multiply-add)
//!   m1  = ∓ i · diff · s              (the upper sign forward)
//!   b1  = m0 + m1
//!   b2  = m0 - m1
//! ```
//!
//! `∓ i · diff · s` is the swapped difference scaled by `s`, added on the
//! even lanes and subtracted on the odd ones (`fmsubadd` by one) for `b1`
//! forward, the mirror for `b2` and on the inverse.

use super::{
    apply_pointwise, cmul, load, scatter_spill, store, store_arms, MAX_COMPLEXES_PER_REGISTER,
};
use crate::application::execution::kernel::components::winograd::WinogradScalar;
use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

/// `√3 / 2`, the sine of the third root of unity.
const SIN_THIRD: f64 = 0.866_025_403_784_438_6;

/// One radix-3 pass: `g_count` groups of `prev_len` butterflies over three
/// source rows `g_count * prev_len` complexes apart, each group's three
/// output arms `prev_len` apart inside its `stage_chunk` block.
pub(in super::super) struct FlatPassR3<'a, T, const INVERSE: bool> {
    pub(in super::super) src: &'a [Complex<T>],
    pub(in super::super) dst: &'a mut [Complex<T>],
    pub(in super::super) prev_len: usize,
    pub(in super::super) g_count: usize,
    pub(in super::super) stage_chunk: usize,
    pub(in super::super) tw: &'a [Complex<T>],
    pub(in super::super) pointwise: Option<&'a [Complex<T>]>,
}

/// The register constants of the butterfly.
#[derive(Clone, Copy)]
struct Constants<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    one: Vector<T, A>,
    half_negative: Vector<T, A>,
    sin_third: Vector<T, A>,
}

/// The three-point butterfly on one register of complexes per arm.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[inline(always)]
fn dft3<T, A, const INVERSE: bool>(
    k: Constants<T, A>,
    a0: Vector<T, A>,
    a1: Vector<T, A>,
    a2: Vector<T, A>,
) -> [Vector<T, A>; 3]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let sum = a1 + a2;
    let m0 = sum.mul_add(k.half_negative, a0);
    let r = (a1 - a2).swap_adjacent() * k.sin_third;
    let (b1, b2) = if INVERSE {
        (m0.fmaddsub(k.one, r), m0.fmsubadd(k.one, r))
    } else {
        (m0.fmsubadd(k.one, r), m0.fmaddsub(k.one, r))
    };
    [a0 + sum, b1, b2]
}

/// The scalar three-point butterfly on twiddled arms.
#[inline]
fn dft3_scalar<T: WinogradScalar, const INVERSE: bool>(
    a0: Complex<T>,
    a1: Complex<T>,
    a2: Complex<T>,
) -> [Complex<T>; 3] {
    let s = T::from_f64(SIN_THIRD);
    let wr = T::from_f64(-0.5);
    let sum = a1 + a2;
    let diff = a1 - a2;
    let m0 = Complex::new(a0.re + sum.re * wr, a0.im + sum.im * wr);
    let m1 = if INVERSE {
        Complex::new(-diff.im * s, diff.re * s)
    } else {
        Complex::new(diff.im * s, -diff.re * s)
    };
    [a0 + sum, m0 + m1, m0 - m1]
}

#[inline]
fn cmul_scalar<T: WinogradScalar>(a: Complex<T>, w: Complex<T>) -> Complex<T> {
    Complex::new(a.re * w.re - a.im * w.im, a.re * w.im + a.im * w.re)
}

impl<T, const INVERSE: bool> LaneKernel<T> for FlatPassR3<'_, T, INVERSE>
where
    T: LaneScalar + WinogradScalar,
{
    /// Whether the dispatched width ran the pass; the scalar backend
    /// declines, and the scalar pass runs instead.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        let per = <A as SimdStorage<T>>::LANE_COUNT / 2;
        if per == 0 || per > MAX_COMPLEXES_PER_REGISTER {
            return false;
        }
        let Self {
            src,
            dst,
            prev_len,
            g_count,
            stage_chunk,
            tw,
            pointwise,
        } = self;
        let stride = g_count * prev_len;
        assert!(
            src.len() >= 3 * stride
                && dst.len() >= g_count * stage_chunk
                && stage_chunk >= 3 * prev_len
                && tw.len() >= 2 * prev_len,
            "invariant: a radix-3 pass reads three rows of g_count * prev_len complexes and two twiddle rows, and writes g_count blocks of stage_chunk"
        );
        let k = Constants {
            one: simd.splat(T::from_f64(1.0)),
            half_negative: simd.splat(T::from_f64(-0.5)),
            sin_third: simd.splat(T::from_f64(SIN_THIRD)),
        };

        if prev_len == 1 {
            // No twiddle; a register holds `per` groups of one arm and a
            // group's three arms are consecutive in `dst`: the arm scatter
            // transposes them in registers, leaving the groups its run-over
            // needs to the scalar tail.
            let slack = scatter_spill::<T, A, 3>().div_ceil(3);
            let mut g = 0;
            while g + per + slack <= g_count {
                // SAFETY: `g + per <= g_count = stride`, so each arm's row
                // stays inside `src`, and `3 (g + per) + spill <= dst.len()`.
                unsafe {
                    let b = dft3::<T, A, INVERSE>(
                        k,
                        load::<T, A>(src, g),
                        load::<T, A>(src, stride + g),
                        load::<T, A>(src, 2 * stride + g),
                    );
                    store_arms::<T, A, 3>(b, dst, 3 * g);
                }
                g += per;
            }
            for (g, out) in dst[3 * g..3 * g_count]
                .chunks_exact_mut(3)
                .enumerate()
                .map(|(k, out)| (g + k, out))
            {
                let b = dft3_scalar::<T, INVERSE>(src[g], src[stride + g], src[2 * stride + g]);
                out.copy_from_slice(&b);
            }
        } else {
            for g in 0..g_count {
                let src_base = g * prev_len;
                let dst_base = g * stage_chunk;
                let mut j = 0;
                while j + 2 * per <= prev_len {
                    // SAFETY: `src_base + j + 2 per <= stride`, so every arm's
                    // row and both twiddle rows stay inside their slices, and
                    // `dst_base + j + 2 prev_len + 2 per <= g_count * stage_chunk`.
                    // Straight-line loads: a closure here compiled out of the
                    // target-feature frame with every vector op a call.
                    unsafe {
                        let (at, bt) = (src_base + j, src_base + j + per);
                        let a0 = load::<T, A>(src, at);
                        let a1 = cmul(load::<T, A>(src, stride + at), load::<T, A>(tw, j));
                        let a2 = cmul(
                            load::<T, A>(src, 2 * stride + at),
                            load::<T, A>(tw, prev_len + j),
                        );
                        let c0 = load::<T, A>(src, bt);
                        let c1 = cmul(load::<T, A>(src, stride + bt), load::<T, A>(tw, j + per));
                        let c2 = cmul(
                            load::<T, A>(src, 2 * stride + bt),
                            load::<T, A>(tw, prev_len + j + per),
                        );
                        let b = dft3::<T, A, INVERSE>(k, a0, a1, a2);
                        let d = dft3::<T, A, INVERSE>(k, c0, c1, c2);
                        for arm in 0..3 {
                            store(b[arm], dst, dst_base + j + arm * prev_len);
                            store(d[arm], dst, dst_base + j + per + arm * prev_len);
                        }
                    }
                    j += 2 * per;
                }
                while j + per <= prev_len {
                    // SAFETY: as above with one register per arm.
                    unsafe {
                        let a0 = load::<T, A>(src, src_base + j);
                        let a1 = cmul(
                            load::<T, A>(src, stride + src_base + j),
                            load::<T, A>(tw, j),
                        );
                        let a2 = cmul(
                            load::<T, A>(src, 2 * stride + src_base + j),
                            load::<T, A>(tw, prev_len + j),
                        );
                        let b = dft3::<T, A, INVERSE>(k, a0, a1, a2);
                        for (arm, row) in b.into_iter().enumerate() {
                            store(row, dst, dst_base + j + arm * prev_len);
                        }
                    }
                    j += per;
                }
                while j < prev_len {
                    let at = src_base + j;
                    let a0 = src[at];
                    let a1 = cmul_scalar(src[stride + at], tw[j]);
                    let a2 = cmul_scalar(src[2 * stride + at], tw[prev_len + j]);
                    let b = dft3_scalar::<T, INVERSE>(a0, a1, a2);
                    for (arm, row) in b.into_iter().enumerate() {
                        dst[dst_base + j + arm * prev_len] = row;
                    }
                    j += 1;
                }
            }
        }

        if let Some(factors) = pointwise {
            apply_pointwise::<T, A>(dst, factors);
        }
        true
    }
}
