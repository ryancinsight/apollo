//! The flat radix-4 Stockham pass.
//!
//! Butterfly on the twiddled arms `a0..a3`: `t0 = a0 + a2`, `t1 = a0 - a2`,
//! `t2 = a1 + a3`, `t3 = a1 - a3`, then `b0 = t0 + t2`, `b1 = t1 ∓ i t3`,
//! `b2 = t0 - t2`, `b3 = t1 ± i t3` (the upper sign forward). The quarter
//! turn is a lane swap folded into the add: `t1 ∓ i t3` is `t1` with the
//! swapped `t3` subtracted on the even lanes and added on the odd ones
//! (`fmaddsub` by one), the mirror on the inverse.

use super::{apply_pointwise, cmul, load, store};
use crate::application::execution::kernel::components::winograd::WinogradScalar;
use eunomia::Complex;
use hermes_simd::{
    ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector,
};

/// One radix-4 pass: `g_count` groups of `prev_len` butterflies over four
/// source rows `g_count * prev_len` complexes apart, each group's four
/// output arms `prev_len` apart inside its `stage_chunk` block.
pub(in super::super) struct FlatPassR4<'a, T, const INVERSE: bool> {
    pub(in super::super) src: &'a [Complex<T>],
    pub(in super::super) dst: &'a mut [Complex<T>],
    pub(in super::super) prev_len: usize,
    pub(in super::super) g_count: usize,
    pub(in super::super) stage_chunk: usize,
    pub(in super::super) tw: &'a [Complex<T>],
    pub(in super::super) pointwise: Option<&'a [Complex<T>]>,
}

/// The four-point butterfly on one register of complexes per arm.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[inline(always)]
fn dft4<T, A, const INVERSE: bool>(
    one: Vector<T, A>,
    a0: Vector<T, A>,
    a1: Vector<T, A>,
    a2: Vector<T, A>,
    a3: Vector<T, A>,
) -> [Vector<T, A>; 4]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let t0 = a0 + a2;
    let t1 = a0 - a2;
    let t2 = a1 + a3;
    let t3 = (a1 - a3).swap_adjacent();
    // `t1 - i t3` reads `(t1r + t3i, t1i - t3r)`: the swapped `t3` added on
    // the even lanes and subtracted on the odd; `t1 + i t3` the mirror.
    let (b1, b3) = if INVERSE {
        (t1.fmaddsub(one, t3), t1.fmsubadd(one, t3))
    } else {
        (t1.fmsubadd(one, t3), t1.fmaddsub(one, t3))
    };
    [t0 + t2, b1, t0 - t2, b3]
}

/// The scalar four-point butterfly on twiddled arms.
#[inline]
fn dft4_scalar<T: WinogradScalar, const INVERSE: bool>(
    a0: Complex<T>,
    a1: Complex<T>,
    a2: Complex<T>,
    a3: Complex<T>,
) -> [Complex<T>; 4] {
    let t0 = a0 + a2;
    let t1 = a0 - a2;
    let t2 = a1 + a3;
    let t3 = a1 - a3;
    let it3 = if INVERSE {
        Complex::new(-t3.im, t3.re)
    } else {
        Complex::new(t3.im, -t3.re)
    };
    [t0 + t2, t1 + it3, t0 - t2, t1 - it3]
}

#[inline]
fn cmul_scalar<T: WinogradScalar>(a: Complex<T>, w: Complex<T>) -> Complex<T> {
    Complex::new(a.re * w.re - a.im * w.im, a.re * w.im + a.im * w.re)
}

impl<T, const INVERSE: bool> LaneKernel<T> for FlatPassR4<'_, T, INVERSE>
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
    #[expect(
        clippy::too_many_lines,
        reason = "the first stage, the twiddled stages and their tails are one pass; splitting them would leave the target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        let per = <A as SimdStorage<T>>::LANE_COUNT / 2;
        // The first stage reorders a group's four arms through a square
        // complex transpose, which the widths one, two and four have; a
        // wider register (sixteen `f32` lanes) takes the scalar pass there
        // until its own reordering is measured.
        if per == 0 || (self.prev_len == 1 && !matches!(per, 1 | 2 | 4)) {
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
            src.len() >= 4 * stride
                && dst.len() >= g_count * stage_chunk
                && stage_chunk >= 4 * prev_len
                && tw.len() >= 3 * prev_len,
            "invariant: a radix-4 pass reads four rows of g_count * prev_len complexes and three twiddle rows, and writes g_count blocks of stage_chunk"
        );
        let one = simd.splat(T::from_f64(1.0));

        if prev_len == 1 {
            // No twiddle (`W^0`); a register holds `per` groups of one arm,
            // and a group's four arms are consecutive in `dst`: a square
            // complex transpose per `per` arms reorders them where the
            // width is four or two, and a staging tile otherwise.
            let mut g = 0;
            while g + per <= g_count {
                // SAFETY: `g + per <= g_count = stride`, so each arm's row
                // stays inside `src`, and `4 (g + per) <= dst.len()`.
                unsafe {
                    let b = dft4::<T, A, INVERSE>(
                        one,
                        load::<T, A>(src, g),
                        load::<T, A>(src, stride + g),
                        load::<T, A>(src, 2 * stride + g),
                        load::<T, A>(src, 3 * stride + g),
                    );
                    if per == 4 {
                        let mut tile = b.map(ComplexReg::from_interleaved);
                        ComplexReg::transpose_square(&mut tile);
                        for (k, row) in tile.into_iter().enumerate() {
                            store(row.into_interleaved(), dst, 4 * (g + k));
                        }
                    } else if per == 2 {
                        let mut low = [b[0], b[1]].map(ComplexReg::from_interleaved);
                        let mut high = [b[2], b[3]].map(ComplexReg::from_interleaved);
                        ComplexReg::transpose_square(&mut low);
                        ComplexReg::transpose_square(&mut high);
                        store(low[0].into_interleaved(), dst, 4 * g);
                        store(high[0].into_interleaved(), dst, 4 * g + 2);
                        store(low[1].into_interleaved(), dst, 4 * (g + 1));
                        store(high[1].into_interleaved(), dst, 4 * (g + 1) + 2);
                    } else {
                        for (k, arm) in b.into_iter().enumerate() {
                            store(arm, dst, 4 * g + k);
                        }
                    }
                }
                g += per;
            }
            while g < g_count {
                let b = dft4_scalar::<T, INVERSE>(
                    src[g],
                    src[stride + g],
                    src[2 * stride + g],
                    src[3 * stride + g],
                );
                dst[4 * g..4 * g + 4].copy_from_slice(&b);
                g += 1;
            }
        } else {
            for g in 0..g_count {
                let src_base = g * prev_len;
                let dst_base = g * stage_chunk;
                let mut j = 0;
                // Two registers per arm abreast while the row allows: two
                // independent butterfly chains in flight.
                while j + 2 * per <= prev_len {
                    // SAFETY: `src_base + j + 2 per <= stride`, so every arm's
                    // row and every twiddle row stay inside their slices, and
                    // `dst_base + j + 3 prev_len + 2 per <= g_count * stage_chunk`.
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
                        let a3 = cmul(
                            load::<T, A>(src, 3 * stride + at),
                            load::<T, A>(tw, 2 * prev_len + j),
                        );
                        let c0 = load::<T, A>(src, bt);
                        let c1 = cmul(load::<T, A>(src, stride + bt), load::<T, A>(tw, j + per));
                        let c2 = cmul(
                            load::<T, A>(src, 2 * stride + bt),
                            load::<T, A>(tw, prev_len + j + per),
                        );
                        let c3 = cmul(
                            load::<T, A>(src, 3 * stride + bt),
                            load::<T, A>(tw, 2 * prev_len + j + per),
                        );
                        let b = dft4::<T, A, INVERSE>(one, a0, a1, a2, a3);
                        let d = dft4::<T, A, INVERSE>(one, c0, c1, c2, c3);
                        for k in 0..4 {
                            store(b[k], dst, dst_base + j + k * prev_len);
                            store(d[k], dst, dst_base + j + per + k * prev_len);
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
                        let a3 = cmul(
                            load::<T, A>(src, 3 * stride + src_base + j),
                            load::<T, A>(tw, 2 * prev_len + j),
                        );
                        let b = dft4::<T, A, INVERSE>(one, a0, a1, a2, a3);
                        for (k, arm) in b.into_iter().enumerate() {
                            store(arm, dst, dst_base + j + k * prev_len);
                        }
                    }
                    j += per;
                }
                while j < prev_len {
                    let at = src_base + j;
                    let a0 = src[at];
                    let a1 = cmul_scalar(src[stride + at], tw[j]);
                    let a2 = cmul_scalar(src[2 * stride + at], tw[prev_len + j]);
                    let a3 = cmul_scalar(src[3 * stride + at], tw[2 * prev_len + j]);
                    let b = dft4_scalar::<T, INVERSE>(a0, a1, a2, a3);
                    for (k, arm) in b.into_iter().enumerate() {
                        dst[dst_base + j + k * prev_len] = arm;
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
