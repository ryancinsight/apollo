//! The flat radix-4 Stockham pass.
//!
//! Butterfly on the twiddled arms `a0..a3`: `t0 = a0 + a2`, `t1 = a0 - a2`,
//! `t2 = a1 + a3`, `t3 = a1 - a3`, then `b0 = t0 + t2`, `b1 = t1 ∓ i t3`,
//! `b2 = t0 - t2`, `b3 = t1 ± i t3` (the upper sign forward). The quarter
//! turn is a lane swap times the sign pair `(1, -1)` forward, `(-1, 1)`
//! inverse: `∓ i t3 = (t3i, -t3r)` forward.

use super::{
    apply_pointwise, cmul, duplicated_row, load, scatter_spill, store, store_arm_halves,
    store_arms, MAX_COMPLEXES_PER_REGISTER,
};
use crate::application::execution::kernel::components::winograd::WinogradScalar;
use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

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

/// The four-point butterfly on one register of complexes per arm; `turn`
/// is the quarter turn's sign pair.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[inline(always)]
fn dft4<T, A>(
    turn: Vector<T, A>,
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
    let r = (a1 - a3).swap_adjacent() * turn;
    [t0 + t2, t1 + r, t0 - t2, t1 - r]
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

/// Group `g`'s butterflies from column `j` on, in scalar arithmetic: the
/// tail of a row past the last whole register.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the pass geometry is the argument list"
)]
fn scalar_columns<T: WinogradScalar, const INVERSE: bool>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    tw: &[Complex<T>],
    stride: usize,
    prev_len: usize,
    stage_chunk: usize,
    g: usize,
    j: usize,
) {
    let src_base = g * prev_len;
    let dst_base = g * stage_chunk;
    for j in j..prev_len {
        let at = src_base + j;
        let a0 = src[at];
        let a1 = cmul_scalar(src[stride + at], tw[j]);
        let a2 = cmul_scalar(src[2 * stride + at], tw[prev_len + j]);
        let a3 = cmul_scalar(src[3 * stride + at], tw[2 * prev_len + j]);
        let b = dft4_scalar::<T, INVERSE>(a0, a1, a2, a3);
        for (k, arm) in b.into_iter().enumerate() {
            dst[dst_base + j + k * prev_len] = arm;
        }
    }
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
        reason = "the first stage, the half-register stage, the twiddled stages and their tails are one pass; splitting them would leave the target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _capability: Simd<T, A>) -> bool {
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
            src.len() >= 4 * stride
                && dst.len() >= g_count * stage_chunk
                && stage_chunk >= 4 * prev_len
                && tw.len() >= 3 * prev_len,
            "invariant: a radix-4 pass reads four rows of g_count * prev_len complexes and three twiddle rows, and writes g_count blocks of stage_chunk"
        );
        let (lo, hi) = if INVERSE { (-1.0, 1.0) } else { (1.0, -1.0) };
        let turn = Vector::<T, A>::splat_pair(T::from_f64(lo), T::from_f64(hi));

        if prev_len == 1 {
            // No twiddle (`W^0`); a register holds `per` groups of one arm,
            // and a group's four arms are consecutive in `dst`: the arm
            // scatter transposes them in registers, leaving the groups its
            // run-over needs to the scalar tail.
            let slack = scatter_spill::<T, A, 4>().div_ceil(4);
            let mut g = 0;
            while g + per + slack <= g_count {
                // SAFETY: `g + per <= g_count = stride`, so each arm's row
                // stays inside `src`, and `4 (g + per) + spill <= dst.len()`.
                unsafe {
                    let b = dft4(
                        turn,
                        load::<T, A>(src, g),
                        load::<T, A>(src, stride + g),
                        load::<T, A>(src, 2 * stride + g),
                        load::<T, A>(src, 3 * stride + g),
                    );
                    store_arms::<T, A, 4>(b, dst, 4 * g);
                }
                g += per;
            }
            for (g, out) in dst[4 * g..4 * g_count]
                .chunks_exact_mut(4)
                .enumerate()
                .map(|(k, out)| (g + k, out))
            {
                let b = dft4_scalar::<T, INVERSE>(
                    src[g],
                    src[stride + g],
                    src[2 * stride + g],
                    src[3 * stride + g],
                );
                out.copy_from_slice(&b);
            }
        } else if 2 * prev_len == per {
            // Rows are half a register: a register holds two groups' rows of
            // one arm, multiplied by the twiddle row twice over, and the two
            // groups' outputs are contiguous in `dst`.
            let w1 = duplicated_row::<T, A>(&tw[..prev_len]);
            let w2 = duplicated_row::<T, A>(&tw[prev_len..2 * prev_len]);
            let w3 = duplicated_row::<T, A>(&tw[2 * prev_len..3 * prev_len]);
            let mut g = 0;
            while g + 2 <= g_count {
                // SAFETY: `(g + 2) prev_len <= stride`, so each arm's row
                // stays inside `src`, and the two groups' `2 stage_chunk`
                // outputs end at `(g + 2) stage_chunk <= dst.len()`.
                unsafe {
                    let at = g * prev_len;
                    let a0 = load::<T, A>(src, at);
                    let a1 = cmul(load::<T, A>(src, stride + at), w1);
                    let a2 = cmul(load::<T, A>(src, 2 * stride + at), w2);
                    let a3 = cmul(load::<T, A>(src, 3 * stride + at), w3);
                    store_arm_halves::<T, A, 4>(dft4(turn, a0, a1, a2, a3), dst, g * stage_chunk);
                }
                g += 2;
            }
            for g in g..g_count {
                scalar_columns::<T, INVERSE>(src, dst, tw, stride, prev_len, stage_chunk, g, 0);
            }
        } else {
            for g in 0..g_count {
                let src_base = g * prev_len;
                let dst_base = g * stage_chunk;
                let mut j = 0;
                // Two registers per arm abreast where a register is two
                // complexes: four complexes in flight either way, without
                // the address pressure of eight rows and six twiddle rows
                // at a wider register.
                while per < 4 && j + 2 * per <= prev_len {
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
                        let b = dft4(turn, a0, a1, a2, a3);
                        let d = dft4(turn, c0, c1, c2, c3);
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
                        let at = src_base + j;
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
                        let b = dft4(turn, a0, a1, a2, a3);
                        for k in 0..4 {
                            store(b[k], dst, dst_base + j + k * prev_len);
                        }
                    }
                    j += per;
                }
                scalar_columns::<T, INVERSE>(src, dst, tw, stride, prev_len, stage_chunk, g, j);
            }
        }

        if let Some(factors) = pointwise {
            apply_pointwise::<T, A>(dst, factors);
        }
        true
    }
}
