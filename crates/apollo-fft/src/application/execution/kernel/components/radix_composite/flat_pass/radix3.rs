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
//! `∓ i · diff · s` is the swapped difference times the signed sine pair
//! `(s, -s)` forward, `(-s, s)` inverse.

use super::{
    apply_pointwise, cmul, duplicated_row, load, load_prefix, prefix_mask, scatter_spill, store,
    store_arm_halves, store_arms, store_prefix, MAX_COMPLEXES_PER_REGISTER,
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
    half_negative: Vector<T, A>,
    /// `(s, -s)` forward, `(-s, s)` inverse: the sine with the quarter
    /// turn's sign pattern.
    sine_turn: Vector<T, A>,
}

/// The three-point butterfly on one register of complexes per arm.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[inline(always)]
fn dft3<T, A>(
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
    let m1 = (a1 - a2).swap_adjacent() * k.sine_turn;
    [a0 + sum, m0 + m1, m0 - m1]
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
    #[expect(
        clippy::too_many_lines,
        reason = "the first stage, the half-register stage, the twiddled stages and their tails are one pass; splitting them would leave the target-feature frame"
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
        let (lo, hi) = if INVERSE {
            (-SIN_THIRD, SIN_THIRD)
        } else {
            (SIN_THIRD, -SIN_THIRD)
        };
        let k = Constants {
            half_negative: simd.splat(T::from_f64(-0.5)),
            sine_turn: Vector::<T, A>::splat_pair(T::from_f64(lo), T::from_f64(hi)),
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
                    let b = dft3(
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
        } else if 2 * prev_len == per {
            // Rows are half a register: a register holds two groups' rows of
            // one arm, multiplied by the twiddle row twice over, and the two
            // groups' outputs are contiguous in `dst`.
            let w1 = duplicated_row::<T, A>(&tw[..prev_len]);
            let w2 = duplicated_row::<T, A>(&tw[prev_len..2 * prev_len]);
            let mut g = 0;
            while g + 2 <= g_count {
                // SAFETY: `(g + 2) prev_len <= stride`, so each arm's row
                // stays inside `src`, and the two groups' outputs end at
                // `(g + 2) stage_chunk <= dst.len()`.
                unsafe {
                    let at = g * prev_len;
                    let a0 = load::<T, A>(src, at);
                    let a1 = cmul(load::<T, A>(src, stride + at), w1);
                    let a2 = cmul(load::<T, A>(src, 2 * stride + at), w2);
                    store_arm_halves::<T, A, 3>(dft3(k, a0, a1, a2), dst, g * stage_chunk);
                }
                g += 2;
            }
            if g < g_count {
                // An odd last group: its half-register rows through masked
                // loads and stores.
                let m = prefix_mask::<T, A>(prev_len);
                // SAFETY: `(g + 1) prev_len <= stride`, so the masked rows
                // stay inside their slices, and the group's outputs end at
                // `(g + 1) stage_chunk <= dst.len()`.
                unsafe {
                    let at = g * prev_len;
                    let a0 = load_prefix::<T, A>(src, at, prev_len, m);
                    let a1 = cmul(
                        load_prefix::<T, A>(src, stride + at, prev_len, m),
                        load_prefix::<T, A>(tw, 0, prev_len, m),
                    );
                    let a2 = cmul(
                        load_prefix::<T, A>(src, 2 * stride + at, prev_len, m),
                        load_prefix::<T, A>(tw, prev_len, prev_len, m),
                    );
                    let b = dft3(k, a0, a1, a2);
                    for arm in 0..3 {
                        store_prefix(b[arm], dst, g * stage_chunk + arm * prev_len, prev_len, m);
                    }
                }
            }
        } else {
            // The ragged tail's width and mask are the stage's, not the group's.
            let rem = prev_len % per;
            let m = prefix_mask::<T, A>(rem.max(1));
            for g in 0..g_count {
                let src_base = g * prev_len;
                let dst_base = g * stage_chunk;
                // The ragged tail runs first: its run-over into the next arm's
                // leading columns is overwritten by that arm's whole-register
                // stores, and the last arm's by the next group's.
                if rem != 0 {
                    let c = rem;
                    let j = prev_len - rem;
                    if g + 1 < g_count && per - c <= prev_len {
                        // The run-over tail: whole-register loads and stores
                        // that run `per - c` samples into the next row, which
                        // the next store (arms ascending, then the next group)
                        // overwrites; twiddle rows run into the next row too, the
                        // last one masked since it has no successor.
                        // SAFETY: with `per - c <= prev_len` every run-over stays
                        // inside the neighbouring row of the same arm (source),
                        // the next arm (output) or the next twiddle row, and
                        // `g + 1 < g_count` puts a whole group after this one in
                        // both slices.
                        unsafe {
                            let at = src_base + j;
                            let a0 = load::<T, A>(src, at);
                            let a1 = cmul(load::<T, A>(src, stride + at), load::<T, A>(tw, j));
                            let a2 = cmul(
                                load::<T, A>(src, 2 * stride + at),
                                load_prefix::<T, A>(tw, prev_len + j, c, m),
                            );
                            let b = dft3(k, a0, a1, a2);
                            for arm in 0..3 {
                                store(b[arm], dst, dst_base + j + arm * prev_len);
                            }
                        }
                    } else {
                        // SAFETY: `src_base + prev_len <= stride`, so the `c`
                        // masked complexes of every row and twiddle row stay
                        // inside their slices, and the outputs end at
                        // `dst_base + 3 prev_len <= g_count * stage_chunk`.
                        unsafe {
                            let at = src_base + j;
                            let a0 = load_prefix::<T, A>(src, at, c, m);
                            let a1 = cmul(
                                load_prefix::<T, A>(src, stride + at, c, m),
                                load_prefix::<T, A>(tw, j, c, m),
                            );
                            let a2 = cmul(
                                load_prefix::<T, A>(src, 2 * stride + at, c, m),
                                load_prefix::<T, A>(tw, prev_len + j, c, m),
                            );
                            let b = dft3(k, a0, a1, a2);
                            for arm in 0..3 {
                                store_prefix(b[arm], dst, dst_base + j + arm * prev_len, c, m);
                            }
                        }
                    }
                }
                let mut j = 0;
                // Two registers per arm abreast where a register is two
                // complexes: four complexes in flight either way.
                while per < 4 && j + 2 * per <= prev_len {
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
                        let b = dft3(k, a0, a1, a2);
                        let d = dft3(k, c0, c1, c2);
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
                        let at = src_base + j;
                        let a0 = load::<T, A>(src, at);
                        let a1 = cmul(load::<T, A>(src, stride + at), load::<T, A>(tw, j));
                        let a2 = cmul(
                            load::<T, A>(src, 2 * stride + at),
                            load::<T, A>(tw, prev_len + j),
                        );
                        let b = dft3(k, a0, a1, a2);
                        for arm in 0..3 {
                            store(b[arm], dst, dst_base + j + arm * prev_len);
                        }
                    }
                    j += per;
                }
            }
        }

        if let Some(factors) = pointwise {
            apply_pointwise::<T, A>(dst, factors);
        }
        true
    }
}
