//! The flat radix-5 Stockham pass.
//!
//! Butterfly on the twiddled arms `a0..a4`, with the fifth roots' cosines
//! `c1 = cos(2π/5)`, `c2 = cos(4π/5)` and sines `s1 = sin(2π/5)`,
//! `s2 = sin(4π/5)` (negated forward):
//!
//! ```text
//!   t1 = a1 + a4,  t2 = a1 - a4,  t3 = a2 + a3,  t4 = a2 - a3
//!   m1 = c1 t1 + c2 t3,   m2 = c2 t1 + c1 t3
//!   q3 = s1 t2 + s2 t4,   q4 = s2 t2 - s1 t4
//!   b0 = a0 + t1 + t3
//!   b1 = a0 + m1 + i q3,  b4 = a0 + m1 - i q3
//!   b2 = a0 + m2 + i q4,  b3 = a0 + m2 - i q4
//! ```
//!
//! `i q` is the swapped `q` times `(-1, 1)`, and the swap distributes over
//! the sums, so `i q3` and `i q4` are fused chains on the swapped
//! differences with the sines carrying the sign pair.

use super::{
    apply_pointwise, cmul, duplicated_row, load, scatter_spill, store, store_arm_halves,
    store_arms, MAX_COMPLEXES_PER_REGISTER,
};
use crate::application::execution::kernel::components::winograd::WinogradScalar;
use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

const C1: f64 = 0.309_016_994_374_947_45;
const C2: f64 = -0.809_016_994_374_947_5;
const S1: f64 = 0.951_056_516_295_153_5;
const S2: f64 = 0.587_785_252_292_473_1;

/// One radix-5 pass: `g_count` groups of `prev_len` butterflies over five
/// source rows `g_count * prev_len` complexes apart, each group's five
/// output arms `prev_len` apart inside its `stage_chunk` block.
pub(in super::super) struct FlatPassR5<'a, T, const INVERSE: bool> {
    pub(in super::super) src: &'a [Complex<T>],
    pub(in super::super) dst: &'a mut [Complex<T>],
    pub(in super::super) prev_len: usize,
    pub(in super::super) g_count: usize,
    pub(in super::super) stage_chunk: usize,
    pub(in super::super) tw: &'a [Complex<T>],
    pub(in super::super) pointwise: Option<&'a [Complex<T>]>,
}

/// The register constants of the butterfly: the cosines, and the sines
/// signed by direction times the `(-1, 1)` of a quarter turn.
#[derive(Clone, Copy)]
struct Constants<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    c1: Vector<T, A>,
    c2: Vector<T, A>,
    s1_turn: Vector<T, A>,
    s2_turn: Vector<T, A>,
}

/// The five-point butterfly on one register of complexes per arm.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[inline(always)]
fn dft5<T, A>(
    k: Constants<T, A>,
    a0: Vector<T, A>,
    a1: Vector<T, A>,
    a2: Vector<T, A>,
    a3: Vector<T, A>,
    a4: Vector<T, A>,
) -> [Vector<T, A>; 5]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let t1 = a1 + a4;
    let t3 = a2 + a3;
    let t2s = (a1 - a4).swap_adjacent();
    let t4s = (a2 - a3).swap_adjacent();
    let m1 = t3.mul_add(k.c2, t1 * k.c1);
    let m2 = t3.mul_add(k.c1, t1 * k.c2);
    let iq3 = t4s.mul_add(k.s2_turn, t2s * k.s1_turn);
    let iq4 = t2s.mul_sub(k.s2_turn, t4s * k.s1_turn);
    let a1c = a0 + m1;
    let a2c = a0 + m2;
    [a0 + (t1 + t3), a1c + iq3, a2c + iq4, a2c - iq4, a1c - iq3]
}

/// The scalar five-point butterfly on twiddled arms.
#[inline]
fn dft5_scalar<T: WinogradScalar, const INVERSE: bool>(a: [Complex<T>; 5]) -> [Complex<T>; 5] {
    let c1 = T::from_f64(C1);
    let c2 = T::from_f64(C2);
    let (s1, s2) = if INVERSE {
        (T::from_f64(S1), T::from_f64(S2))
    } else {
        (T::from_f64(-S1), T::from_f64(-S2))
    };
    let t1 = a[1] + a[4];
    let t2 = a[1] - a[4];
    let t3 = a[2] + a[3];
    let t4 = a[2] - a[3];
    let m1 = Complex::new(t1.re * c1 + t3.re * c2, t1.im * c1 + t3.im * c2);
    let m2 = Complex::new(t1.re * c2 + t3.re * c1, t1.im * c2 + t3.im * c1);
    let q3 = Complex::new(t2.re * s1 + t4.re * s2, t2.im * s1 + t4.im * s2);
    let q4 = Complex::new(t2.re * s2 - t4.re * s1, t2.im * s2 - t4.im * s1);
    let iq3 = Complex::new(-q3.im, q3.re);
    let iq4 = Complex::new(-q4.im, q4.re);
    let a1c = a[0] + m1;
    let a2c = a[0] + m2;
    [a[0] + (t1 + t3), a1c + iq3, a2c + iq4, a2c - iq4, a1c - iq3]
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
        let a: [Complex<T>; 5] = core::array::from_fn(|arm| {
            let x = src[arm * stride + at];
            if arm == 0 {
                x
            } else {
                cmul_scalar(x, tw[(arm - 1) * prev_len + j])
            }
        });
        let b = dft5_scalar::<T, INVERSE>(a);
        for (arm, row) in b.into_iter().enumerate() {
            dst[dst_base + j + arm * prev_len] = row;
        }
    }
}

impl<T, const INVERSE: bool> LaneKernel<T> for FlatPassR5<'_, T, INVERSE>
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
            src.len() >= 5 * stride
                && dst.len() >= g_count * stage_chunk
                && stage_chunk >= 5 * prev_len
                && tw.len() >= 4 * prev_len,
            "invariant: a radix-5 pass reads five rows of g_count * prev_len complexes and four twiddle rows, and writes g_count blocks of stage_chunk"
        );
        // `i q` on the swapped `q` is `(-qi, qr)`: the sign pair `(-1, 1)`,
        // folded into the direction-signed sines.
        let (s1, s2) = if INVERSE { (S1, S2) } else { (-S1, -S2) };
        let k = Constants {
            c1: simd.splat(T::from_f64(C1)),
            c2: simd.splat(T::from_f64(C2)),
            s1_turn: Vector::<T, A>::splat_pair(T::from_f64(-s1), T::from_f64(s1)),
            s2_turn: Vector::<T, A>::splat_pair(T::from_f64(-s2), T::from_f64(s2)),
        };

        if prev_len == 1 {
            // No twiddle; a register holds `per` groups of one arm and a
            // group's five arms are consecutive in `dst`: the arm scatter
            // transposes them in registers, leaving the groups its run-over
            // needs to the scalar tail.
            let slack = scatter_spill::<T, A, 5>().div_ceil(5);
            let mut g = 0;
            while g + per + slack <= g_count {
                // SAFETY: `g + per <= g_count = stride`, so each arm's row
                // stays inside `src`, and `5 (g + per) + spill <= dst.len()`.
                unsafe {
                    let b = dft5(
                        k,
                        load::<T, A>(src, g),
                        load::<T, A>(src, stride + g),
                        load::<T, A>(src, 2 * stride + g),
                        load::<T, A>(src, 3 * stride + g),
                        load::<T, A>(src, 4 * stride + g),
                    );
                    store_arms::<T, A, 5>(b, dst, 5 * g);
                }
                g += per;
            }
            for (g, out) in dst[5 * g..5 * g_count]
                .chunks_exact_mut(5)
                .enumerate()
                .map(|(k, out)| (g + k, out))
            {
                let b =
                    dft5_scalar::<T, INVERSE>(core::array::from_fn(|arm| src[arm * stride + g]));
                out.copy_from_slice(&b);
            }
        } else if 2 * prev_len == per {
            // Rows are half a register: a register holds two groups' rows of
            // one arm, multiplied by the twiddle row twice over, and the two
            // groups' outputs are contiguous in `dst`.
            let w1 = duplicated_row::<T, A>(&tw[..prev_len]);
            let w2 = duplicated_row::<T, A>(&tw[prev_len..2 * prev_len]);
            let w3 = duplicated_row::<T, A>(&tw[2 * prev_len..3 * prev_len]);
            let w4 = duplicated_row::<T, A>(&tw[3 * prev_len..4 * prev_len]);
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
                    let a3 = cmul(load::<T, A>(src, 3 * stride + at), w3);
                    let a4 = cmul(load::<T, A>(src, 4 * stride + at), w4);
                    store_arm_halves::<T, A, 5>(dft5(k, a0, a1, a2, a3, a4), dst, g * stage_chunk);
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
                while j + per <= prev_len {
                    // SAFETY: `src_base + j + per <= stride`, so every arm's row
                    // and every twiddle row stay inside their slices, and
                    // `dst_base + j + 4 prev_len + per <= g_count * stage_chunk`.
                    // Straight-line arms: an array filled in a loop here
                    // round-tripped the stack on every iteration.
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
                        let a4 = cmul(
                            load::<T, A>(src, 4 * stride + at),
                            load::<T, A>(tw, 3 * prev_len + j),
                        );
                        let b = dft5(k, a0, a1, a2, a3, a4);
                        for arm in 0..5 {
                            store(b[arm], dst, dst_base + j + arm * prev_len);
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
