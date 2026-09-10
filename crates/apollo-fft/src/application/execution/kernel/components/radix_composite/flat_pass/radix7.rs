//! The flat radix-7 Stockham pass.
//!
//! Butterfly on the twiddled arms `a0..a6`, with the seventh roots'
//! cosines `c1..c3` and sines `s1..s3`:
//!
//! ```text
//!   xr_k = a_k + a_(7-k),  xi_k = ∓ i (a_k - a_(7-k))      (k = 1, 2, 3; upper sign forward)
//!   re1 = a0 + c1 xr1 + c2 xr2 + c3 xr3      d1 =  s1 xi1 + s2 xi2 + s3 xi3
//!   re2 = a0 + c2 xr1 + c3 xr2 + c1 xr3      d2 =  s2 xi1 - s3 xi2 - s1 xi3
//!   re3 = a0 + c3 xr1 + c1 xr2 + c2 xr3      d3 =  s3 xi1 - s1 xi2 + s2 xi3
//!   b0 = a0 + xr1 + xr2 + xr3
//!   b_k = re_k + d_k,  b_(7-k) = re_k - d_k
//! ```
//!
//! `∓ i x` is the swapped `x` times the sign pair `(1, -1)` forward,
//! `(-1, 1)` inverse, folded into the sines: the `d` chains run on the
//! swapped differences with `s_k` carrying the pair.

use super::{
    apply_pointwise, cmul, duplicated_row, load, scatter_spill, store, store_arm_halves,
    store_arms, MAX_COMPLEXES_PER_REGISTER,
};
use crate::application::execution::kernel::components::winograd::WinogradScalar;
use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

const C1: f64 = 0.623_489_801_858_733_6;
const C2: f64 = -0.222_520_933_956_314_4;
const C3: f64 = -0.900_968_867_902_419_1;
const S1: f64 = 0.781_831_482_468_029_8;
const S2: f64 = 0.974_927_912_181_823_6;
const S3: f64 = 0.433_883_739_117_558_2;

/// One radix-7 pass: `g_count` groups of `prev_len` butterflies over seven
/// source rows `g_count * prev_len` complexes apart, each group's seven
/// output arms `prev_len` apart inside its `stage_chunk` block.
pub(in super::super) struct FlatPassR7<'a, T, const INVERSE: bool> {
    pub(in super::super) src: &'a [Complex<T>],
    pub(in super::super) dst: &'a mut [Complex<T>],
    pub(in super::super) prev_len: usize,
    pub(in super::super) g_count: usize,
    pub(in super::super) stage_chunk: usize,
    pub(in super::super) tw: &'a [Complex<T>],
    pub(in super::super) pointwise: Option<&'a [Complex<T>]>,
}

/// The register constants of the butterfly: the cosines, and the sines
/// times the quarter turn's sign pair.
#[derive(Clone, Copy)]
struct Constants<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    c: [Vector<T, A>; 3],
    s_turn: [Vector<T, A>; 3],
}

/// The seven-point butterfly on one register of complexes per arm.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "seven arms is the butterfly's arity, not a parameter list"
)]
#[inline(always)]
fn dft7<T, A>(
    k: Constants<T, A>,
    a0: Vector<T, A>,
    a1: Vector<T, A>,
    a2: Vector<T, A>,
    a3: Vector<T, A>,
    a4: Vector<T, A>,
    a5: Vector<T, A>,
    a6: Vector<T, A>,
) -> [Vector<T, A>; 7]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let xr1 = a1 + a6;
    let xr2 = a2 + a5;
    let xr3 = a3 + a4;
    let xs1 = (a1 - a6).swap_adjacent();
    let xs2 = (a2 - a5).swap_adjacent();
    let xs3 = (a3 - a4).swap_adjacent();
    let [c1, c2, c3] = k.c;
    let [s1, s2, s3] = k.s_turn;
    let re1 = xr3.mul_add(c3, xr2.mul_add(c2, xr1.mul_add(c1, a0)));
    let re2 = xr3.mul_add(c1, xr2.mul_add(c3, xr1.mul_add(c2, a0)));
    let re3 = xr3.mul_add(c2, xr2.mul_add(c1, xr1.mul_add(c3, a0)));
    let d1 = xs3.mul_add(s3, xs2.mul_add(s2, xs1 * s1));
    let d2 = xs1.mul_sub(s2, xs2.mul_add(s3, xs3 * s1));
    let d3 = xs3.mul_add(s2, xs1.mul_sub(s3, xs2 * s1));
    [
        a0 + (xr1 + (xr2 + xr3)),
        re1 + d1,
        re2 + d2,
        re3 + d3,
        re3 - d3,
        re2 - d2,
        re1 - d1,
    ]
}

/// The scalar seven-point butterfly on twiddled arms.
#[inline]
fn dft7_scalar<T: WinogradScalar, const INVERSE: bool>(a: [Complex<T>; 7]) -> [Complex<T>; 7] {
    let turn = |v: Complex<T>| {
        if INVERSE {
            Complex::new(-v.im, v.re)
        } else {
            Complex::new(v.im, -v.re)
        }
    };
    let xr1 = a[1] + a[6];
    let xr2 = a[2] + a[5];
    let xr3 = a[3] + a[4];
    let xi1 = turn(a[1] - a[6]);
    let xi2 = turn(a[2] - a[5]);
    let xi3 = turn(a[3] - a[4]);
    let (c1, c2, c3) = (T::from_f64(C1), T::from_f64(C2), T::from_f64(C3));
    let (s1, s2, s3) = (T::from_f64(S1), T::from_f64(S2), T::from_f64(S3));
    let x0 = a[0];
    let lin = |p: T, q: T, r: T| {
        Complex::new(
            x0.re + xr1.re * p + xr2.re * q + xr3.re * r,
            x0.im + xr1.im * p + xr2.im * q + xr3.im * r,
        )
    };
    let re1 = lin(c1, c2, c3);
    let re2 = lin(c2, c3, c1);
    let re3 = lin(c3, c1, c2);
    let d1 = Complex::new(
        xi1.re * s1 + xi2.re * s2 + xi3.re * s3,
        xi1.im * s1 + xi2.im * s2 + xi3.im * s3,
    );
    let d2 = Complex::new(
        xi1.re * s2 - xi2.re * s3 - xi3.re * s1,
        xi1.im * s2 - xi2.im * s3 - xi3.im * s1,
    );
    let d3 = Complex::new(
        xi1.re * s3 - xi2.re * s1 + xi3.re * s2,
        xi1.im * s3 - xi2.im * s1 + xi3.im * s2,
    );
    [
        x0 + (xr1 + (xr2 + xr3)),
        re1 + d1,
        re2 + d2,
        re3 + d3,
        re3 - d3,
        re2 - d2,
        re1 - d1,
    ]
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
        let a: [Complex<T>; 7] = core::array::from_fn(|arm| {
            let x = src[arm * stride + at];
            if arm == 0 {
                x
            } else {
                cmul_scalar(x, tw[(arm - 1) * prev_len + j])
            }
        });
        let b = dft7_scalar::<T, INVERSE>(a);
        for (arm, row) in b.into_iter().enumerate() {
            dst[dst_base + j + arm * prev_len] = row;
        }
    }
}

impl<T, const INVERSE: bool> LaneKernel<T> for FlatPassR7<'_, T, INVERSE>
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
            src.len() >= 7 * stride
                && dst.len() >= g_count * stage_chunk
                && stage_chunk >= 7 * prev_len
                && tw.len() >= 6 * prev_len,
            "invariant: a radix-7 pass reads seven rows of g_count * prev_len complexes and six twiddle rows, and writes g_count blocks of stage_chunk"
        );
        let (lo, hi) = if INVERSE { (-1.0, 1.0) } else { (1.0, -1.0) };
        let k = Constants {
            c: [C1, C2, C3].map(|c| simd.splat(T::from_f64(c))),
            s_turn: [S1, S2, S3]
                .map(|s| Vector::<T, A>::splat_pair(T::from_f64(lo * s), T::from_f64(hi * s))),
        };

        if prev_len == 1 {
            // No twiddle; a register holds `per` groups of one arm and a
            // group's seven arms are consecutive in `dst`: the arm scatter
            // transposes them in registers, leaving the groups its run-over
            // needs to the scalar tail.
            let slack = scatter_spill::<T, A, 7>().div_ceil(7);
            let mut g = 0;
            while g + per + slack <= g_count {
                // SAFETY: `g + per <= g_count = stride`, so each arm's row
                // stays inside `src`, and `7 (g + per) + spill <= dst.len()`.
                unsafe {
                    let b = dft7(
                        k,
                        load::<T, A>(src, g),
                        load::<T, A>(src, stride + g),
                        load::<T, A>(src, 2 * stride + g),
                        load::<T, A>(src, 3 * stride + g),
                        load::<T, A>(src, 4 * stride + g),
                        load::<T, A>(src, 5 * stride + g),
                        load::<T, A>(src, 6 * stride + g),
                    );
                    store_arms::<T, A, 7>(b, dst, 7 * g);
                }
                g += per;
            }
            for (g, out) in dst[7 * g..7 * g_count]
                .chunks_exact_mut(7)
                .enumerate()
                .map(|(k, out)| (g + k, out))
            {
                let b =
                    dft7_scalar::<T, INVERSE>(core::array::from_fn(|arm| src[arm * stride + g]));
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
            let w5 = duplicated_row::<T, A>(&tw[4 * prev_len..5 * prev_len]);
            let w6 = duplicated_row::<T, A>(&tw[5 * prev_len..6 * prev_len]);
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
                    let a5 = cmul(load::<T, A>(src, 5 * stride + at), w5);
                    let a6 = cmul(load::<T, A>(src, 6 * stride + at), w6);
                    store_arm_halves::<T, A, 7>(
                        dft7(k, a0, a1, a2, a3, a4, a5, a6),
                        dst,
                        g * stage_chunk,
                    );
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
                    // `dst_base + j + 6 prev_len + per <= g_count * stage_chunk`.
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
                        let a5 = cmul(
                            load::<T, A>(src, 5 * stride + at),
                            load::<T, A>(tw, 4 * prev_len + j),
                        );
                        let a6 = cmul(
                            load::<T, A>(src, 6 * stride + at),
                            load::<T, A>(tw, 5 * prev_len + j),
                        );
                        let b = dft7(k, a0, a1, a2, a3, a4, a5, a6);
                        for arm in 0..7 {
                            store(b[arm], dst, dst_base + j + arm * prev_len);
                        }
                    }
                    j += per;
                }
                if j < prev_len {
                    scalar_columns::<T, INVERSE>(src, dst, tw, stride, prev_len, stage_chunk, g, j);
                }
            }
        }

        if let Some(factors) = pointwise {
            apply_pointwise::<T, A>(dst, factors);
        }
        true
    }
}
