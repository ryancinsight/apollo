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
//! `x ± i q` is the swapped `q` subtracted on the even lanes and added on
//! the odd ones (`fmaddsub` by one), or the mirror.

use super::{apply_pointwise, cmul, load, store};
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

/// The register constants of the butterfly, the sines signed by direction.
#[derive(Clone, Copy)]
struct Constants<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    one: Vector<T, A>,
    c1: Vector<T, A>,
    c2: Vector<T, A>,
    s1: Vector<T, A>,
    s2: Vector<T, A>,
}

/// The five-point butterfly on one register of complexes per arm.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[inline(always)]
fn dft5<T, A>(k: Constants<T, A>, a: [Vector<T, A>; 5]) -> [Vector<T, A>; 5]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let t1 = a[1] + a[4];
    let t2 = a[1] - a[4];
    let t3 = a[2] + a[3];
    let t4 = a[2] - a[3];
    let m1 = t3.mul_add(k.c2, t1 * k.c1);
    let m2 = t3.mul_add(k.c1, t1 * k.c2);
    let q3 = t4.mul_add(k.s2, t2 * k.s1).swap_adjacent();
    let q4 = t2.mul_add(k.s2, -(t4 * k.s1)).swap_adjacent();
    let a1c = a[0] + m1;
    let a2c = a[0] + m2;
    // `x + i q` reads `(xr - qi, xi + qr)`: the swapped `q` subtracted on the
    // even lanes and added on the odd.
    [
        a[0] + (t1 + t3),
        a1c.fmaddsub(k.one, q3),
        a2c.fmaddsub(k.one, q4),
        a2c.fmsubadd(k.one, q4),
        a1c.fmsubadd(k.one, q3),
    ]
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
        if per == 0 || per > 8 {
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
        let (s1, s2) = if INVERSE { (S1, S2) } else { (-S1, -S2) };
        let k = Constants {
            one: simd.splat(T::from_f64(1.0)),
            c1: simd.splat(T::from_f64(C1)),
            c2: simd.splat(T::from_f64(C2)),
            s1: simd.splat(T::from_f64(s1)),
            s2: simd.splat(T::from_f64(s2)),
        };

        if prev_len == 1 {
            // No twiddle; a register holds `per` groups of one arm and a
            // group's five arms are consecutive in `dst`: the arms go through
            // a tile of `per` groups.
            let zero = Complex::new(T::from_f64(0.0), T::from_f64(0.0));
            let mut tile = [zero; 5 * 8];
            let mut g = 0;
            while g + per <= g_count {
                // SAFETY: `g + per <= g_count = stride`, so each arm's row stays
                // inside `src`; the tile holds `5 per` complexes.
                unsafe {
                    // Straight-line loads: a closure here compiled out of the
                    // target-feature frame with every vector op a call.
                    let mut a = [load::<T, A>(src, g); 5];
                    for (arm, slot) in a.iter_mut().enumerate().skip(1) {
                        *slot = load::<T, A>(src, arm * stride + g);
                    }
                    let b = dft5(k, a);
                    for (arm, row) in b.into_iter().enumerate() {
                        store(row, &mut tile, arm * per);
                    }
                }
                for (i, group) in dst[5 * g..5 * (g + per)].chunks_exact_mut(5).enumerate() {
                    for (arm, out) in group.iter_mut().enumerate() {
                        *out = tile[arm * per + i];
                    }
                }
                g += per;
            }
            while g < g_count {
                let b =
                    dft5_scalar::<T, INVERSE>(core::array::from_fn(|arm| src[arm * stride + g]));
                dst[5 * g..5 * g + 5].copy_from_slice(&b);
                g += 1;
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
                    unsafe {
                        let at = src_base + j;
                        let mut a = [load::<T, A>(src, at); 5];
                        for (arm, slot) in a.iter_mut().enumerate().skip(1) {
                            *slot = cmul(
                                load::<T, A>(src, arm * stride + at),
                                load::<T, A>(tw, (arm - 1) * prev_len + j),
                            );
                        }
                        let b = dft5(k, a);
                        for (arm, row) in b.into_iter().enumerate() {
                            store(row, dst, dst_base + j + arm * prev_len);
                        }
                    }
                    j += per;
                }
                while j < prev_len {
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
