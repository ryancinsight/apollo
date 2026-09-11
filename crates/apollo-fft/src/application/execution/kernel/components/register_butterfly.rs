//! Register-resident radix butterflies shared by fixed-size kernels.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{ComplexReg, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

/// Multiplies by `-i` forward or `+i` inverse.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
pub(super) fn rot90<T, A, const INVERSE: bool>(value: ComplexReg<T, A>) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    if INVERSE {
        value.mul_i()
    } else {
        value.mul_neg_i()
    }
}

/// Computes four lane-wise DFT-4s in natural order.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
pub(super) fn radix4<T, A, const INVERSE: bool>(
    values: [ComplexReg<T, A>; 4],
) -> [ComplexReg<T, A>; 4]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let (m0, m2) = values[0].butterfly(values[2]);
    let (m1, m3) = values[1].butterfly(values[3]);
    let (out0, out2) = m0.butterfly(m1);
    let (out1, out3) = m2.butterfly(rot90::<T, A, INVERSE>(m3));
    [out0, out1, out2, out3]
}

/// The eighth-turn twiddles `W_8^1` and `W_8^3` a radix-8 applies to its
/// odd branch; how they are formed is the caller's choice of registers.
pub(super) trait Eighths<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    /// `W_8^1 v`.
    fn one(&self, v: ComplexReg<T, A>) -> ComplexReg<T, A>;
    /// `W_8^3 v`.
    fn three(&self, v: ComplexReg<T, A>) -> ComplexReg<T, A>;
}

/// The eighths from the real `sqrt(2)/2` alone: a rotation, one butterfly
/// arm, and a real multiply — four operations and a chain of twelve cycles
/// each, no table.
pub(super) struct HalfRoot2<T, A, const INVERSE: bool>(pub(super) Vector<T, A>)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>;

impl<T, A, const INVERSE: bool> Eighths<T, A> for HalfRoot2<T, A, INVERSE>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn one(&self, v: ComplexReg<T, A>) -> ComplexReg<T, A> {
        root2_twiddle::<T, A, INVERSE, false>(v, self.0)
    }

    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn three(&self, v: ComplexReg<T, A>) -> ComplexReg<T, A> {
        root2_twiddle::<T, A, INVERSE, true>(v, self.0)
    }
}

/// The eighths as dup-split broadcast pairs from a plan table: one swap,
/// one multiply, one `fmaddsub` each, a chain of nine cycles. The table
/// carries the direction in its values.
pub(super) struct DupSplitEighths<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    /// `W_8^1` as `([re; n], [im; n])`.
    pub(super) one: (Vector<T, A>, Vector<T, A>),
    /// `W_8^3` as `([re; n], [im; n])`.
    pub(super) three: (Vector<T, A>, Vector<T, A>),
}

impl<T, A> Eighths<T, A> for DupSplitEighths<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn one(&self, v: ComplexReg<T, A>) -> ComplexReg<T, A> {
        let vi = v.into_interleaved();
        ComplexReg::from_interleaved(vi.fmaddsub(self.one.0, vi.swap_adjacent() * self.one.1))
    }

    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn three(&self, v: ComplexReg<T, A>) -> ComplexReg<T, A> {
        let vi = v.into_interleaved();
        ComplexReg::from_interleaved(vi.fmaddsub(self.three.0, vi.swap_adjacent() * self.three.1))
    }
}

/// Multiplies by `W_8^1` or `W_8^3` without a complex multiply.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
pub(super) fn root2_twiddle<T, A, const INVERSE: bool, const SUBTRACT: bool>(
    value: ComplexReg<T, A>,
    half_root2: Vector<T, A>,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let (sum, difference) = rot90::<T, A, INVERSE>(value).butterfly(value);
    let selected = if SUBTRACT { difference } else { sum };
    ComplexReg::from_interleaved(selected.into_interleaved() * half_root2)
}

/// Computes four lane-wise DFT-8s in natural order.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
pub(super) fn radix8<T, A, const INVERSE: bool, E>(
    values: [ComplexReg<T, A>; 8],
    eighths: &E,
) -> [ComplexReg<T, A>; 8]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    E: Eighths<T, A>,
{
    let even = radix4::<T, A, INVERSE>([values[0], values[2], values[4], values[6]]);
    let odd = radix4::<T, A, INVERSE>([values[1], values[3], values[5], values[7]]);
    let odd = [
        odd[0],
        eighths.one(odd[1]),
        rot90::<T, A, INVERSE>(odd[2]),
        eighths.three(odd[3]),
    ];
    let (out0, out4) = even[0].butterfly(odd[0]);
    let (out1, out5) = even[1].butterfly(odd[1]);
    let (out2, out6) = even[2].butterfly(odd[2]);
    let (out3, out7) = even[3].butterfly(odd[3]);
    [out0, out1, out2, out3, out4, out5, out6, out7]
}

/// The third-turn constants a radix-3 applies: `-1/2` on the sum of its
/// outer arms and `sin(2π/3)` on their rotated difference, both real scales
/// splat from the capability token, so no host probe stands in a kernel.
pub(super) struct Thirds<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    half_negative: Vector<T, A>,
    sine: Vector<T, A>,
}

impl<T, A> Thirds<T, A>
where
    T: LaneScalar + MixedRadixScalar,
    A: SimdArch + SimdKernel<T>,
{
    /// `sin(2π/3)`, exactly rounded.
    const SINE: f64 = 0.866_025_403_784_438_6;

    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    pub(super) fn new(simd: Simd<T, A>) -> Self {
        Self {
            half_negative: simd.splat(T::from_precise(-0.5)),
            sine: simd.splat(T::from_precise(Self::SINE)),
        }
    }
}

/// Computes lane-wise DFT-3s across three registers, natural order.
///
/// `[a0 + s, m0 + m1, m0 - m1]` for `s = a1 + a2`, `m0 = a0 - s/2` and
/// `m1 = -+ i sin(2π/3) (a1 - a2)`, the sign by direction.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
pub(super) fn radix3<T, A, const INVERSE: bool>(
    values: [ComplexReg<T, A>; 3],
    thirds: &Thirds<T, A>,
) -> [ComplexReg<T, A>; 3]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let (sum, difference) = values[1].butterfly(values[2]);
    let m0 =
        values[0] + ComplexReg::from_interleaved(sum.into_interleaved() * thirds.half_negative);
    let m1 = ComplexReg::from_interleaved(
        rot90::<T, A, INVERSE>(difference).into_interleaved() * thirds.sine,
    );
    [values[0] + sum, m0 + m1, m0 - m1]
}

/// The fifth-turn constants a radix-5 applies: the cosines `c1 = cos(2π/5)`,
/// `c2 = cos(4π/5)` broadcast, and the sines `s1 = sin(2π/5)`,
/// `s2 = sin(4π/5)` carrying the direction and the `(-1, 1)` of a quarter
/// turn on alternate lanes, so the rotated cross terms are one fused
/// multiply-add on the swapped differences.
pub(super) struct Fifths<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    pub(super) c1: Vector<T, A>,
    pub(super) c2: Vector<T, A>,
    pub(super) s1_turn: Vector<T, A>,
    pub(super) s2_turn: Vector<T, A>,
}

/// `cos(2π/5)`, exactly rounded.
pub(super) const FIFTH_C1: f64 = 0.309_016_994_374_947_45;
/// `cos(4π/5)`, exactly rounded.
pub(super) const FIFTH_C2: f64 = -0.809_016_994_374_947_5;
/// `sin(2π/5)`, exactly rounded.
pub(super) const FIFTH_S1: f64 = 0.951_056_516_295_153_5;
/// `sin(4π/5)`, exactly rounded.
pub(super) const FIFTH_S2: f64 = 0.587_785_252_292_473_1;

/// Computes lane-wise DFT-5s across five registers, natural order.
///
/// ```text
///   t1 = a1 + a4,  t2 = a1 - a4,  t3 = a2 + a3,  t4 = a2 - a3
///   m1 = c1 t1 + c2 t3,   m2 = c2 t1 + c1 t3
///   q3 = s1 t2 + s2 t4,   q4 = s2 t2 - s1 t4
///   b0 = a0 + t1 + t3
///   b1 = a0 + m1 + i q3,  b4 = a0 + m1 - i q3
///   b2 = a0 + m2 + i q4,  b3 = a0 + m2 - i q4
/// ```
///
/// `i q` is the swapped `q` times `(-1, 1)`, and the swap distributes over
/// the sums, so `i q3` and `i q4` are fused chains on the swapped
/// differences with the sines carrying the sign pair.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
pub(super) fn radix5<T, A>(values: [ComplexReg<T, A>; 5], k: &Fifths<T, A>) -> [ComplexReg<T, A>; 5]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let [a0, a1, a2, a3, a4] = [
        values[0].into_interleaved(),
        values[1].into_interleaved(),
        values[2].into_interleaved(),
        values[3].into_interleaved(),
        values[4].into_interleaved(),
    ];
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
    [
        ComplexReg::from_interleaved(a0 + (t1 + t3)),
        ComplexReg::from_interleaved(a1c + iq3),
        ComplexReg::from_interleaved(a2c + iq4),
        ComplexReg::from_interleaved(a2c - iq4),
        ComplexReg::from_interleaved(a1c - iq3),
    ]
}
