//! Register-resident radix butterflies shared by fixed-size kernels.

use hermes_simd::{ComplexReg, LaneScalar, SimdArch, SimdKernel, Vector};

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
pub(super) fn radix8<T, A, const INVERSE: bool>(
    values: [ComplexReg<T, A>; 8],
    half_root2: Vector<T, A>,
) -> [ComplexReg<T, A>; 8]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let even = radix4::<T, A, INVERSE>([values[0], values[2], values[4], values[6]]);
    let odd = radix4::<T, A, INVERSE>([values[1], values[3], values[5], values[7]]);
    let odd = [
        odd[0],
        root2_twiddle::<T, A, INVERSE, false>(odd[1], half_root2),
        rot90::<T, A, INVERSE>(odd[2]),
        root2_twiddle::<T, A, INVERSE, true>(odd[3], half_root2),
    ];
    let (out0, out4) = even[0].butterfly(odd[0]);
    let (out1, out5) = even[1].butterfly(odd[1]);
    let (out2, out6) = even[2].butterfly(odd[2]);
    let (out3, out7) = even[3].butterfly(odd[3]);
    [out0, out1, out2, out3, out4, out5, out6, out7]
}
