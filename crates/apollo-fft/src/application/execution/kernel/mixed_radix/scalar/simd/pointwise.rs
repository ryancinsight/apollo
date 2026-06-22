use hermes_simd::{interleaved_complex_mul_assign, PreferredArch};
use num_complex::{Complex32, Complex64};

#[inline]
fn pointwise_mul_precise_hermes<const CONJ_B: bool>(a: &mut [Complex64], b: &[Complex64]) {
    let a_lanes = bytemuck::cast_slice_mut(a);
    let b_lanes = bytemuck::cast_slice(b);
    interleaved_complex_mul_assign::<f64, PreferredArch, CONJ_B>(a_lanes, b_lanes)
        .expect("pointwise operands must have matching complex lengths");
}

#[inline]
fn pointwise_mul_reduced_hermes<const CONJ_B: bool>(a: &mut [Complex32], b: &[Complex32]) {
    let a_lanes = bytemuck::cast_slice_mut(a);
    let b_lanes = bytemuck::cast_slice(b);
    interleaved_complex_mul_assign::<f32, PreferredArch, CONJ_B>(a_lanes, b_lanes)
        .expect("pointwise operands must have matching complex lengths");
}

#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) fn pointwise_mul_precise<
    const CONJ_B: bool,
>(
    a: &mut [Complex64],
    b: &[Complex64],
) {
    debug_assert_eq!(a.len(), b.len());
    pointwise_mul_precise_hermes::<CONJ_B>(a, b);
}

#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) fn pointwise_mul_reduced<
    const CONJ_B: bool,
>(
    a: &mut [Complex32],
    b: &[Complex32],
) {
    debug_assert_eq!(a.len(), b.len());
    pointwise_mul_reduced_hermes::<CONJ_B>(a, b);
}
