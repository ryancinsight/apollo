//! Unrolled small power-of-two codelets at precise lane density.
//!
//! Extracted verbatim from the `MixedRadixScalar` implementation so the trait
//! wiring and the unrolled codelet bodies occupy separate leaf modules.

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use super::super::simd::avx::{
    avx_cmul_precise, avx_fft4_parallel_precise, avx_fft8_parallel_precise, avx_fft8_precise,
    rotate_minus_i, rotate_plus_i, sse_cmul,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use super::super::twiddle_constants::{
    TWIDDLES_COMBINE_FWD_32, TWIDDLES_COMBINE_FWD_64, TWIDDLES_COMBINE_INV_32,
    TWIDDLES_COMBINE_INV_64,
};
use eunomia::Complex64;

/// Applies an unrolled codelet when `data.len()` is a supported size.
///
/// # Safety
///
/// Carries the `MixedRadixScalar::small_pot_inplace` contract unchanged.
#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn small_pot_inplace_precise<
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [Complex64],
) -> bool {
    let n = data.len();
    match n {
        2 => {
            small_pot_inplace_sized_precise::<2, INVERSE, NORMALIZE>(data);
            true
        }
        3 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 3]>();
            <f64 as crate::application::execution::kernel::mixed_radix::traits::ShortDft<3>>::dft::<
                INVERSE,
            >(data_ref);
            if INVERSE && NORMALIZE {
                let scale = Complex64::new(1.0 / 3.0, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        4 => {
            small_pot_inplace_sized_precise::<4, INVERSE, NORMALIZE>(data);
            true
        }
        5 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 5]>();
            <f64 as crate::application::execution::kernel::mixed_radix::traits::ShortDft<5>>::dft::<
                INVERSE,
            >(data_ref);
            if INVERSE && NORMALIZE {
                let scale = Complex64::new(0.2, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        6 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 6]>();
            crate::application::execution::kernel::components::winograd::dft6_impl::<f64, INVERSE>(
                data_ref,
            );
            if INVERSE && NORMALIZE {
                let scale = Complex64::new(1.0 / 6.0, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        7 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 7]>();
            <f64 as crate::application::execution::kernel::mixed_radix::traits::ShortDft<7>>::dft::<
                INVERSE,
            >(data_ref);
            if INVERSE && NORMALIZE {
                let scale = Complex64::new(1.0 / 7.0, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        8 => {
            small_pot_inplace_sized_precise::<8, INVERSE, NORMALIZE>(data);
            true
        }
        16 => {
            small_pot_inplace_sized_precise::<16, INVERSE, NORMALIZE>(data);
            true
        }
        32 => {
            small_pot_inplace_sized_precise::<32, INVERSE, NORMALIZE>(data);
            true
        }
        64 => {
            small_pot_inplace_sized_precise::<64, INVERSE, NORMALIZE>(data);
            true
        }
        9 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 9]>();
            crate::application::execution::kernel::components::winograd::dft9_impl::<f64, INVERSE>(
                data_ref,
            );
            if INVERSE && NORMALIZE {
                let scale = Complex64::new(1.0 / 9.0, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        _ => false,
    }
}

/// Applies the unrolled codelet for the const-selected size `N`.
///
/// # Safety
///
/// Carries the `MixedRadixScalar::small_pot_inplace_sized` contract unchanged.
#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn small_pot_inplace_sized_precise<
    const N: usize,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [Complex64],
) {
    match N {
        2 => {
            let a = *data.get_unchecked(0);
            let b = *data.get_unchecked(1);
            if INVERSE && NORMALIZE {
                let half = Complex64::new(0.5, 0.0);
                *data.get_unchecked_mut(0) = (a + b) * half;
                *data.get_unchecked_mut(1) = (a - b) * half;
            } else {
                *data.get_unchecked_mut(0) = a + b;
                *data.get_unchecked_mut(1) = a - b;
            }
        }
        4 => {
            let x0 = *data.get_unchecked(0);
            let x1 = *data.get_unchecked(1);
            let x2 = *data.get_unchecked(2);
            let x3 = *data.get_unchecked(3);
            let a0 = x0 + x2;
            let a1 = x1 + x3;
            let a2 = x0 - x2;
            let a3 = x1 - x3;
            let i_a3 = Complex64::new(-a3.im, a3.re);
            if INVERSE && NORMALIZE {
                let quarter = Complex64::new(0.25, 0.0);
                *data.get_unchecked_mut(0) = (a0 + a1) * quarter;
                *data.get_unchecked_mut(2) = (a0 - a1) * quarter;
                if INVERSE {
                    *data.get_unchecked_mut(1) = (a2 + i_a3) * quarter;
                    *data.get_unchecked_mut(3) = (a2 - i_a3) * quarter;
                } else {
                    *data.get_unchecked_mut(1) = (a2 - i_a3) * quarter;
                    *data.get_unchecked_mut(3) = (a2 + i_a3) * quarter;
                }
            } else {
                *data.get_unchecked_mut(0) = a0 + a1;
                *data.get_unchecked_mut(2) = a0 - a1;
                if INVERSE {
                    *data.get_unchecked_mut(1) = a2 + i_a3;
                    *data.get_unchecked_mut(3) = a2 - i_a3;
                } else {
                    *data.get_unchecked_mut(1) = a2 - i_a3;
                    *data.get_unchecked_mut(3) = a2 + i_a3;
                }
            }
        }
        8 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                use std::arch::x86_64::{
                    _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_set_pd, _mm_storeu_pd,
                    _mm_sub_pd,
                };
                let ptr = data.as_mut_ptr().cast::<f64>();
                let x0 = _mm_loadu_pd(ptr);
                let x1 = _mm_loadu_pd(ptr.add(2));
                let x2 = _mm_loadu_pd(ptr.add(4));
                let x3 = _mm_loadu_pd(ptr.add(6));
                let x4 = _mm_loadu_pd(ptr.add(8));
                let x5 = _mm_loadu_pd(ptr.add(10));
                let x6 = _mm_loadu_pd(ptr.add(12));
                let x7 = _mm_loadu_pd(ptr.add(14));

                // Bit-reversed loading into variables
                let r0 = x0;
                let r1 = x4;
                let r2 = x2;
                let r3 = x6;
                let r4 = x1;
                let r5 = x5;
                let r6 = x3;
                let r7 = x7;

                // Stage 1
                let a0 = _mm_add_pd(r0, r1);
                let a1 = _mm_sub_pd(r0, r1);
                let a2 = _mm_add_pd(r2, r3);
                let a3 = _mm_sub_pd(r2, r3);
                let a4 = _mm_add_pd(r4, r5);
                let a5 = _mm_sub_pd(r4, r5);
                let a6 = _mm_add_pd(r6, r7);
                let a7 = _mm_sub_pd(r6, r7);

                // Stage 2
                let a3_rot = if INVERSE {
                    rotate_plus_i(a3)
                } else {
                    rotate_minus_i(a3)
                };
                let a7_rot = if INVERSE {
                    rotate_plus_i(a7)
                } else {
                    rotate_minus_i(a7)
                };

                let b0 = _mm_add_pd(a0, a2);
                let b2 = _mm_sub_pd(a0, a2);
                let b1 = _mm_add_pd(a1, a3_rot);
                let b3 = _mm_sub_pd(a1, a3_rot);
                let b4 = _mm_add_pd(a4, a6);
                let b6 = _mm_sub_pd(a4, a6);
                let b5 = _mm_add_pd(a5, a7_rot);
                let b7 = _mm_sub_pd(a5, a7_rot);

                // Stage 3
                let c = std::f64::consts::FRAC_1_SQRT_2;
                let (w1, w3) = if INVERSE {
                    (_mm_set_pd(c, c), _mm_set_pd(c, -c))
                } else {
                    (_mm_set_pd(-c, c), _mm_set_pd(-c, -c))
                };

                let b5_tw = sse_cmul(b5, w1);
                let b6_tw = if INVERSE {
                    rotate_plus_i(b6)
                } else {
                    rotate_minus_i(b6)
                };
                let b7_tw = sse_cmul(b7, w3);

                let mut c0 = _mm_add_pd(b0, b4);
                let mut c4 = _mm_sub_pd(b0, b4);
                let mut c1 = _mm_add_pd(b1, b5_tw);
                let mut c5 = _mm_sub_pd(b1, b5_tw);
                let mut c2 = _mm_add_pd(b2, b6_tw);
                let mut c6 = _mm_sub_pd(b2, b6_tw);
                let mut c3 = _mm_add_pd(b3, b7_tw);
                let mut c7 = _mm_sub_pd(b3, b7_tw);

                if INVERSE && NORMALIZE {
                    let scale = _mm_set1_pd(0.125);
                    c0 = _mm_mul_pd(c0, scale);
                    c1 = _mm_mul_pd(c1, scale);
                    c2 = _mm_mul_pd(c2, scale);
                    c3 = _mm_mul_pd(c3, scale);
                    c4 = _mm_mul_pd(c4, scale);
                    c5 = _mm_mul_pd(c5, scale);
                    c6 = _mm_mul_pd(c6, scale);
                    c7 = _mm_mul_pd(c7, scale);
                }

                _mm_storeu_pd(ptr, c0);
                _mm_storeu_pd(ptr.add(2), c1);
                _mm_storeu_pd(ptr.add(4), c2);
                _mm_storeu_pd(ptr.add(6), c3);
                _mm_storeu_pd(ptr.add(8), c4);
                _mm_storeu_pd(ptr.add(10), c5);
                _mm_storeu_pd(ptr.add(12), c6);
                _mm_storeu_pd(ptr.add(14), c7);
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 8]>();
                crate::application::execution::kernel::components::winograd::dft8_array_impl::<
                    f64,
                    INVERSE,
                    false,
                >(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex64::new(0.125, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
            }
        }
        16 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                use std::arch::x86_64::{
                    _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_set_pd, _mm_storeu_pd,
                    _mm_sub_pd,
                };
                let ptr = data.as_mut_ptr().cast::<f64>();
                let mut x0 = _mm_loadu_pd(ptr);
                let mut x1 = _mm_loadu_pd(ptr.add(2));
                let mut x2 = _mm_loadu_pd(ptr.add(4));
                let mut x3 = _mm_loadu_pd(ptr.add(6));
                let mut x4 = _mm_loadu_pd(ptr.add(8));
                let mut x5 = _mm_loadu_pd(ptr.add(10));
                let mut x6 = _mm_loadu_pd(ptr.add(12));
                let mut x7 = _mm_loadu_pd(ptr.add(14));
                let mut x8 = _mm_loadu_pd(ptr.add(16));
                let mut x9 = _mm_loadu_pd(ptr.add(18));
                let mut x10 = _mm_loadu_pd(ptr.add(20));
                let mut x11 = _mm_loadu_pd(ptr.add(22));
                let mut x12 = _mm_loadu_pd(ptr.add(24));
                let mut x13 = _mm_loadu_pd(ptr.add(26));
                let mut x14 = _mm_loadu_pd(ptr.add(28));
                let mut x15 = _mm_loadu_pd(ptr.add(30));

                // Stage 1 (radix 2, distance 8)
                let a0 = _mm_add_pd(x0, x8);
                let a8 = _mm_sub_pd(x0, x8);
                let a1 = _mm_add_pd(x1, x9);
                let mut a9 = _mm_sub_pd(x1, x9);
                let a2 = _mm_add_pd(x2, x10);
                let mut a10 = _mm_sub_pd(x2, x10);
                let a3 = _mm_add_pd(x3, x11);
                let mut a11 = _mm_sub_pd(x3, x11);
                let a4 = _mm_add_pd(x4, x12);
                let mut a12 = _mm_sub_pd(x4, x12);
                let a5 = _mm_add_pd(x5, x13);
                let mut a13 = _mm_sub_pd(x5, x13);
                let a6 = _mm_add_pd(x6, x14);
                let mut a14 = _mm_sub_pd(x6, x14);
                let a7 = _mm_add_pd(x7, x15);
                let mut a15 = _mm_sub_pd(x7, x15);

                // Twiddle multiplications for Stage 1
                let c = std::f64::consts::FRAC_1_SQRT_2;
                let (w1, w2, w3, w5, w6, w7) = if INVERSE {
                    (
                        _mm_set_pd(0.38268343236508978, 0.9238795325112867),
                        _mm_set_pd(c, c),
                        _mm_set_pd(0.9238795325112867, 0.3826834323650898),
                        _mm_set_pd(0.9238795325112867, -0.3826834323650898),
                        _mm_set_pd(c, -c),
                        _mm_set_pd(0.38268343236508978, -0.9238795325112867),
                    )
                } else {
                    (
                        _mm_set_pd(-0.38268343236508978, 0.9238795325112867),
                        _mm_set_pd(-c, c),
                        _mm_set_pd(-0.9238795325112867, 0.3826834323650898),
                        _mm_set_pd(-0.9238795325112867, -0.3826834323650898),
                        _mm_set_pd(-c, -c),
                        _mm_set_pd(-0.38268343236508978, -0.9238795325112867),
                    )
                };

                a9 = sse_cmul(a9, w1);
                a10 = sse_cmul(a10, w2);
                a11 = sse_cmul(a11, w3);
                a12 = if INVERSE {
                    rotate_plus_i(a12)
                } else {
                    rotate_minus_i(a12)
                };
                a13 = sse_cmul(a13, w5);
                a14 = sse_cmul(a14, w6);
                a15 = sse_cmul(a15, w7);

                // Stage 2 (radix 2, distance 4)
                let b0 = _mm_add_pd(a0, a4);
                let b4 = _mm_sub_pd(a0, a4);
                let b1 = _mm_add_pd(a1, a5);
                let mut b5 = _mm_sub_pd(a1, a5);
                let b2 = _mm_add_pd(a2, a6);
                let mut b6 = _mm_sub_pd(a2, a6);
                let b3 = _mm_add_pd(a3, a7);
                let mut b7 = _mm_sub_pd(a3, a7);

                let b8 = _mm_add_pd(a8, a12);
                let b12 = _mm_sub_pd(a8, a12);
                let b9 = _mm_add_pd(a9, a13);
                let mut b13 = _mm_sub_pd(a9, a13);
                let b10 = _mm_add_pd(a10, a14);
                let mut b14 = _mm_sub_pd(a10, a14);
                let b11 = _mm_add_pd(a11, a15);
                let mut b15 = _mm_sub_pd(a11, a15);

                let (w8_1, w8_3) = if INVERSE {
                    (_mm_set_pd(c, c), _mm_set_pd(c, -c))
                } else {
                    (_mm_set_pd(-c, c), _mm_set_pd(-c, -c))
                };

                b5 = sse_cmul(b5, w8_1);
                b6 = if INVERSE {
                    rotate_plus_i(b6)
                } else {
                    rotate_minus_i(b6)
                };
                b7 = sse_cmul(b7, w8_3);

                b13 = sse_cmul(b13, w8_1);
                b14 = if INVERSE {
                    rotate_plus_i(b14)
                } else {
                    rotate_minus_i(b14)
                };
                b15 = sse_cmul(b15, w8_3);

                // Stage 3 (radix 2, distance 2)
                let c0 = _mm_add_pd(b0, b2);
                let c2 = _mm_sub_pd(b0, b2);
                let c1 = _mm_add_pd(b1, b3);
                let mut c3 = _mm_sub_pd(b1, b3);
                c3 = if INVERSE {
                    rotate_plus_i(c3)
                } else {
                    rotate_minus_i(c3)
                };

                let c4 = _mm_add_pd(b4, b6);
                let c6 = _mm_sub_pd(b4, b6);
                let c5 = _mm_add_pd(b5, b7);
                let mut c7 = _mm_sub_pd(b5, b7);
                c7 = if INVERSE {
                    rotate_plus_i(c7)
                } else {
                    rotate_minus_i(c7)
                };

                let c8 = _mm_add_pd(b8, b10);
                let c10 = _mm_sub_pd(b8, b10);
                let c9 = _mm_add_pd(b9, b11);
                let mut c11 = _mm_sub_pd(b9, b11);
                c11 = if INVERSE {
                    rotate_plus_i(c11)
                } else {
                    rotate_minus_i(c11)
                };

                let c12 = _mm_add_pd(b12, b14);
                let c14 = _mm_sub_pd(b12, b14);
                let c13 = _mm_add_pd(b13, b15);
                let mut c15 = _mm_sub_pd(b13, b15);
                c15 = if INVERSE {
                    rotate_plus_i(c15)
                } else {
                    rotate_minus_i(c15)
                };

                // Stage 4 (radix 2, distance 1)
                x0 = _mm_add_pd(c0, c1);
                x8 = _mm_sub_pd(c0, c1);
                x4 = _mm_add_pd(c2, c3);
                x12 = _mm_sub_pd(c2, c3);

                x2 = _mm_add_pd(c4, c5);
                x10 = _mm_sub_pd(c4, c5);
                x6 = _mm_add_pd(c6, c7);
                x14 = _mm_sub_pd(c6, c7);

                x1 = _mm_add_pd(c8, c9);
                x9 = _mm_sub_pd(c8, c9);
                x5 = _mm_add_pd(c10, c11);
                x13 = _mm_sub_pd(c10, c11);

                x3 = _mm_add_pd(c12, c13);
                x11 = _mm_sub_pd(c12, c13);
                x7 = _mm_add_pd(c14, c15);
                x15 = _mm_sub_pd(c14, c15);

                if INVERSE && NORMALIZE {
                    let scale = _mm_set1_pd(0.0625);
                    x0 = _mm_mul_pd(x0, scale);
                    x1 = _mm_mul_pd(x1, scale);
                    x2 = _mm_mul_pd(x2, scale);
                    x3 = _mm_mul_pd(x3, scale);
                    x4 = _mm_mul_pd(x4, scale);
                    x5 = _mm_mul_pd(x5, scale);
                    x6 = _mm_mul_pd(x6, scale);
                    x7 = _mm_mul_pd(x7, scale);
                    x8 = _mm_mul_pd(x8, scale);
                    x9 = _mm_mul_pd(x9, scale);
                    x10 = _mm_mul_pd(x10, scale);
                    x11 = _mm_mul_pd(x11, scale);
                    x12 = _mm_mul_pd(x12, scale);
                    x13 = _mm_mul_pd(x13, scale);
                    x14 = _mm_mul_pd(x14, scale);
                    x15 = _mm_mul_pd(x15, scale);
                }

                _mm_storeu_pd(ptr, x0);
                _mm_storeu_pd(ptr.add(2), x1);
                _mm_storeu_pd(ptr.add(4), x2);
                _mm_storeu_pd(ptr.add(6), x3);
                _mm_storeu_pd(ptr.add(8), x4);
                _mm_storeu_pd(ptr.add(10), x5);
                _mm_storeu_pd(ptr.add(12), x6);
                _mm_storeu_pd(ptr.add(14), x7);
                _mm_storeu_pd(ptr.add(16), x8);
                _mm_storeu_pd(ptr.add(18), x9);
                _mm_storeu_pd(ptr.add(20), x10);
                _mm_storeu_pd(ptr.add(22), x11);
                _mm_storeu_pd(ptr.add(24), x12);
                _mm_storeu_pd(ptr.add(26), x13);
                _mm_storeu_pd(ptr.add(28), x14);
                _mm_storeu_pd(ptr.add(30), x15);
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 16]>();
                crate::application::execution::kernel::components::winograd::dft16_impl::<
                    f64,
                    INVERSE,
                >(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex64::new(0.0625, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
            }
        }
        32 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                use std::arch::x86_64::{
                    _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute2f128_pd, _mm256_set1_pd,
                    _mm256_storeu_pd,
                };
                let ptr = data.as_mut_ptr().cast::<f64>();

                let regs = [
                    _mm256_loadu_pd(ptr),
                    _mm256_loadu_pd(ptr.add(4)),
                    _mm256_loadu_pd(ptr.add(8)),
                    _mm256_loadu_pd(ptr.add(12)),
                    _mm256_loadu_pd(ptr.add(16)),
                    _mm256_loadu_pd(ptr.add(20)),
                    _mm256_loadu_pd(ptr.add(24)),
                    _mm256_loadu_pd(ptr.add(28)),
                    _mm256_loadu_pd(ptr.add(32)),
                    _mm256_loadu_pd(ptr.add(36)),
                    _mm256_loadu_pd(ptr.add(40)),
                    _mm256_loadu_pd(ptr.add(44)),
                    _mm256_loadu_pd(ptr.add(48)),
                    _mm256_loadu_pd(ptr.add(52)),
                    _mm256_loadu_pd(ptr.add(56)),
                    _mm256_loadu_pd(ptr.add(60)),
                ];

                let [c01_0, c01_1, c01_2, c01_3] =
                    avx_fft4_parallel_precise::<INVERSE>(regs[0], regs[4], regs[8], regs[12]);
                let [c23_0, c23_1, c23_2, c23_3] =
                    avx_fft4_parallel_precise::<INVERSE>(regs[1], regs[5], regs[9], regs[13]);
                let [c45_0, c45_1, c45_2, c45_3] =
                    avx_fft4_parallel_precise::<INVERSE>(regs[2], regs[6], regs[10], regs[14]);
                let [c67_0, c67_1, c67_2, c67_3] =
                    avx_fft4_parallel_precise::<INVERSE>(regs[3], regs[7], regs[11], regs[15]);

                let tw_table = if INVERSE {
                    &TWIDDLES_COMBINE_INV_32
                } else {
                    &TWIDDLES_COMBINE_FWD_32
                };

                let mut c01 = [c01_0, c01_1, c01_2, c01_3];
                let mut c23 = [c23_0, c23_1, c23_2, c23_3];
                let mut c45 = [c45_0, c45_1, c45_2, c45_3];
                let mut c67 = [c67_0, c67_1, c67_2, c67_3];

                let tw_ptr = tw_table.as_ptr().cast::<f64>();

                // k = 1
                let base_1 = 16;
                c01[1] = avx_cmul_precise(c01[1], _mm256_loadu_pd(tw_ptr.add(base_1)));
                c23[1] = avx_cmul_precise(c23[1], _mm256_loadu_pd(tw_ptr.add(base_1 + 4)));
                c45[1] = avx_cmul_precise(c45[1], _mm256_loadu_pd(tw_ptr.add(base_1 + 8)));
                c67[1] = avx_cmul_precise(c67[1], _mm256_loadu_pd(tw_ptr.add(base_1 + 12)));

                // k = 2
                let base_2 = 32;
                c01[2] = avx_cmul_precise(c01[2], _mm256_loadu_pd(tw_ptr.add(base_2)));
                c23[2] = avx_cmul_precise(c23[2], _mm256_loadu_pd(tw_ptr.add(base_2 + 4)));
                c45[2] = avx_cmul_precise(c45[2], _mm256_loadu_pd(tw_ptr.add(base_2 + 8)));
                c67[2] = avx_cmul_precise(c67[2], _mm256_loadu_pd(tw_ptr.add(base_2 + 12)));

                // k = 3
                let base_3 = 48;
                c01[3] = avx_cmul_precise(c01[3], _mm256_loadu_pd(tw_ptr.add(base_3)));
                c23[3] = avx_cmul_precise(c23[3], _mm256_loadu_pd(tw_ptr.add(base_3 + 4)));
                c45[3] = avx_cmul_precise(c45[3], _mm256_loadu_pd(tw_ptr.add(base_3 + 8)));
                c67[3] = avx_cmul_precise(c67[3], _mm256_loadu_pd(tw_ptr.add(base_3 + 12)));

                let [r0_0, r0_1, r0_2, r0_3] =
                    avx_fft8_precise::<INVERSE>(c01[0], c23[0], c45[0], c67[0]);
                let [r1_0, r1_1, r1_2, r1_3] =
                    avx_fft8_precise::<INVERSE>(c01[1], c23[1], c45[1], c67[1]);
                let [r2_0, r2_1, r2_2, r2_3] =
                    avx_fft8_precise::<INVERSE>(c01[2], c23[2], c45[2], c67[2]);
                let [r3_0, r3_1, r3_2, r3_3] =
                    avx_fft8_precise::<INVERSE>(c01[3], c23[3], c45[3], c67[3]);

                let mut out0_0 = _mm256_permute2f128_pd::<0x20>(r0_0, r1_0);
                let mut out0_1 = _mm256_permute2f128_pd::<0x20>(r2_0, r3_0);
                let mut out1_0 = _mm256_permute2f128_pd::<0x31>(r0_0, r1_0);
                let mut out1_1 = _mm256_permute2f128_pd::<0x31>(r2_0, r3_0);

                let mut out2_0 = _mm256_permute2f128_pd::<0x20>(r0_1, r1_1);
                let mut out2_1 = _mm256_permute2f128_pd::<0x20>(r2_1, r3_1);
                let mut out3_0 = _mm256_permute2f128_pd::<0x31>(r0_1, r1_1);
                let mut out3_1 = _mm256_permute2f128_pd::<0x31>(r2_1, r3_1);

                let mut out4_0 = _mm256_permute2f128_pd::<0x20>(r0_2, r1_2);
                let mut out4_1 = _mm256_permute2f128_pd::<0x20>(r2_2, r3_2);
                let mut out5_0 = _mm256_permute2f128_pd::<0x31>(r0_2, r1_2);
                let mut out5_1 = _mm256_permute2f128_pd::<0x31>(r2_2, r3_2);

                let mut out6_0 = _mm256_permute2f128_pd::<0x20>(r0_3, r1_3);
                let mut out6_1 = _mm256_permute2f128_pd::<0x20>(r2_3, r3_3);
                let mut out7_0 = _mm256_permute2f128_pd::<0x31>(r0_3, r1_3);
                let mut out7_1 = _mm256_permute2f128_pd::<0x31>(r2_3, r3_3);

                if INVERSE && NORMALIZE {
                    let scale = _mm256_set1_pd(1.0 / 32.0);
                    out0_0 = _mm256_mul_pd(out0_0, scale);
                    out0_1 = _mm256_mul_pd(out0_1, scale);
                    out1_0 = _mm256_mul_pd(out1_0, scale);
                    out1_1 = _mm256_mul_pd(out1_1, scale);
                    out2_0 = _mm256_mul_pd(out2_0, scale);
                    out2_1 = _mm256_mul_pd(out2_1, scale);
                    out3_0 = _mm256_mul_pd(out3_0, scale);
                    out3_1 = _mm256_mul_pd(out3_1, scale);
                    out4_0 = _mm256_mul_pd(out4_0, scale);
                    out4_1 = _mm256_mul_pd(out4_1, scale);
                    out5_0 = _mm256_mul_pd(out5_0, scale);
                    out5_1 = _mm256_mul_pd(out5_1, scale);
                    out6_0 = _mm256_mul_pd(out6_0, scale);
                    out6_1 = _mm256_mul_pd(out6_1, scale);
                    out7_0 = _mm256_mul_pd(out7_0, scale);
                    out7_1 = _mm256_mul_pd(out7_1, scale);
                }

                _mm256_storeu_pd(ptr, out0_0);
                _mm256_storeu_pd(ptr.add(4), out0_1);
                _mm256_storeu_pd(ptr.add(8), out1_0);
                _mm256_storeu_pd(ptr.add(12), out1_1);
                _mm256_storeu_pd(ptr.add(16), out2_0);
                _mm256_storeu_pd(ptr.add(20), out2_1);
                _mm256_storeu_pd(ptr.add(24), out3_0);
                _mm256_storeu_pd(ptr.add(28), out3_1);
                _mm256_storeu_pd(ptr.add(32), out4_0);
                _mm256_storeu_pd(ptr.add(36), out4_1);
                _mm256_storeu_pd(ptr.add(40), out5_0);
                _mm256_storeu_pd(ptr.add(44), out5_1);
                _mm256_storeu_pd(ptr.add(48), out6_0);
                _mm256_storeu_pd(ptr.add(52), out6_1);
                _mm256_storeu_pd(ptr.add(56), out7_0);
                _mm256_storeu_pd(ptr.add(60), out7_1);
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 32]>();
                crate::application::execution::kernel::components::winograd::dft32_impl::<
                    f64,
                    INVERSE,
                >(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex64::new(1.0 / 32.0, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
            }
        }
        64 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                use std::arch::x86_64::{
                    _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute2f128_pd, _mm256_set1_pd,
                    _mm256_setzero_pd, _mm256_storeu_pd,
                };
                let ptr = data.as_mut_ptr().cast::<f64>();

                let mut c0 = avx_fft8_parallel_precise::<INVERSE>(
                    _mm256_loadu_pd(ptr),
                    _mm256_loadu_pd(ptr.add(16)),
                    _mm256_loadu_pd(ptr.add(32)),
                    _mm256_loadu_pd(ptr.add(48)),
                    _mm256_loadu_pd(ptr.add(64)),
                    _mm256_loadu_pd(ptr.add(80)),
                    _mm256_loadu_pd(ptr.add(96)),
                    _mm256_loadu_pd(ptr.add(112)),
                );

                let mut c1 = avx_fft8_parallel_precise::<INVERSE>(
                    _mm256_loadu_pd(ptr.add(4)),
                    _mm256_loadu_pd(ptr.add(20)),
                    _mm256_loadu_pd(ptr.add(36)),
                    _mm256_loadu_pd(ptr.add(52)),
                    _mm256_loadu_pd(ptr.add(68)),
                    _mm256_loadu_pd(ptr.add(84)),
                    _mm256_loadu_pd(ptr.add(100)),
                    _mm256_loadu_pd(ptr.add(116)),
                );

                let mut c2 = avx_fft8_parallel_precise::<INVERSE>(
                    _mm256_loadu_pd(ptr.add(8)),
                    _mm256_loadu_pd(ptr.add(24)),
                    _mm256_loadu_pd(ptr.add(40)),
                    _mm256_loadu_pd(ptr.add(56)),
                    _mm256_loadu_pd(ptr.add(72)),
                    _mm256_loadu_pd(ptr.add(88)),
                    _mm256_loadu_pd(ptr.add(104)),
                    _mm256_loadu_pd(ptr.add(120)),
                );

                let mut c3 = avx_fft8_parallel_precise::<INVERSE>(
                    _mm256_loadu_pd(ptr.add(12)),
                    _mm256_loadu_pd(ptr.add(28)),
                    _mm256_loadu_pd(ptr.add(44)),
                    _mm256_loadu_pd(ptr.add(60)),
                    _mm256_loadu_pd(ptr.add(76)),
                    _mm256_loadu_pd(ptr.add(92)),
                    _mm256_loadu_pd(ptr.add(108)),
                    _mm256_loadu_pd(ptr.add(124)),
                );

                let tw_table = if INVERSE {
                    &TWIDDLES_COMBINE_INV_64
                } else {
                    &TWIDDLES_COMBINE_FWD_64
                };
                let tw_ptr = tw_table.as_ptr().cast::<f64>();

                // Unrolled twiddle multiplications
                // k = 1
                let base_1 = 16;
                c0[1] = avx_cmul_precise(c0[1], _mm256_loadu_pd(tw_ptr.add(base_1)));
                c1[1] = avx_cmul_precise(c1[1], _mm256_loadu_pd(tw_ptr.add(base_1 + 4)));
                c2[1] = avx_cmul_precise(c2[1], _mm256_loadu_pd(tw_ptr.add(base_1 + 8)));
                c3[1] = avx_cmul_precise(c3[1], _mm256_loadu_pd(tw_ptr.add(base_1 + 12)));

                // k = 2
                let base_2 = 32;
                c0[2] = avx_cmul_precise(c0[2], _mm256_loadu_pd(tw_ptr.add(base_2)));
                c1[2] = avx_cmul_precise(c1[2], _mm256_loadu_pd(tw_ptr.add(base_2 + 4)));
                c2[2] = avx_cmul_precise(c2[2], _mm256_loadu_pd(tw_ptr.add(base_2 + 8)));
                c3[2] = avx_cmul_precise(c3[2], _mm256_loadu_pd(tw_ptr.add(base_2 + 12)));

                // k = 3
                let base_3 = 48;
                c0[3] = avx_cmul_precise(c0[3], _mm256_loadu_pd(tw_ptr.add(base_3)));
                c1[3] = avx_cmul_precise(c1[3], _mm256_loadu_pd(tw_ptr.add(base_3 + 4)));
                c2[3] = avx_cmul_precise(c2[3], _mm256_loadu_pd(tw_ptr.add(base_3 + 8)));
                c3[3] = avx_cmul_precise(c3[3], _mm256_loadu_pd(tw_ptr.add(base_3 + 12)));

                // k = 4
                let base_4 = 64;
                c0[4] = avx_cmul_precise(c0[4], _mm256_loadu_pd(tw_ptr.add(base_4)));
                c1[4] = avx_cmul_precise(c1[4], _mm256_loadu_pd(tw_ptr.add(base_4 + 4)));
                c2[4] = avx_cmul_precise(c2[4], _mm256_loadu_pd(tw_ptr.add(base_4 + 8)));
                c3[4] = avx_cmul_precise(c3[4], _mm256_loadu_pd(tw_ptr.add(base_4 + 12)));

                // k = 5
                let base_5 = 80;
                c0[5] = avx_cmul_precise(c0[5], _mm256_loadu_pd(tw_ptr.add(base_5)));
                c1[5] = avx_cmul_precise(c1[5], _mm256_loadu_pd(tw_ptr.add(base_5 + 4)));
                c2[5] = avx_cmul_precise(c2[5], _mm256_loadu_pd(tw_ptr.add(base_5 + 8)));
                c3[5] = avx_cmul_precise(c3[5], _mm256_loadu_pd(tw_ptr.add(base_5 + 12)));

                // k = 6
                let base_6 = 96;
                c0[6] = avx_cmul_precise(c0[6], _mm256_loadu_pd(tw_ptr.add(base_6)));
                c1[6] = avx_cmul_precise(c1[6], _mm256_loadu_pd(tw_ptr.add(base_6 + 4)));
                c2[6] = avx_cmul_precise(c2[6], _mm256_loadu_pd(tw_ptr.add(base_6 + 8)));
                c3[6] = avx_cmul_precise(c3[6], _mm256_loadu_pd(tw_ptr.add(base_6 + 12)));

                // k = 7
                let base_7 = 112;
                c0[7] = avx_cmul_precise(c0[7], _mm256_loadu_pd(tw_ptr.add(base_7)));
                c1[7] = avx_cmul_precise(c1[7], _mm256_loadu_pd(tw_ptr.add(base_7 + 4)));
                c2[7] = avx_cmul_precise(c2[7], _mm256_loadu_pd(tw_ptr.add(base_7 + 8)));
                c3[7] = avx_cmul_precise(c3[7], _mm256_loadu_pd(tw_ptr.add(base_7 + 12)));

                let mut r = [[_mm256_setzero_pd(); 4]; 8];
                r[0] = avx_fft8_precise::<INVERSE>(c0[0], c1[0], c2[0], c3[0]);
                r[1] = avx_fft8_precise::<INVERSE>(c0[1], c1[1], c2[1], c3[1]);
                r[2] = avx_fft8_precise::<INVERSE>(c0[2], c1[2], c2[2], c3[2]);
                r[3] = avx_fft8_precise::<INVERSE>(c0[3], c1[3], c2[3], c3[3]);
                r[4] = avx_fft8_precise::<INVERSE>(c0[4], c1[4], c2[4], c3[4]);
                r[5] = avx_fft8_precise::<INVERSE>(c0[5], c1[5], c2[5], c3[5]);
                r[6] = avx_fft8_precise::<INVERSE>(c0[6], c1[6], c2[6], c3[6]);
                r[7] = avx_fft8_precise::<INVERSE>(c0[7], c1[7], c2[7], c3[7]);

                let mut out = [_mm256_setzero_pd(); 32];

                // p = 0
                out[0] = _mm256_permute2f128_pd::<0x20>(r[0][0], r[1][0]);
                out[1] = _mm256_permute2f128_pd::<0x20>(r[2][0], r[3][0]);
                out[2] = _mm256_permute2f128_pd::<0x20>(r[4][0], r[5][0]);
                out[3] = _mm256_permute2f128_pd::<0x20>(r[6][0], r[7][0]);
                out[4] = _mm256_permute2f128_pd::<0x31>(r[0][0], r[1][0]);
                out[5] = _mm256_permute2f128_pd::<0x31>(r[2][0], r[3][0]);
                out[6] = _mm256_permute2f128_pd::<0x31>(r[4][0], r[5][0]);
                out[7] = _mm256_permute2f128_pd::<0x31>(r[6][0], r[7][0]);

                // p = 1
                out[8] = _mm256_permute2f128_pd::<0x20>(r[0][1], r[1][1]);
                out[9] = _mm256_permute2f128_pd::<0x20>(r[2][1], r[3][1]);
                out[10] = _mm256_permute2f128_pd::<0x20>(r[4][1], r[5][1]);
                out[11] = _mm256_permute2f128_pd::<0x20>(r[6][1], r[7][1]);
                out[12] = _mm256_permute2f128_pd::<0x31>(r[0][1], r[1][1]);
                out[13] = _mm256_permute2f128_pd::<0x31>(r[2][1], r[3][1]);
                out[14] = _mm256_permute2f128_pd::<0x31>(r[4][1], r[5][1]);
                out[15] = _mm256_permute2f128_pd::<0x31>(r[6][1], r[7][1]);

                // p = 2
                out[16] = _mm256_permute2f128_pd::<0x20>(r[0][2], r[1][2]);
                out[17] = _mm256_permute2f128_pd::<0x20>(r[2][2], r[3][2]);
                out[18] = _mm256_permute2f128_pd::<0x20>(r[4][2], r[5][2]);
                out[19] = _mm256_permute2f128_pd::<0x20>(r[6][2], r[7][2]);
                out[20] = _mm256_permute2f128_pd::<0x31>(r[0][2], r[1][2]);
                out[21] = _mm256_permute2f128_pd::<0x31>(r[2][2], r[3][2]);
                out[22] = _mm256_permute2f128_pd::<0x31>(r[4][2], r[5][2]);
                out[23] = _mm256_permute2f128_pd::<0x31>(r[6][2], r[7][2]);

                // p = 3
                out[24] = _mm256_permute2f128_pd::<0x20>(r[0][3], r[1][3]);
                out[25] = _mm256_permute2f128_pd::<0x20>(r[2][3], r[3][3]);
                out[26] = _mm256_permute2f128_pd::<0x20>(r[4][3], r[5][3]);
                out[27] = _mm256_permute2f128_pd::<0x20>(r[6][3], r[7][3]);
                out[28] = _mm256_permute2f128_pd::<0x31>(r[0][3], r[1][3]);
                out[29] = _mm256_permute2f128_pd::<0x31>(r[2][3], r[3][3]);
                out[30] = _mm256_permute2f128_pd::<0x31>(r[4][3], r[5][3]);
                out[31] = _mm256_permute2f128_pd::<0x31>(r[6][3], r[7][3]);

                if INVERSE && NORMALIZE {
                    let scale = _mm256_set1_pd(1.0 / 64.0);
                    for i in 0..32 {
                        out[i] = _mm256_mul_pd(out[i], scale);
                    }
                }

                for i in 0..32 {
                    _mm256_storeu_pd(ptr.add(i * 4), out[i]);
                }
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 64]>();
                crate::application::execution::kernel::components::winograd::dft64_impl::<
                    f64,
                    INVERSE,
                >(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex64::new(1.0 / 64.0, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}
