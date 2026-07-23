//! Unrolled small power-of-two codelets at reduced lane density.
//!
//! Extracted verbatim from the `MixedRadixScalar` implementation so the trait
//! wiring and the unrolled codelet bodies occupy separate leaf modules.

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use super::super::simd::avx::{avx_fft4_reduced, rotate_minus_i_ps, rotate_plus_i_ps, sse_cmul_ps};
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use super::super::trait_def::MixedRadixScalar;
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use crate::application::execution::kernel::radix_stage::normalize_inplace;
use eunomia::Complex32;

/// Applies an unrolled codelet when `data.len()` is a supported size.
///
/// # Safety
///
/// Carries the `MixedRadixScalar::small_pot_inplace` contract unchanged.
#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn small_pot_inplace_reduced<
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [Complex32],
) -> bool {
    let n = data.len();
    match n {
        2 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                use std::arch::x86_64::{
                    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_shuffle_ps,
                    _mm_storeu_ps, _mm_sub_ps,
                };
                let ptr = data.as_mut_ptr().cast::<f32>();
                let reg = _mm_loadu_ps(ptr);
                let a_reg = _mm_shuffle_ps::<0x44>(reg, reg);
                let b_reg = _mm_shuffle_ps::<0xEE>(reg, reg);
                let sum = _mm_add_ps(a_reg, b_reg);
                let diff = _mm_sub_ps(a_reg, b_reg);
                let mut res = _mm_shuffle_ps::<0xE4>(sum, diff);
                if INVERSE && NORMALIZE {
                    let scale = _mm_set1_ps(0.5);
                    res = _mm_mul_ps(res, scale);
                }
                _mm_storeu_ps(ptr, res);
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let a = *data.get_unchecked(0);
                let b = *data.get_unchecked(1);
                if INVERSE && NORMALIZE {
                    let half = Complex32::new(0.5, 0.0);
                    *data.get_unchecked_mut(0) = (a + b) * half;
                    *data.get_unchecked_mut(1) = (a - b) * half;
                } else {
                    *data.get_unchecked_mut(0) = a + b;
                    *data.get_unchecked_mut(1) = a - b;
                }
            }
            true
        }
        3 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 3]>();
            <f32 as crate::application::execution::kernel::mixed_radix::traits::ShortDft<3>>::dft::<
                INVERSE,
            >(data_ref);
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(1.0 / 3.0, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        4 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                use std::arch::x86_64::{
                    _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
                };
                let ptr = data.as_mut_ptr().cast::<f32>();
                let v = _mm256_loadu_ps(ptr);
                let mut res = avx_fft4_reduced::<INVERSE>(v);
                if NORMALIZE {
                    let scale = _mm256_set1_ps(0.25);
                    res = _mm256_mul_ps(res, scale);
                }
                _mm256_storeu_ps(ptr, res);
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let x0 = *data.get_unchecked(0);
                let x1 = *data.get_unchecked(1);
                let x2 = *data.get_unchecked(2);
                let x3 = *data.get_unchecked(3);
                let a0 = x0 + x2;
                let a1 = x1 + x3;
                let a2 = x0 - x2;
                let a3 = x1 - x3;
                let i_a3 = Complex32::new(-a3.im, a3.re);
                if INVERSE && NORMALIZE {
                    let quarter = Complex32::new(0.25, 0.0);
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
            true
        }
        5 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 5]>();
            <f32 as crate::application::execution::kernel::mixed_radix::traits::ShortDft<5>>::dft::<
                INVERSE,
            >(data_ref);
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(0.2, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        6 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 6]>();
            crate::application::execution::kernel::components::winograd::dft6_impl::<f32, INVERSE>(
                data_ref,
            );
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(1.0 / 6.0, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        7 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 7]>();
            <f32 as crate::application::execution::kernel::mixed_radix::traits::ShortDft<7>>::dft::<
                INVERSE,
            >(data_ref);
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(1.0 / 7.0, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        8 => {
            let x0 = *data.get_unchecked(0);
            let x1 = *data.get_unchecked(1);
            let x2 = *data.get_unchecked(2);
            let x3 = *data.get_unchecked(3);
            let x4 = *data.get_unchecked(4);
            let x5 = *data.get_unchecked(5);
            let x6 = *data.get_unchecked(6);
            let x7 = *data.get_unchecked(7);
            let c = std::f32::consts::FRAC_1_SQRT_2;
            let diff15 = x1 - x5;
            let a5 = if INVERSE {
                Complex32::new((diff15.re - diff15.im) * c, (diff15.re + diff15.im) * c)
            } else {
                Complex32::new((diff15.re + diff15.im) * c, (diff15.im - diff15.re) * c)
            };
            let diff37 = x3 - x7;
            let a7 = if INVERSE {
                Complex32::new(-(diff37.re + diff37.im) * c, (diff37.re - diff37.im) * c)
            } else {
                Complex32::new((diff37.im - diff37.re) * c, -(diff37.re + diff37.im) * c)
            };
            let a0 = x0 + x4;
            let a1 = x1 + x5;
            let a2 = x2 + x6;
            let a3 = x3 + x7;
            let a4 = x0 - x4;
            let diff26 = x2 - x6;
            let a6 = if INVERSE {
                Complex32::new(-diff26.im, diff26.re)
            } else {
                Complex32::new(diff26.im, -diff26.re)
            };
            let b0 = a0 + a2;
            let b1 = a1 + a3;
            let b2 = a0 - a2;
            let tmp3 = a1 - a3;
            let b3 = if INVERSE {
                Complex32::new(-tmp3.im, tmp3.re)
            } else {
                Complex32::new(tmp3.im, -tmp3.re)
            };
            let b4 = a4 + a6;
            let b5 = a5 + a7;
            let b6 = a4 - a6;
            let tmp7 = a5 - a7;
            let b7 = if INVERSE {
                Complex32::new(-tmp7.im, tmp7.re)
            } else {
                Complex32::new(tmp7.im, -tmp7.re)
            };
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(0.125, 0.0);
                *data.get_unchecked_mut(0) = (b0 + b1) * scale;
                *data.get_unchecked_mut(4) = (b0 - b1) * scale;
                *data.get_unchecked_mut(2) = (b2 + b3) * scale;
                *data.get_unchecked_mut(6) = (b2 - b3) * scale;
                *data.get_unchecked_mut(1) = (b4 + b5) * scale;
                *data.get_unchecked_mut(5) = (b4 - b5) * scale;
                *data.get_unchecked_mut(3) = (b6 + b7) * scale;
                *data.get_unchecked_mut(7) = (b6 - b7) * scale;
            } else {
                *data.get_unchecked_mut(0) = b0 + b1;
                *data.get_unchecked_mut(4) = b0 - b1;
                *data.get_unchecked_mut(2) = b2 + b3;
                *data.get_unchecked_mut(6) = b2 - b3;
                *data.get_unchecked_mut(1) = b4 + b5;
                *data.get_unchecked_mut(5) = b4 - b5;
                *data.get_unchecked_mut(3) = b6 + b7;
                *data.get_unchecked_mut(7) = b6 - b7;
            }
            true
        }
        16 => {
            // Use winograd dft16 for correctness (avx2_dft16_reduced produced wrong spectra
            // for f32 plans; winograd verified).
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 16]>();
            crate::application::execution::kernel::components::winograd::dft16_impl::<f32, INVERSE>(
                data_ref,
            );
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(0.0625, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
            true
        }
        32 => {
            small_pot_inplace_sized_reduced::<32, INVERSE, NORMALIZE>(data);
            true
        }
        64 => {
            small_pot_inplace_sized_reduced::<64, INVERSE, NORMALIZE>(data);
            true
        }
        9 => {
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 9]>();
            crate::application::execution::kernel::components::winograd::dft9_impl::<f32, INVERSE>(
                data_ref,
            );
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(1.0 / 9.0, 0.0);
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
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn small_pot_inplace_sized_reduced<
    const N: usize,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [Complex32],
) {
    match N {
        2 => {
            let a = *data.get_unchecked(0);
            let b = *data.get_unchecked(1);
            if INVERSE && NORMALIZE {
                let half = Complex32::new(0.5, 0.0);
                *data.get_unchecked_mut(0) = (a + b) * half;
                *data.get_unchecked_mut(1) = (a - b) * half;
            } else {
                *data.get_unchecked_mut(0) = a + b;
                *data.get_unchecked_mut(1) = a - b;
            }
        }
        4 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                use std::arch::x86_64::{
                    _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
                };
                let ptr = data.as_mut_ptr().cast::<f32>();
                let v = _mm256_loadu_ps(ptr);
                let mut res = avx_fft4_reduced::<INVERSE>(v);
                if NORMALIZE {
                    let scale = _mm256_set1_ps(0.25);
                    res = _mm256_mul_ps(res, scale);
                }
                _mm256_storeu_ps(ptr, res);
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let x0 = *data.get_unchecked(0);
                let x1 = *data.get_unchecked(1);
                let x2 = *data.get_unchecked(2);
                let x3 = *data.get_unchecked(3);
                let a0 = x0 + x2;
                let a1 = x1 + x3;
                let a2 = x0 - x2;
                let a3 = x1 - x3;
                let i_a3 = Complex32::new(-a3.im, a3.re);
                if INVERSE && NORMALIZE {
                    let quarter = Complex32::new(0.25, 0.0);
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
        }
        8 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                use std::arch::x86_64::{
                    _mm_add_ps, _mm_castpd_ps, _mm_castps_pd, _mm_loadu_ps, _mm_mul_ps,
                    _mm_set1_ps, _mm_setr_ps, _mm_shuffle_pd, _mm_shuffle_ps, _mm_storeu_ps,
                    _mm_sub_ps,
                };
                let ptr = data.as_mut_ptr().cast::<f32>();
                let d01 = _mm_loadu_ps(ptr);
                let d23 = _mm_loadu_ps(ptr.add(4));
                let d45 = _mm_loadu_ps(ptr.add(8));
                let d67 = _mm_loadu_ps(ptr.add(12));

                let d01_d = _mm_castps_pd(d01);
                let d23_d = _mm_castps_pd(d23);
                let d45_d = _mm_castps_pd(d45);
                let d67_d = _mm_castps_pd(d67);

                // Bit-reversed loading into SSE registers
                let r0 = _mm_castpd_ps(_mm_shuffle_pd(d01_d, d45_d, 0)); // holds [x[0], x[4]]
                let r1 = _mm_castpd_ps(_mm_shuffle_pd(d23_d, d67_d, 0)); // holds [x[2], x[6]]
                let r2 = _mm_castpd_ps(_mm_shuffle_pd(d01_d, d45_d, 3)); // holds [x[1], x[5]]
                let r3 = _mm_castpd_ps(_mm_shuffle_pd(d23_d, d67_d, 3)); // holds [x[3], x[7]]

                // Stage 1: Size-2 butterflies within registers
                let a_reg0 = _mm_shuffle_ps(r0, r0, 0x44);
                let b_reg0 = _mm_shuffle_ps(r0, r0, 0xEE);
                let sum0 = _mm_add_ps(a_reg0, b_reg0);
                let diff0 = _mm_sub_ps(a_reg0, b_reg0);
                let s1_0 = _mm_shuffle_ps(sum0, diff0, 0xE4); // holds [x[0]+x[4], x[0]-x[4]] = [a0, a1]

                let a_reg1 = _mm_shuffle_ps(r1, r1, 0x44);
                let b_reg1 = _mm_shuffle_ps(r1, r1, 0xEE);
                let sum1 = _mm_add_ps(a_reg1, b_reg1);
                let diff1 = _mm_sub_ps(a_reg1, b_reg1);
                let s1_1 = _mm_shuffle_ps(sum1, diff1, 0xE4); // holds [x[2]+x[6], x[2]-x[6]] = [a2, a3]

                let a_reg2 = _mm_shuffle_ps(r2, r2, 0x44);
                let b_reg2 = _mm_shuffle_ps(r2, r2, 0xEE);
                let sum2 = _mm_add_ps(a_reg2, b_reg2);
                let diff2 = _mm_sub_ps(a_reg2, b_reg2);
                let s1_2 = _mm_shuffle_ps(sum2, diff2, 0xE4); // holds [x[1]+x[5], x[1]-x[5]] = [a4, a5]

                let a_reg3 = _mm_shuffle_ps(r3, r3, 0x44);
                let b_reg3 = _mm_shuffle_ps(r3, r3, 0xEE);
                let sum3 = _mm_add_ps(a_reg3, b_reg3);
                let diff3 = _mm_sub_ps(a_reg3, b_reg3);
                let s1_3 = _mm_shuffle_ps(sum3, diff3, 0xE4); // holds [x[3]+x[7], x[3]-x[7]] = [a6, a7]

                // Stage 2: Size-4 butterflies between s1_0, s1_1 and s1_2, s1_3
                let s1_1_tw = if INVERSE {
                    rotate_plus_i_ps(s1_1)
                } else {
                    rotate_minus_i_ps(s1_1)
                };
                let s1_1_mixed = _mm_castpd_ps(_mm_shuffle_pd(
                    _mm_castps_pd(s1_1),
                    _mm_castps_pd(s1_1_tw),
                    2,
                )); // holds [a2, a3_rot]

                let s1_3_tw = if INVERSE {
                    rotate_plus_i_ps(s1_3)
                } else {
                    rotate_minus_i_ps(s1_3)
                };
                let s1_3_mixed = _mm_castpd_ps(_mm_shuffle_pd(
                    _mm_castps_pd(s1_3),
                    _mm_castps_pd(s1_3_tw),
                    2,
                )); // holds [a6, a7_rot]

                let b0_1 = _mm_add_ps(s1_0, s1_1_mixed); // holds [b0, b1]
                let b2_3 = _mm_sub_ps(s1_0, s1_1_mixed); // holds [b2, b3]
                let b4_5 = _mm_add_ps(s1_2, s1_3_mixed); // holds [b4, b5]
                let b6_7 = _mm_sub_ps(s1_2, s1_3_mixed); // holds [b6, b7]

                // Stage 3: Size-8 butterflies
                let c = std::f32::consts::FRAC_1_SQRT_2;
                let w0_1 = if INVERSE {
                    _mm_setr_ps(1.0, 0.0, c, c)
                } else {
                    _mm_setr_ps(1.0, 0.0, c, -c)
                };
                let w2_3 = if INVERSE {
                    _mm_setr_ps(0.0, 1.0, -c, c)
                } else {
                    _mm_setr_ps(0.0, -1.0, -c, -c)
                };

                let b4_5_tw = sse_cmul_ps(b4_5, w0_1);
                let b6_7_tw = sse_cmul_ps(b6_7, w2_3);

                let mut out0_1 = _mm_add_ps(b0_1, b4_5_tw);
                let mut out4_5 = _mm_sub_ps(b0_1, b4_5_tw);
                let mut out2_3 = _mm_add_ps(b2_3, b6_7_tw);
                let mut out6_7 = _mm_sub_ps(b2_3, b6_7_tw);

                if INVERSE && NORMALIZE {
                    let scale = _mm_set1_ps(0.125);
                    out0_1 = _mm_mul_ps(out0_1, scale);
                    out2_3 = _mm_mul_ps(out2_3, scale);
                    out4_5 = _mm_mul_ps(out4_5, scale);
                    out6_7 = _mm_mul_ps(out6_7, scale);
                }

                _mm_storeu_ps(ptr, out0_1);
                _mm_storeu_ps(ptr.add(4), out2_3);
                _mm_storeu_ps(ptr.add(8), out4_5);
                _mm_storeu_ps(ptr.add(12), out6_7);
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 8]>();
                crate::application::execution::kernel::components::winograd::dft8_array_impl::<
                    f32,
                    INVERSE,
                    false,
                >(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex32::new(0.125, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
            }
        }
        16 => {
            // Use winograd dft16 for correctness (avx2_dft16_reduced path produced wrong spectra
            // for f32 n=16 plans; winograd dft16 verified in dft_small and other paths).
            // TODO: repair avx2_dft16_reduced or replace with correct column avx for 16.
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 16]>();
            crate::application::execution::kernel::components::winograd::dft16_impl::<f32, INVERSE>(
                data_ref,
            );
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(0.0625, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
        }
        32 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                let mut scratch = [Complex32::new(0.0, 0.0); 32];
                let twiddles = <f32 as MixedRadixScalar>::small_pot_twiddles::<INVERSE>(32);
                <f32 as crate::application::execution::kernel::components::stockham::StockhamKernel>::forward_with_scratch(
                    data,
                    &mut scratch,
                    twiddles,
                );
                if INVERSE && NORMALIZE {
                    normalize_inplace(data, 1.0 / 32.0);
                }
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 32]>();
                crate::application::execution::kernel::components::winograd::dft32_impl::<
                    f32,
                    INVERSE,
                >(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex32::new(1.0 / 32.0, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
            }
        }
        64 => {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                let mut scratch = [Complex32::new(0.0, 0.0); 64];
                let twiddles = <f32 as MixedRadixScalar>::small_pot_twiddles::<INVERSE>(64);
                <f32 as crate::application::execution::kernel::components::stockham::StockhamKernel>::forward_with_scratch(
                    data,
                    &mut scratch,
                    twiddles,
                );
                if INVERSE && NORMALIZE {
                    normalize_inplace(data, 1.0 / 64.0);
                }
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 64]>();
                crate::application::execution::kernel::components::winograd::dft64_impl::<
                    f32,
                    INVERSE,
                >(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex32::new(1.0 / 64.0, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}
