//! Unrolled small power-of-two codelets at precise lane density.
//!
//! Extracted verbatim from the `MixedRadixScalar` implementation so the trait
//! wiring and the unrolled codelet bodies occupy separate leaf modules.

#[cfg(target_arch = "x86_64")]
use super::super::simd::avx::{
    avx_cmul_precise, avx_fft4_parallel_precise, avx_fft8_parallel_precise, avx_fft8_precise,
};
#[cfg(target_arch = "x86_64")]
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
            // Scalar by measurement: n = 8: vector arm measured +12% (call plus probe outweigh the body); kept scalar.
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
        16 => {
            // Scalar by measurement: n = 16: vector arm measured +8% (call plus probe outweigh the body); kept scalar.
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 16]>();
            crate::application::execution::kernel::components::winograd::dft16_impl::<f64, INVERSE>(
                data_ref,
            );
            if INVERSE && NORMALIZE {
                let scale = Complex64::new(0.0625, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
        }
        32 => {
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx,fma")]
            #[inline]
            unsafe fn vector_arm<const INVERSE: bool, const NORMALIZE: bool>(
                data: &mut [Complex64],
            ) {
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
            let vector_done = {
                #[cfg(target_arch = "x86_64")]
                {
                    if super::super::simd::avx::avx_fma_available() {
                        // SAFETY: the probe proved AVX and FMA on this host, and the
                        // caller's contract supplies at least this arm's lanes.
                        vector_arm::<INVERSE, NORMALIZE>(data);
                        true
                    } else {
                        false
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            };
            if !vector_done {
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
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx,fma")]
            #[inline]
            unsafe fn vector_arm<const INVERSE: bool, const NORMALIZE: bool>(
                data: &mut [Complex64],
            ) {
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
            let vector_done = {
                #[cfg(target_arch = "x86_64")]
                {
                    if super::super::simd::avx::avx_fma_available() {
                        // SAFETY: the probe proved AVX and FMA on this host, and the
                        // caller's contract supplies at least this arm's lanes.
                        vector_arm::<INVERSE, NORMALIZE>(data);
                        true
                    } else {
                        false
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            };
            if !vector_done {
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
