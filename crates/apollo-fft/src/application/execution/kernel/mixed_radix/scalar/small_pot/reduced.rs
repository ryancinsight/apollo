//! Unrolled small power-of-two codelets at reduced lane density.
//!
//! Extracted verbatim from the `MixedRadixScalar` implementation so the trait
//! wiring and the unrolled codelet bodies occupy separate leaf modules.

#[cfg(target_arch = "x86_64")]
use super::super::trait_def::MixedRadixScalar;
#[cfg(target_arch = "x86_64")]
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
            // Scalar by measurement: n = 2: vector arm measured neutral; a call is not worth it here.
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
            // Scalar by measurement: n = 4: vector arm measured neutral (-1%); a call is not worth it here.
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
            small_pot_inplace_sized_reduced::<16, INVERSE, NORMALIZE>(data);
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
            // Scalar by measurement: sized n = 4: vector arm measured neutral (-1%); a call is not worth it here.
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
        8 => {
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx,fma")]
            #[inline]
            unsafe fn vector_arm<const INVERSE: bool, const NORMALIZE: bool>(
                data: &mut [Complex32],
            ) {
                // One 256-bit register holds four complexes, so the eight
                // samples are two registers: `u = lo + hi` and
                // `v = (lo - hi) W_8^n` are the even and odd four-point
                // transforms' inputs, and each four-point transform runs on
                // both at once through the pair layout `[a_0 b_0 | a_2 b_2]`
                // (`unpack`), which lands the outputs contiguous:
                // `P = [t_0 t_0' | r_0 r_0']`, `Q = [t_1 t_1' | -+ i r_1,
                // -+ i r_1']`, `X[0..4] = P + Q`, `X[4..8] = P - Q`.
                // The SSE form this replaces ran four 128-bit registers and
                // read no faster than the eight-byte scalar codelet.
                use std::arch::x86_64::{
                    _mm256_add_ps, _mm256_castpd_ps, _mm256_castps_pd, _mm256_fmaddsub_ps,
                    _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute2f128_ps, _mm256_permute_ps,
                    _mm256_set1_ps, _mm256_setr_ps, _mm256_storeu_ps, _mm256_sub_ps,
                    _mm256_unpackhi_pd, _mm256_unpacklo_pd, _mm256_xor_ps,
                };
                let ptr = data.as_mut_ptr().cast::<f32>();
                let lo = _mm256_loadu_ps(ptr);
                let hi = _mm256_loadu_ps(ptr.add(8));
                let u = _mm256_add_ps(lo, hi);
                let d = _mm256_sub_ps(lo, hi);
                // `d` times `[W_8^0, W_8^1, W_8^2, W_8^3]`, the twiddles as
                // real and imaginary parts duplicated over each sample's
                // two lanes; one swap, one multiply, one fused multiply-add
                // with alternating signs.
                let s = core::f32::consts::FRAC_1_SQRT_2;
                let w_re = _mm256_setr_ps(1.0, 1.0, s, s, 0.0, 0.0, -s, -s);
                let w_im = if INVERSE {
                    _mm256_setr_ps(0.0, 0.0, s, s, 1.0, 1.0, s, s)
                } else {
                    _mm256_setr_ps(0.0, 0.0, -s, -s, -1.0, -1.0, -s, -s)
                };
                let d_swapped = _mm256_permute_ps(d, 0b1011_0001);
                let v = _mm256_fmaddsub_ps(d, w_re, _mm256_mul_ps(d_swapped, w_im));
                // The pair layout: `g = [u_0 v_0 | u_2 v_2]`, `h = [u_1 v_1 |
                // u_3 v_3]`, and their half swaps.
                let g =
                    _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(u), _mm256_castps_pd(v)));
                let h =
                    _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(u), _mm256_castps_pd(v)));
                let g_swapped = _mm256_permute2f128_ps(g, g, 0x01);
                let h_swapped = _mm256_permute2f128_ps(h, h, 0x01);
                let tg = _mm256_add_ps(g, g_swapped);
                let rg = _mm256_sub_ps(g, g_swapped);
                let th = _mm256_add_ps(h, h_swapped);
                let rh = _mm256_sub_ps(h, h_swapped);
                // `-+ i rh`: swap real and imaginary, negate the imaginary
                // lanes forward and the real lanes inverse.
                let rh_swapped = _mm256_permute_ps(rh, 0b1011_0001);
                let sign = if INVERSE {
                    _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0)
                } else {
                    _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0)
                };
                let rh_turned = _mm256_xor_ps(rh_swapped, sign);
                let p = _mm256_permute2f128_ps(tg, rg, 0x20);
                let q = _mm256_permute2f128_ps(th, rh_turned, 0x20);
                let mut out_lo = _mm256_add_ps(p, q);
                let mut out_hi = _mm256_sub_ps(p, q);
                if INVERSE && NORMALIZE {
                    let scale = _mm256_set1_ps(0.125);
                    out_lo = _mm256_mul_ps(out_lo, scale);
                    out_hi = _mm256_mul_ps(out_hi, scale);
                }
                _mm256_storeu_ps(ptr, out_lo);
                _mm256_storeu_ps(ptr.add(8), out_hi);
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
            // The eight-lane register kernel when the host has that width, the
            // Winograd codelet otherwise; both leave natural order in place.
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 16]>();
            if !crate::application::execution::kernel::components::winograd::composite::try_dft16_hardware::<INVERSE>(data_ref) {
                crate::application::execution::kernel::components::winograd::dft16_impl::<f32, INVERSE>(
                    data_ref,
                );
            }
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(0.0625, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
        }
        32 => {
            // Scalar by measurement: n = 32: the vector body measured 5x slower than the scalar arm (240 vs 47 ns pinned); kept scalar, kernel filed.
            let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 32]>();
            if !crate::application::execution::kernel::components::winograd::composite::try_dft32_hardware::<INVERSE>(data_ref) {
                crate::application::execution::kernel::components::winograd::dft32_impl::<f32, INVERSE>(
                    data_ref,
                );
            }
            if INVERSE && NORMALIZE {
                let scale = Complex32::new(1.0 / 32.0, 0.0);
                for x in data_ref.iter_mut() {
                    *x *= scale;
                }
            }
        }
        64 => {
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx,fma")]
            #[inline]
            unsafe fn vector_arm<const INVERSE: bool, const NORMALIZE: bool>(
                data: &mut [Complex32],
            ) {
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
