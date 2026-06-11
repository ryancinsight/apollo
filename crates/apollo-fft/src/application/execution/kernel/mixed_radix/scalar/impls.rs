//! Concrete `MixedRadixScalar` implementations for the two production
//! floating-point precisions.
//!
//! The two impls are kept together rather than split per type because they
//! perform identical trait wiring; the only differences are the concrete
//! complex element type and the precision-specific SIMD/transpose routines
//! (`pointwise_mul_reduced`/`_precise`, `transpose_matrix_reduced`/`_precise`).
//! The precision-tagged names refer to SIMD lane density, not to the type
//! suffix, so they remain compliant with the naming policy in CLAUDE.md.

use super::rader::{
    build_rader_negacyclic_spectra, build_rader_negacyclic_twiddles, build_rader_spectrum_vec,
};
use super::simd::{pointwise_mul_precise, pointwise_mul_reduced};
use super::trait_def::MixedRadixScalar;
use super::transpose::{transpose_matrix_precise, transpose_matrix_reduced};
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use super::twiddle_constants::{
    TWIDDLES_COMBINE_FWD_32, TWIDDLES_COMBINE_FWD_64, TWIDDLES_COMBINE_INV_32,
    TWIDDLES_COMBINE_INV_64,
};
use super::twiddle_constants::{
    TWIDDLES_FWD_PRECISE, TWIDDLES_FWD_REDUCED, TWIDDLES_INV_PRECISE, TWIDDLES_INV_REDUCED,
};
use crate::application::execution::kernel::components::{radix_composite, stockham};
use crate::application::execution::kernel::mixed_radix::caches::{
    cached_four_step_twiddles, cached_rader_neg_twiddles, cached_rader_negacyclic_spectra,
    cached_rader_spectrum, cached_twiddle_fwd, cached_twiddle_inv, with_bluestein_scratch,
    with_pfa_scratch, with_rader_padded_scratch, with_stockham_scratch,
};
use crate::application::execution::kernel::pot::{PoTStrategy, SizedPoT};
use crate::application::execution::kernel::radix_stage::normalize_inplace;
use num_complex::{Complex32, Complex64};
use std::sync::Arc;

// ── AVX/SIMD helpers (shared by f32 and f64 impls) ─────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn avx_fft4_reduced<const INVERSE: bool>(
    v: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_permute2f128_ps, _mm256_permute_ps, _mm256_set_ps,
        _mm256_shuffle_ps, _mm256_sub_ps,
    };
    let v_low = _mm256_permute2f128_ps::<0x00>(v, v);
    let v_high = _mm256_permute2f128_ps::<0x11>(v, v);

    let sum = _mm256_add_ps(v_low, v_high);
    let diff = _mm256_sub_ps(v_low, v_high);

    let shuf_a0 = _mm256_shuffle_ps::<0b01000100>(sum, sum);
    let shuf_a1 = _mm256_shuffle_ps::<0b11101110>(sum, sum);

    let add_res = _mm256_add_ps(shuf_a0, shuf_a1);
    let sub_res = _mm256_sub_ps(shuf_a0, shuf_a1);

    let out0_out2 = _mm256_shuffle_ps::<0b11100100>(add_res, sub_res);

    let shuf_a2 = _mm256_shuffle_ps::<0b01000100>(diff, diff);
    let perm_diff = _mm256_permute_ps::<0b10110001>(diff);
    let shuf_a3 = _mm256_shuffle_ps::<0b11101110>(perm_diff, perm_diff);

    let sign_const = if INVERSE {
        _mm256_set_ps(-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0)
    } else {
        _mm256_set_ps(1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0)
    };
    let out1_out3 = _mm256_fmadd_ps(shuf_a3, sign_const, shuf_a2);

    let low_lane = _mm256_shuffle_ps::<0b01000100>(out0_out2, out1_out3);
    let high_lane = _mm256_shuffle_ps::<0b11101110>(out0_out2, out1_out3);
    _mm256_permute2f128_ps::<0x20>(low_lane, high_lane)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn avx_cmul_precise(
    a: std::arch::x86_64::__m256d,
    tw: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_movedup_pd, _mm256_mul_pd, _mm256_permute_pd,
    };
    let re_a = _mm256_movedup_pd(a);
    let im_a_shuf = _mm256_permute_pd::<0x0F>(a);
    let tw_shuf = _mm256_permute_pd::<0x05>(tw);
    let prod2 = _mm256_mul_pd(im_a_shuf, tw_shuf);
    _mm256_fmaddsub_pd(re_a, tw, prod2)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn avx_fft4_precise<const INVERSE: bool>(
    v0: std::arch::x86_64::__m256d,
    v1: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 2] {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_fmadd_pd, _mm256_permute2f128_pd, _mm256_permute_pd, _mm256_set_pd,
        _mm256_sub_pd,
    };
    let sum = _mm256_add_pd(v0, v1);
    let diff = _mm256_sub_pd(v0, v1);

    let shuf_a0 = _mm256_permute2f128_pd::<0x00>(sum, sum);
    let shuf_a1 = _mm256_permute2f128_pd::<0x11>(sum, sum);

    let add_a01 = _mm256_add_pd(shuf_a0, shuf_a1);
    let sub_a01 = _mm256_sub_pd(shuf_a0, shuf_a1);

    let out0_out2 = _mm256_permute2f128_pd::<0x20>(add_a01, sub_a01);

    let shuf_a2 = _mm256_permute2f128_pd::<0x00>(diff, diff);
    let perm_diff = _mm256_permute_pd::<0x05>(diff);
    let shuf_a3 = _mm256_permute2f128_pd::<0x11>(perm_diff, perm_diff);

    let sign_const = if INVERSE {
        _mm256_set_pd(-1.0, 1.0, 1.0, -1.0)
    } else {
        _mm256_set_pd(1.0, -1.0, -1.0, 1.0)
    };
    let out1_out3 = _mm256_fmadd_pd(shuf_a3, sign_const, shuf_a2);

    let reg0 = _mm256_permute2f128_pd::<0x20>(out0_out2, out1_out3);
    let reg1 = _mm256_permute2f128_pd::<0x31>(out0_out2, out1_out3);
    [reg0, reg1]
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn sse_cmul_ps(
    a: std::arch::x86_64::__m128,
    b: std::arch::x86_64::__m128,
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::{
        _mm_addsub_ps, _mm_movehdup_ps, _mm_moveldup_ps, _mm_mul_ps, _mm_shuffle_ps,
    };
    let re_a = _mm_moveldup_ps(a);
    let im_a = _mm_movehdup_ps(a);
    let b_shuf = _mm_shuffle_ps(b, b, 0xB1);
    let prod1 = _mm_mul_ps(re_a, b);
    let prod2 = _mm_mul_ps(im_a, b_shuf);
    _mm_addsub_ps(prod1, prod2)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn rotate_minus_i_ps(v: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::{_mm_setr_ps, _mm_shuffle_ps, _mm_xor_ps};
    let perm = _mm_shuffle_ps(v, v, 0xB1);
    _mm_xor_ps(perm, _mm_setr_ps(0.0, -0.0, 0.0, -0.0))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn rotate_plus_i_ps(v: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::{_mm_setr_ps, _mm_shuffle_ps, _mm_xor_ps};
    let perm = _mm_shuffle_ps(v, v, 0xB1);
    _mm_xor_ps(perm, _mm_setr_ps(-0.0, 0.0, -0.0, 0.0))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn avx_fft4_parallel_precise<const INVERSE: bool>(
    r0: std::arch::x86_64::__m256d,
    r1: std::arch::x86_64::__m256d,
    r2: std::arch::x86_64::__m256d,
    r3: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 4] {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set_pd, _mm256_sub_pd,
    };
    let a = _mm256_add_pd(r0, r2);
    let b = _mm256_add_pd(r1, r3);
    let c = _mm256_sub_pd(r0, r2);
    let d = _mm256_sub_pd(r1, r3);

    let out0 = _mm256_add_pd(a, b);
    let out2 = _mm256_sub_pd(a, b);

    let d_shuf = _mm256_permute_pd::<0x05>(d);
    let sign = _mm256_set_pd(-1.0, 1.0, -1.0, 1.0);
    let v = _mm256_mul_pd(d_shuf, sign);

    let (out1, out3) = if INVERSE {
        (_mm256_sub_pd(c, v), _mm256_add_pd(c, v))
    } else {
        (_mm256_add_pd(c, v), _mm256_sub_pd(c, v))
    };
    [out0, out1, out2, out3]
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn avx_fft8_parallel_precise<const INVERSE: bool>(
    r0: std::arch::x86_64::__m256d,
    r1: std::arch::x86_64::__m256d,
    r2: std::arch::x86_64::__m256d,
    r3: std::arch::x86_64::__m256d,
    r4: std::arch::x86_64::__m256d,
    r5: std::arch::x86_64::__m256d,
    r6: std::arch::x86_64::__m256d,
    r7: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 8] {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set_pd, _mm256_sub_pd,
    };
    let [e0, e1, e2, e3] = avx_fft4_parallel_precise::<INVERSE>(r0, r2, r4, r6);
    let [o0, o1, o2, o3] = avx_fft4_parallel_precise::<INVERSE>(r1, r3, r5, r7);

    let c = std::f64::consts::FRAC_1_SQRT_2;
    let tw1 = if INVERSE {
        _mm256_set_pd(c, c, c, c)
    } else {
        _mm256_set_pd(-c, c, -c, c)
    };
    let tw3 = if INVERSE {
        _mm256_set_pd(c, -c, c, -c)
    } else {
        _mm256_set_pd(-c, -c, -c, -c)
    };

    let t0 = o0;
    let t1 = avx_cmul_precise(o1, tw1);

    let o2_shuf = _mm256_permute_pd::<0x05>(o2);
    let sign_t2 = if INVERSE {
        _mm256_set_pd(1.0, -1.0, 1.0, -1.0)
    } else {
        _mm256_set_pd(-1.0, 1.0, -1.0, 1.0)
    };
    let t2 = _mm256_mul_pd(o2_shuf, sign_t2);

    let t3 = avx_cmul_precise(o3, tw3);

    let res0 = _mm256_add_pd(e0, t0);
    let res1 = _mm256_add_pd(e1, t1);
    let res2 = _mm256_add_pd(e2, t2);
    let res3 = _mm256_add_pd(e3, t3);
    let res4 = _mm256_sub_pd(e0, t0);
    let res5 = _mm256_sub_pd(e1, t1);
    let res6 = _mm256_sub_pd(e2, t2);
    let res7 = _mm256_sub_pd(e3, t3);

    [res0, res1, res2, res3, res4, res5, res6, res7]
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn avx_fft8_precise<const INVERSE: bool>(
    r0: std::arch::x86_64::__m256d,
    r1: std::arch::x86_64::__m256d,
    r2: std::arch::x86_64::__m256d,
    r3: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 4] {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_permute2f128_pd, _mm256_set_pd, _mm256_sub_pd};

    let even_lo = _mm256_permute2f128_pd::<0x20>(r0, r1);
    let even_hi = _mm256_permute2f128_pd::<0x20>(r2, r3);
    let odd_lo = _mm256_permute2f128_pd::<0x31>(r0, r1);
    let odd_hi = _mm256_permute2f128_pd::<0x31>(r2, r3);

    let [e_lo, e_hi] = avx_fft4_precise::<INVERSE>(even_lo, even_hi);
    let [o_lo, o_hi] = avx_fft4_precise::<INVERSE>(odd_lo, odd_hi);

    let c = std::f64::consts::FRAC_1_SQRT_2;
    let tw1_im = if INVERSE { c } else { -c };
    let tw3_im = if INVERSE { c } else { -c };
    let tw2_im = if INVERSE { 1.0 } else { -1.0 };

    let tw_lo = _mm256_set_pd(tw1_im, c, 0.0, 1.0);
    let tw_hi = _mm256_set_pd(tw3_im, -c, tw2_im, 0.0);

    let t_lo = avx_cmul_precise(o_lo, tw_lo);
    let t_hi = avx_cmul_precise(o_hi, tw_hi);

    let res0 = _mm256_add_pd(e_lo, t_lo);
    let res1 = _mm256_add_pd(e_hi, t_hi);
    let res2 = _mm256_sub_pd(e_lo, t_lo);
    let res3 = _mm256_sub_pd(e_hi, t_hi);

    [res0, res1, res2, res3]
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn sse_cmul(
    a: std::arch::x86_64::__m128d,
    b: std::arch::x86_64::__m128d,
) -> std::arch::x86_64::__m128d {
    use std::arch::x86_64::{_mm_fmaddsub_pd, _mm_movedup_pd, _mm_mul_pd, _mm_permute_pd};
    let re_a = _mm_movedup_pd(a);
    let im_a_shuf = _mm_permute_pd::<0x03>(a);
    let b_shuf = _mm_permute_pd::<0x01>(b);
    let prod2 = _mm_mul_pd(im_a_shuf, b_shuf);
    _mm_fmaddsub_pd(re_a, b, prod2)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn rotate_minus_i(v: std::arch::x86_64::__m128d) -> std::arch::x86_64::__m128d {
    use std::arch::x86_64::{_mm_permute_pd, _mm_set_pd, _mm_xor_pd};
    let perm = _mm_permute_pd::<0x01>(v);
    _mm_xor_pd(perm, _mm_set_pd(-0.0, 0.0))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn rotate_plus_i(v: std::arch::x86_64::__m128d) -> std::arch::x86_64::__m128d {
    use std::arch::x86_64::{_mm_permute_pd, _mm_set_pd, _mm_xor_pd};
    let perm = _mm_permute_pd::<0x01>(v);
    _mm_xor_pd(perm, _mm_set_pd(0.0, -0.0))
}

impl MixedRadixScalar for f32 {
    const HALF_CYCLIC_RADER_THRESHOLD: usize =
        crate::application::execution::kernel::components::rader::HALF_CYCLIC_THRESHOLD;
    const HALF_CYCLIC_RADER_PRIMES: &'static [usize] = &[];
    const COMPOSITE_RADICES_200: &'static [usize] = &[4, 2, 5, 5];
    const FORCE_COMPOSITE_63: bool = true;
    const FORCE_COMPOSITE_72: bool = true;
    const PREFER_BLUESTEIN_MID_RADER: bool = true;
    const BLUESTEIN_PAD_POWER_OF_TWO: bool = true;
    const BLUESTEIN_NATIVE_PHASE_TRIG: bool = true;

    type Complex = Complex32;

    #[inline]
    fn complex(re: f64, im: f64) -> Complex32 {
        Complex32::new(re as f32, im as f32)
    }

    #[inline]
    fn cached_twiddle_fwd(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_fwd(n)
    }
    #[inline]
    fn cached_twiddle_inv(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_inv(n)
    }
    #[inline]
    fn with_twiddle_fwd<R>(n: usize, f: impl FnOnce(&[Complex32]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::twiddle::with_twiddle_fwd(n, f)
    }
    #[inline]
    fn with_twiddle_inv<R>(n: usize, f: impl FnOnce(&[Complex32]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::twiddle::with_twiddle_inv(n, f)
    }
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_stockham_scratch(n, f)
    }
    #[inline]
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_pfa_scratch(n, f)
    }
    #[inline]
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_rader_padded_scratch(n, f)
    }
    #[inline]
    fn with_bluestein_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_bluestein_scratch(n, f)
    }

    #[inline]
    fn cached_rader_spectrum<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> Arc<[Complex32]> {
        let key = (n, INVERSE as usize, generator_inverse);
        cached_rader_spectrum(key, |_| {
            build_rader_spectrum_vec::<f32, INVERSE>(n, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_negacyclic_spectra<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> (Arc<[Complex32]>, Arc<[Complex32]>) {
        let key = (n, INVERSE as usize, generator_inverse);
        cached_rader_negacyclic_spectra(key, |_| {
            build_rader_negacyclic_spectra::<f32, INVERSE>(n, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_neg_twiddles(m: usize) -> Arc<[Complex32]> {
        cached_rader_neg_twiddles(m, build_rader_negacyclic_twiddles::<f32>)
    }

    #[inline]
    fn cached_four_step_twiddles<const INVERSE: bool>(
        n: usize,
        n1: usize,
        n2: usize,
    ) -> Arc<[Complex32]> {
        cached_four_step_twiddles::<Complex32, INVERSE>(n, n1, n2)
    }
    #[inline]
    fn pointwise_mul(a: &mut [Complex32], b: &[Complex32]) {
        pointwise_mul_reduced::<false>(a, b);
    }
    #[inline]
    fn pointwise_mul_conj(a: &mut [Complex32], b: &[Complex32]) {
        pointwise_mul_reduced::<true>(a, b);
    }
    #[inline]
    fn stockham_forward(data: &mut [Complex32], scratch: &mut [Complex32], twiddles: &[Complex32]) {
        <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
    }
    #[inline]
    fn stockham_forward_normalized(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
        n: usize,
    ) {
        <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
        normalize_inplace(data, 1.0_f32 / n as f32);
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline)]
    fn stockham_forward_sized<const LOG2: u32>(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
    ) {
        <f32 as stockham::StockhamKernel>::forward_with_scratch_sized::<LOG2>(
            data, scratch, twiddles,
        );
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline)]
    fn stockham_forward_normalized_sized<const LOG2: u32>(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
    ) {
        Self::stockham_forward_sized::<LOG2>(data, scratch, twiddles);
        normalize_inplace(data, 1.0_f32 / (1usize << LOG2) as f32);
    }

    #[inline]
    fn short_winograd<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex32]) -> bool {
        crate::application::execution::kernel::mixed_radix::traits::short_winograd::<
            Self,
            INVERSE,
            NORMALIZE,
        >(data)
    }
    #[inline]
    unsafe fn small_pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
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
                <Self as crate::application::execution::kernel::mixed_radix::traits::ShortDft<3>>::dft::<INVERSE>(data_ref);
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
                <Self as crate::application::execution::kernel::mixed_radix::traits::ShortDft<5>>::dft::<INVERSE>(data_ref);
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
                crate::application::execution::kernel::components::winograd::dft6_impl::<
                    Self,
                    INVERSE,
                >(data_ref);
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
                <Self as crate::application::execution::kernel::mixed_radix::traits::ShortDft<7>>::dft::<INVERSE>(data_ref);
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
                crate::application::execution::kernel::components::winograd::dft16_impl::<
                    Self,
                    INVERSE,
                >(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex32::new(0.0625, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
                true
            }
            32 => {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
                true
            }
            64 => {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
                true
            }
            9 => {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex32; 9]>();
                crate::application::execution::kernel::components::winograd::dft9_impl::<
                    Self,
                    INVERSE,
                >(data_ref);
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
    #[inline]
    fn composite_forward(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::forward_inplace_with_radices(data, radices);
    }
    #[inline]
    fn composite_forward_with_pointwise(
        data: &mut [Complex32],
        radices: &[usize],
        pointwise_spectrum: &[Complex32],
    ) {
        radix_composite::forward_inplace_with_pointwise(data, radices, pointwise_spectrum);
    }
    #[inline]
    fn composite_inverse_unnorm(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::inverse_inplace_unnorm_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::inverse_inplace_with_radices(data, radices);
    }
    #[inline]
    fn normalize(data: &mut [Complex32], n: usize) {
        normalize_inplace(data, 1.0_f32 / n as f32);
    }
    #[inline]
    fn transpose_matrix(src: &[Complex32], dst: &mut [Complex32], n1: usize, n2: usize) {
        transpose_matrix_reduced(src, dst, n1, n2);
    }

    #[inline]
    unsafe fn small_pot_inplace_sized<
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
                crate::application::execution::kernel::components::winograd::dft16_impl::<
                    Self,
                    INVERSE,
                >(data_ref);
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
                    let twiddles = Self::small_pot_twiddles::<INVERSE>(32);
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
                        Self,
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
                    let twiddles = Self::small_pot_twiddles::<INVERSE>(64);
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
                        Self,
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

    #[inline]
    fn pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    ) {
        let n = data.len();
        match n {
            2 => unsafe {
                Self::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data);
            },
            4 => unsafe {
                Self::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data);
            },
            8 => unsafe {
                Self::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data);
            },
            16 => unsafe {
                Self::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data);
            },
            32 => unsafe {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
            },
            64 => unsafe {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
            },
            _ => {
                Self::with_scratch(n, |scratch| {
                    if INVERSE && NORMALIZE {
                        Self::stockham_forward_normalized(data, scratch, twiddles, n);
                    } else {
                        Self::stockham_forward(data, scratch, twiddles);
                    }
                });
            }
        }
    }

    fn pot_inplace_sized<
        const INVERSE: bool,
        const NORMALIZE: bool,
        S: PoTStrategy,
        const LOG2: u32,
    >(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
        _s: SizedPoT<S, LOG2>,
    ) {
        // const LOG2 drives selection (zero-cost monomorph); for <=64 preserve direct small
        // no-scratch path (memory efficiency + best for 32/64 which have dedicated AVX fixed column);
        // for 128+ (md-worst PoT) use stockham sized path so const LOG2 flows end-to-end to
        // kernel forward_with_scratch_sized -> transform_sized / with_strategy / len* bodies.
        match LOG2 {
            1 => unsafe {
                Self::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data);
            },
            2 => unsafe {
                Self::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data);
            },
            3 => unsafe {
                Self::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data);
            },
            4 => unsafe {
                Self::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data);
            },
            5 => unsafe {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
            },
            6 => unsafe {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
            },
            _ => {
                let n = 1usize << LOG2;
                Self::with_scratch(n, |scratch| {
                    if INVERSE && NORMALIZE {
                        Self::stockham_forward_normalized_sized::<LOG2>(data, scratch, twiddles);
                    } else {
                        Self::stockham_forward_sized::<LOG2>(data, scratch, twiddles);
                    }
                });
            }
        }
    }

    #[inline]
    fn small_pot_twiddles<const INVERSE: bool>(n: usize) -> &'static [Self::Complex] {
        let idx = n.trailing_zeros() as usize;
        if INVERSE {
            &TWIDDLES_INV_REDUCED[idx]
        } else {
            &TWIDDLES_FWD_REDUCED[idx]
        }
    }

    fn use_generated_codelet_plan(_n: usize) -> bool {
        // Default; actual policy in higher layers or for f32 reduced.
        false
    }
}

impl MixedRadixScalar for f64 {
    const HALF_CYCLIC_RADER_THRESHOLD: usize = 1024;
    const HALF_CYCLIC_RADER_PRIMES: &'static [usize] = &[67];
    const COMPOSITE_RADICES_200: &'static [usize] = &[4, 5, 5, 2];
    const FORCE_COMPOSITE_63: bool = false;
    const FORCE_COMPOSITE_72: bool = false;
    const PREFER_BLUESTEIN_MID_RADER: bool = false;
    const BLUESTEIN_PAD_POWER_OF_TWO: bool = false;
    const BLUESTEIN_NATIVE_PHASE_TRIG: bool = false;

    type Complex = Complex64;

    #[inline]
    fn complex(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[inline]
    fn cached_twiddle_fwd(n: usize) -> Arc<[Complex64]> {
        cached_twiddle_fwd(n)
    }
    #[inline]
    fn cached_twiddle_inv(n: usize) -> Arc<[Complex64]> {
        cached_twiddle_inv(n)
    }
    #[inline]
    fn with_twiddle_fwd<R>(n: usize, f: impl FnOnce(&[Complex64]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::twiddle::with_twiddle_fwd(n, f)
    }
    #[inline]
    fn with_twiddle_inv<R>(n: usize, f: impl FnOnce(&[Complex64]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::twiddle::with_twiddle_inv(n, f)
    }
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_stockham_scratch(n, f)
    }
    #[inline]
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_pfa_scratch(n, f)
    }
    #[inline]
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_rader_padded_scratch(n, f)
    }
    #[inline]
    fn with_bluestein_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_bluestein_scratch(n, f)
    }

    #[inline]
    fn cached_rader_spectrum<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> Arc<[Complex64]> {
        let key = (n, INVERSE as usize, generator_inverse);
        cached_rader_spectrum(key, |_| {
            build_rader_spectrum_vec::<f64, INVERSE>(n, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_negacyclic_spectra<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> (Arc<[Complex64]>, Arc<[Complex64]>) {
        let key = (n, INVERSE as usize, generator_inverse);
        cached_rader_negacyclic_spectra(key, |_| {
            build_rader_negacyclic_spectra::<f64, INVERSE>(n, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_neg_twiddles(m: usize) -> Arc<[Complex64]> {
        cached_rader_neg_twiddles(m, build_rader_negacyclic_twiddles::<f64>)
    }

    #[inline]
    fn cached_four_step_twiddles<const INVERSE: bool>(
        n: usize,
        n1: usize,
        n2: usize,
    ) -> Arc<[Complex64]> {
        cached_four_step_twiddles::<Complex64, INVERSE>(n, n1, n2)
    }
    #[inline]
    fn pointwise_mul(a: &mut [Complex64], b: &[Complex64]) {
        pointwise_mul_precise::<false>(a, b);
    }
    #[inline]
    fn pointwise_mul_conj(a: &mut [Complex64], b: &[Complex64]) {
        pointwise_mul_precise::<true>(a, b);
    }
    #[inline]
    fn stockham_forward(data: &mut [Complex64], scratch: &mut [Complex64], twiddles: &[Complex64]) {
        <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
    }
    #[inline]
    fn stockham_forward_normalized(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
        n: usize,
    ) {
        <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
        normalize_inplace(data, 1.0_f64 / n as f64);
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline)]
    fn stockham_forward_sized<const LOG2: u32>(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
    ) {
        <f64 as stockham::StockhamKernel>::forward_with_scratch_sized::<LOG2>(
            data, scratch, twiddles,
        );
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline)]
    fn stockham_forward_normalized_sized<const LOG2: u32>(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
    ) {
        Self::stockham_forward_sized::<LOG2>(data, scratch, twiddles);
        normalize_inplace(data, 1.0_f64 / (1usize << LOG2) as f64);
    }

    #[inline]
    fn short_winograd<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex64]) -> bool {
        crate::application::execution::kernel::mixed_radix::traits::short_winograd::<
            Self,
            INVERSE,
            NORMALIZE,
        >(data)
    }
    #[inline]
    unsafe fn small_pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Complex64],
    ) -> bool {
        let n = data.len();
        match n {
            2 => {
                Self::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data);
                true
            }
            3 => {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 3]>();
                <Self as crate::application::execution::kernel::mixed_radix::traits::ShortDft<3>>::dft::<INVERSE>(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex64::new(1.0 / 3.0, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
                true
            }
            4 => {
                Self::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data);
                true
            }
            5 => {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 5]>();
                <Self as crate::application::execution::kernel::mixed_radix::traits::ShortDft<5>>::dft::<INVERSE>(data_ref);
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
                crate::application::execution::kernel::components::winograd::dft6_impl::<
                    Self,
                    INVERSE,
                >(data_ref);
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
                <Self as crate::application::execution::kernel::mixed_radix::traits::ShortDft<7>>::dft::<INVERSE>(data_ref);
                if INVERSE && NORMALIZE {
                    let scale = Complex64::new(1.0 / 7.0, 0.0);
                    for x in data_ref.iter_mut() {
                        *x *= scale;
                    }
                }
                true
            }
            8 => {
                Self::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data);
                true
            }
            16 => {
                Self::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data);
                true
            }
            32 => {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
                true
            }
            64 => {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
                true
            }
            9 => {
                let data_ref = &mut *data.as_mut_ptr().cast::<[Complex64; 9]>();
                crate::application::execution::kernel::components::winograd::dft9_impl::<
                    Self,
                    INVERSE,
                >(data_ref);
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
    #[inline]
    fn composite_forward(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::forward_inplace_with_radices(data, radices);
    }
    #[inline]
    fn composite_forward_with_pointwise(
        data: &mut [Complex64],
        radices: &[usize],
        pointwise_spectrum: &[Complex64],
    ) {
        radix_composite::forward_inplace_with_pointwise(data, radices, pointwise_spectrum);
    }
    #[inline]
    fn composite_inverse_unnorm(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::inverse_inplace_unnorm_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::inverse_inplace_with_radices(data, radices);
    }
    #[inline]
    fn normalize(data: &mut [Complex64], n: usize) {
        normalize_inplace(data, 1.0_f64 / n as f64);
    }
    #[inline]
    fn transpose_matrix(src: &[Complex64], dst: &mut [Complex64], n1: usize, n2: usize) {
        transpose_matrix_precise(src, dst, n1, n2);
    }

    #[inline]
    unsafe fn small_pot_inplace_sized<
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
                        _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_set_pd,
                        _mm_storeu_pd, _mm_sub_pd,
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
                        _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_set_pd,
                        _mm_storeu_pd, _mm_sub_pd,
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
                        Self,
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
                        Self,
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
                        Self,
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

    #[inline]
    fn pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    ) {
        let n = data.len();
        match n {
            2 => unsafe {
                Self::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data);
            },
            4 => unsafe {
                Self::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data);
            },
            8 => unsafe {
                Self::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data);
            },
            16 => unsafe {
                Self::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data);
            },
            32 => unsafe {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
            },
            64 => unsafe {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
            },
            _ => {
                Self::with_scratch(n, |scratch| {
                    if INVERSE && NORMALIZE {
                        Self::stockham_forward_normalized(data, scratch, twiddles, n);
                    } else {
                        Self::stockham_forward(data, scratch, twiddles);
                    }
                });
            }
        }
    }

    fn pot_inplace_sized<
        const INVERSE: bool,
        const NORMALIZE: bool,
        S: PoTStrategy,
        const LOG2: u32,
    >(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
        _s: SizedPoT<S, LOG2>,
    ) {
        // const LOG2 drives selection (zero-cost monomorph); for <=64 preserve direct small
        // no-scratch path (memory efficiency + best for 32/64 which have dedicated AVX fixed column);
        // for 128+ (md-worst PoT) use stockham sized path so const LOG2 flows end-to-end to
        // kernel forward_with_scratch_sized -> transform_sized / with_strategy / len* bodies.
        // twiddles passed directly as &[Complex] (zero-copy reference from plan).
        match LOG2 {
            1 => unsafe {
                Self::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data);
            },
            2 => unsafe {
                Self::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data);
            },
            3 => unsafe {
                Self::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data);
            },
            4 => unsafe {
                Self::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data);
            },
            5 => unsafe {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
            },
            6 => unsafe {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
            },
            _ => {
                let n = 1usize << LOG2;
                Self::with_scratch(n, |scratch| {
                    if INVERSE && NORMALIZE {
                        Self::stockham_forward_normalized_sized::<LOG2>(data, scratch, twiddles);
                    } else {
                        Self::stockham_forward_sized::<LOG2>(data, scratch, twiddles);
                    }
                });
            }
        }
    }

    #[inline]
    fn small_pot_twiddles<const INVERSE: bool>(n: usize) -> &'static [Self::Complex] {
        let idx = n.trailing_zeros() as usize;
        if INVERSE {
            &TWIDDLES_INV_PRECISE[idx]
        } else {
            &TWIDDLES_FWD_PRECISE[idx]
        }
    }

    fn use_generated_codelet_plan(_n: usize) -> bool {
        false
    }
}

static BLUESTEIN_REDUCED_CACHE: std::sync::LazyLock<
    parking_lot::RwLock<
        rustc_hash::FxHashMap<
            super::trait_def::BluesteinKey,
            super::trait_def::BluesteinEntry<Complex32>,
        >,
    >,
> = std::sync::LazyLock::new(|| parking_lot::RwLock::new(rustc_hash::FxHashMap::default()));

static BLUESTEIN_PRECISE_CACHE: std::sync::LazyLock<
    parking_lot::RwLock<
        rustc_hash::FxHashMap<
            super::trait_def::BluesteinKey,
            super::trait_def::BluesteinEntry<Complex64>,
        >,
    >,
> = std::sync::LazyLock::new(|| parking_lot::RwLock::new(rustc_hash::FxHashMap::default()));

const NONE_BLUESTEIN_C32: Option<super::trait_def::BluesteinEntry<Complex32>> = None;
const NONE_BLUESTEIN_C64: Option<super::trait_def::BluesteinEntry<Complex64>> = None;
const FLAT_CACHE_LIMIT: usize = 4096;

thread_local! {
    static TL_BLUESTEIN_REDUCED: std::cell::RefCell<rustc_hash::FxHashMap<super::trait_def::BluesteinKey, super::trait_def::BluesteinEntry<Complex32>>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_BLUESTEIN_PRECISE: std::cell::RefCell<rustc_hash::FxHashMap<super::trait_def::BluesteinKey, super::trait_def::BluesteinEntry<Complex64>>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::with_capacity_and_hasher(8, Default::default()));

    static TL_BLUESTEIN_REDUCED_FLAT: std::cell::RefCell<[Option<super::trait_def::BluesteinEntry<Complex32>>; 8192]> =
        std::cell::RefCell::new([NONE_BLUESTEIN_C32; 8192]);
    static TL_BLUESTEIN_PRECISE_FLAT: std::cell::RefCell<[Option<super::trait_def::BluesteinEntry<Complex64>>; 8192]> =
        std::cell::RefCell::new([NONE_BLUESTEIN_C64; 8192]);
}

impl super::trait_def::BluesteinStore for f32 {
    type Cpx = Complex32;
    #[inline]
    fn tl_get(
        key: super::trait_def::BluesteinKey,
    ) -> Option<super::trait_def::BluesteinEntry<Self::Cpx>> {
        let (n, inv, _) = key;
        if n < FLAT_CACHE_LIMIT {
            let idx = (n << 1) | usize::from(inv);
            TL_BLUESTEIN_REDUCED_FLAT.with(|c| c.borrow()[idx].clone())
        } else {
            let result = TL_BLUESTEIN_REDUCED.with(|c| c.borrow().get(&key).cloned());
            #[cfg(feature = "cache-profiling")]
            if result.is_some() {
                crate::application::execution::kernel::mixed_radix::caches::profiler::get()
                    .bluestein_reduced
                    .tl_hit();
            }
            result
        }
    }
    #[inline]
    fn tl_insert(
        key: super::trait_def::BluesteinKey,
        val: super::trait_def::BluesteinEntry<Self::Cpx>,
    ) {
        let (n, inv, _) = key;
        if n < FLAT_CACHE_LIMIT {
            let idx = (n << 1) | usize::from(inv);
            TL_BLUESTEIN_REDUCED_FLAT.with(|c| c.borrow_mut()[idx] = Some(val));
        } else {
            TL_BLUESTEIN_REDUCED.with(|c| c.borrow_mut().insert(key, val));
        }
    }
    #[inline]
    fn global() -> &'static parking_lot::RwLock<
        rustc_hash::FxHashMap<
            super::trait_def::BluesteinKey,
            super::trait_def::BluesteinEntry<Self::Cpx>,
        >,
    > {
        &BLUESTEIN_REDUCED_CACHE
    }
}

impl super::trait_def::BluesteinStore for f64 {
    type Cpx = Complex64;
    #[inline]
    fn tl_get(
        key: super::trait_def::BluesteinKey,
    ) -> Option<super::trait_def::BluesteinEntry<Self::Cpx>> {
        let (n, inv, _) = key;
        if n < FLAT_CACHE_LIMIT {
            let idx = (n << 1) | usize::from(inv);
            TL_BLUESTEIN_PRECISE_FLAT.with(|c| c.borrow()[idx].clone())
        } else {
            let result = TL_BLUESTEIN_PRECISE.with(|c| c.borrow().get(&key).cloned());
            #[cfg(feature = "cache-profiling")]
            if result.is_some() {
                crate::application::execution::kernel::mixed_radix::caches::profiler::get()
                    .bluestein_precise
                    .tl_hit();
            }
            result
        }
    }
    #[inline]
    fn tl_insert(
        key: super::trait_def::BluesteinKey,
        val: super::trait_def::BluesteinEntry<Self::Cpx>,
    ) {
        let (n, inv, _) = key;
        if n < FLAT_CACHE_LIMIT {
            let idx = (n << 1) | usize::from(inv);
            TL_BLUESTEIN_PRECISE_FLAT.with(|c| c.borrow_mut()[idx] = Some(val));
        } else {
            TL_BLUESTEIN_PRECISE.with(|c| c.borrow_mut().insert(key, val));
        }
    }
    #[inline]
    fn global() -> &'static parking_lot::RwLock<
        rustc_hash::FxHashMap<
            super::trait_def::BluesteinKey,
            super::trait_def::BluesteinEntry<Self::Cpx>,
        >,
    > {
        &BLUESTEIN_PRECISE_CACHE
    }
}
