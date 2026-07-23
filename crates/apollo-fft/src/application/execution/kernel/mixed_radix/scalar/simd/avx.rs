//! Shared x86-64 SIMD primitives for the mixed-radix scalar kernels.
//!
//! These wrap raw intrinsics behind one home so the trait wiring in `impls`
//! and the unrolled codelets in `small_pot` share a single definition.

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn avx_cmul_precise(
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
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn avx_fft4_parallel_precise<
    const INVERSE: bool,
>(
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
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn avx_fft4_precise<
    const INVERSE: bool,
>(
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
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn avx_fft4_reduced<
    const INVERSE: bool,
>(
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
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn avx_fft8_parallel_precise<
    const INVERSE: bool,
>(
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
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn avx_fft8_precise<
    const INVERSE: bool,
>(
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
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn sse_cmul_ps(
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
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn rotate_minus_i_ps(
    v: std::arch::x86_64::__m128,
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::{_mm_setr_ps, _mm_shuffle_ps, _mm_xor_ps};
    let perm = _mm_shuffle_ps(v, v, 0xB1);
    _mm_xor_ps(perm, _mm_setr_ps(0.0, -0.0, 0.0, -0.0))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn rotate_plus_i_ps(
    v: std::arch::x86_64::__m128,
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::{_mm_setr_ps, _mm_shuffle_ps, _mm_xor_ps};
    let perm = _mm_shuffle_ps(v, v, 0xB1);
    _mm_xor_ps(perm, _mm_setr_ps(-0.0, 0.0, -0.0, 0.0))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn sse_cmul(
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
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn rotate_minus_i(
    v: std::arch::x86_64::__m128d,
) -> std::arch::x86_64::__m128d {
    use std::arch::x86_64::{_mm_permute_pd, _mm_set_pd, _mm_xor_pd};
    let perm = _mm_permute_pd::<0x01>(v);
    _mm_xor_pd(perm, _mm_set_pd(-0.0, 0.0))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) unsafe fn rotate_plus_i(
    v: std::arch::x86_64::__m128d,
) -> std::arch::x86_64::__m128d {
    use std::arch::x86_64::{_mm_permute_pd, _mm_set_pd, _mm_xor_pd};
    let perm = _mm_permute_pd::<0x01>(v);
    _mm_xor_pd(perm, _mm_set_pd(0.0, -0.0))
}
