#![allow(dead_code)]
use num_complex::Complex32;
/// Packed 4×Complex32 complex multiplication using AVX+FMA.
///
/// Computes `[a0*b0, a1*b1, a2*b2, a3*b3]` in one `__m256` register
/// (4 Complex32 = 8 f32).  Uses `moveldup`/`movehdup` to broadcast re/im
/// lanes and `fmaddsub` for the Gauss-trick complex multiply.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
pub(crate) unsafe fn cmul4_32(
    a: std::arch::x86_64::__m256,
    b: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps,
    };
    let ar = _mm256_moveldup_ps(a); // broadcast .re of each Complex32
    let ai = _mm256_movehdup_ps(a); // broadcast .im of each Complex32
    let bsw = _mm256_permute_ps(b, 0xB1); // swap .re/.im of each Complex32
    _mm256_fmaddsub_ps(ar, b, _mm256_mul_ps(ai, bsw))
}

/// AVX+FMA in-place DFT-4 for `Complex32`.
///
/// Fits all 4 Complex32 (= 8 f32) into one `__m256`, performing the entire
/// DFT-4 with cross-lane permutes instead of scalar extract/insert.
///
/// **Stage 1** swaps 128-bit halves of the register to form `sum = [t0, t2]`
/// and `dif = [t1, t3]`.  **Twiddle** applies W_4^1 = −i (forward) or +i
/// (inverse) to t3 via a single `_mm_permute_ps` + `_mm_xor_ps`.  **Stage 2**
/// uses 64-bit pair swaps and add/sub, then `_mm_movelh_ps` to pack output.
///
/// # Safety
/// Caller must ensure `target_feature = "avx"`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
pub unsafe fn dft4_avx_fma_32(data: &mut [Complex32; 4], inverse: bool) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castps128_ps256, _mm256_castps256_ps128, _mm256_insertf128_ps,
        _mm256_loadu_ps, _mm256_permute2f128_ps, _mm256_storeu_ps, _mm256_sub_ps, _mm_add_ps,
        _mm_movelh_ps, _mm_permute_ps, _mm_set_ps, _mm_sub_ps, _mm_xor_ps,
    };

    // Load 4×Complex32 into one __m256: [x0.re, x0.im, x1.re, x1.im, x2.re, x2.im, x3.re, x3.im].
    let v = _mm256_loadu_ps(data.as_ptr().cast::<f32>());

    // Stage 1: two simultaneous DFT-2 butterflies via 128-bit lane swap.
    // vswap = [x2.re, x2.im, x3.re, x3.im, x0.re, x0.im, x1.re, x1.im]
    let vswap = _mm256_permute2f128_ps(v, v, 0x01);
    // sum lower half = [t0.re, t0.im, t2.re, t2.im] where t0=x0+x2, t2=x1+x3
    // dif lower half = [t1.re, t1.im, t3.re, t3.im] where t1=x0-x2, t3=x1-x3
    let sum = _mm256_add_ps(v, vswap);
    let dif = _mm256_sub_ps(v, vswap);
    let sum_lo = _mm256_castps256_ps128(sum);
    let dif_lo = _mm256_castps256_ps128(dif);

    // Twiddle W_4^1 on t3 (lanes 2,3 of dif_lo).
    // permute 0xB4 = 0b10_11_01_00: out[2]=a[3], out[3]=a[2] — swap t3.re,t3.im.
    // Result: [t1.re, t1.im, t3.im, t3.re]
    let dif_perm = _mm_permute_ps(dif_lo, 0xB4);
    // Forward W_4^1 = −i: (re,im)→(im,−re) → negate lane 3 (t3.re after swap).
    // Inverse W_4^{−1} = +i: (re,im)→(−im,re) → negate lane 2 (t3.im after swap).
    // _mm_set_ps(e3,e2,e1,e0): e0→result[0], e3→result[3].
    let sign = if inverse {
        _mm_set_ps(0.0f32, -0.0f32, 0.0f32, 0.0f32) // negate lane 2
    } else {
        _mm_set_ps(-0.0f32, 0.0f32, 0.0f32, 0.0f32) // negate lane 3
    };
    let dif_tw = _mm_xor_ps(dif_perm, sign); // [t1.re, t1.im, t3_tw.re, t3_tw.im]

    // Stage 2: DFT-2 butterflies via 64-bit pair swap + add/sub.
    // permute 0x4E = 0b01_00_11_10: swap 64-bit pairs.
    let s_perm = _mm_permute_ps(sum_lo, 0x4E); // [t2.re, t2.im, t0.re, t0.im]
    let d_perm = _mm_permute_ps(dif_tw, 0x4E); // [t3_tw.re, t3_tw.im, t1.re, t1.im]
                                               // Lower 64 of each add/sub holds the correct output element.
    let add_s = _mm_add_ps(sum_lo, s_perm); // lower 64: out[0]=t0+t2
    let sub_s = _mm_sub_ps(sum_lo, s_perm); // lower 64: out[2]=t0-t2
    let add_d = _mm_add_ps(dif_tw, d_perm); // lower 64: out[1]=t1+t3_tw
    let sub_d = _mm_sub_ps(dif_tw, d_perm); // lower 64: out[3]=t1-t3_tw

    // Pack outputs: movelh concatenates lower 64 bits of each __m128.
    let out01 = _mm_movelh_ps(add_s, add_d); // [out0.re, out0.im, out1.re, out1.im]
    let out23 = _mm_movelh_ps(sub_s, sub_d); // [out2.re, out2.im, out3.re, out3.im]
    let result = _mm256_insertf128_ps(_mm256_castps128_ps256(out01), out23, 1);
    _mm256_storeu_ps(data.as_mut_ptr().cast::<f32>(), result);
}

/// AVX+FMA in-place DFT-8 for `Complex32`.
///
/// Mirrors `dft8_avx_fma_64` but uses `__m256` (8 f32 = 4 Complex32) throughout,
/// so both the even and odd sub-arrays each fit in one register.
///
/// Pipeline:
/// 1. `dft4_avx_fma_32` on even and odd sub-arrays.
/// 2. Twiddle W_8^k (k=0..3) via `cmul4_32` on the 4-complex odd vector.
/// 3. `add`/`sub` butterfly with even.
///
/// # Safety
/// Caller must ensure `target_feature = "avx"` and `target_feature = "fma"`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
pub unsafe fn dft8_avx_fma_32(data: &mut [Complex32; 8], inverse: bool) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_setr_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    // Step 1: gather even/odd sub-arrays.
    let mut even = [data[0], data[2], data[4], data[6]];
    let mut odd = [data[1], data[3], data[5], data[7]];

    // Step 2: vectorised DFT-4 on each sub-array.
    dft4_avx_fma_32(&mut even, inverse);
    dft4_avx_fma_32(&mut odd, inverse);

    // Step 3: AVX twiddle W_8^k via cmul4_32.
    // Twiddle constants (packed 4×Complex32):
    //   Forward: W^0=1+0i, W^1=SQ2O2−iSQ2O2, W^2=0−1i, W^3=−SQ2O2−iSQ2O2
    //   Inverse: W^0=1+0i, W^{-1}=SQ2O2+iSQ2O2, W^{-2}=0+1i, W^{-3}=−SQ2O2+iSQ2O2
    const SQ2O2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let odd_v = _mm256_loadu_ps(odd.as_ptr().cast::<f32>());
    let tw = if inverse {
        _mm256_setr_ps(1.0, 0.0, SQ2O2, SQ2O2, 0.0, 1.0, -SQ2O2, SQ2O2)
    } else {
        _mm256_setr_ps(1.0, 0.0, SQ2O2, -SQ2O2, 0.0, -1.0, -SQ2O2, -SQ2O2)
    };
    let ot = cmul4_32(odd_v, tw);

    // Step 4: butterfly combine.
    let ev = _mm256_loadu_ps(even.as_ptr().cast::<f32>());
    _mm256_storeu_ps(data.as_mut_ptr().cast::<f32>(), _mm256_add_ps(ev, ot));
    _mm256_storeu_ps(
        data.as_mut_ptr().add(4).cast::<f32>(),
        _mm256_sub_ps(ev, ot),
    );
}
