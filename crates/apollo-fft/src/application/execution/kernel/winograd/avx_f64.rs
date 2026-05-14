#![allow(dead_code)]
use num_complex::Complex64;
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use std::arch::x86_64::{
    __m128d, __m256d, _mm256_add_pd, _mm256_castpd128_pd256, _mm256_extractf128_pd,
    _mm256_fmaddsub_pd, _mm256_insertf128_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd,
    _mm256_setr_pd, _mm256_storeu_pd, _mm256_sub_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    _mm_add_pd, _mm_permute_pd, _mm_set_pd, _mm_sub_pd, _mm_xor_pd,
};

/// Packed 2×Complex64 complex multiplication using AVX+FMA.
///
/// Computes `[a0*b0, a1*b1]` where `a = [a0.re, a0.im, a1.re, a1.im]`.
/// Uses the identity `(ar + i·ai)·(br + i·bi) = ar·br − ai·bi + i·(ar·bi + ai·br)`,
/// mapped to `_mm256_fmaddsub_pd(ar, b, ai·bsw)` where `bsw = permute(b, 0b0101)`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
unsafe fn cmul2_64(a: __m256d, b: __m256d) -> __m256d {
    let ar = _mm256_unpacklo_pd(a, a); // [a0.re, a0.re, a1.re, a1.re]
    let ai = _mm256_unpackhi_pd(a, a); // [a0.im, a0.im, a1.im, a1.im]
    let bsw = _mm256_permute_pd(b, 0b0101); // [b0.im, b0.re, b1.im, b1.re]
    _mm256_fmaddsub_pd(ar, b, _mm256_mul_pd(ai, bsw))
}

/// AVX-accelerated in-place DFT-4 for `Complex64`.
///
/// Decomposition: DFT-4 = two parallel DFT-2 butterflies (stage 1) followed by
/// a single twiddle W_4^1 = −i (forward) or +i (inverse) on t3, then two
/// DFT-2 butterflies (stage 2).  All stages are fully vectorised with AVX/SSE.
///
/// **Stage 1** packs `[x0,x2]` and `[x1,x3]` into two `__m256d` registers and
/// computes sum/dif with a single `add_pd`/`sub_pd` pair.  **Twiddle** is done
/// with `_mm_permute_pd` + `_mm_xor_pd` (sign bit flip) on the low 128-bit half
/// of `dif`.  **Stage 2** uses four 128-bit add/sub ops and two `insertf128_pd`
/// to pack the four output values into two `__m256d` for a 2×`storeu_pd` store.
///
/// Operation count (vs scalar `dft4_impl`):
/// - 2 load (vs 4 scalar loads of 4 Complex64)
/// - 2 add/sub (stage 1)
/// - 5 128-bit ops (extract ×4 + permute ×1 + xor ×1 + add/sub ×4 + insert ×2)
/// - 2 store
/// Total: ~17 µops vs ~32 scalar ops.
///
/// # Safety
/// Caller must ensure `target_feature = "avx"` and `target_feature = "fma"`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
pub unsafe fn dft4_avx_fma_64(data: &mut [Complex64; 4], inverse: bool) {
    // Load [x0, x1] and [x2, x3] as packed __m256d.
    // Layout: [x_k.re, x_k.im, x_{k+1}.re, x_{k+1}.im] per register.
    let v01 = _mm256_loadu_pd(data.as_ptr().cast::<f64>());
    let v23 = _mm256_loadu_pd(data.as_ptr().add(2).cast::<f64>());

    // Stage 1: two simultaneous DFT-2 butterflies.
    // sum = [t0.re, t0.im, t2.re, t2.im]  where t0 = x0+x2, t2 = x1+x3
    // dif = [t1.re, t1.im, t3.re, t3.im]  where t1 = x0-x2, t3 = x1-x3
    let sum = _mm256_add_pd(v01, v23);
    let dif = _mm256_sub_pd(v01, v23);

    // Extract 128-bit halves for cross-lane butterfly (AVX has no native
    // cross-lane SIMD add for the DFT-4 final stage structure).
    let sum_lo: __m128d = _mm256_extractf128_pd(sum, 0); // [t0.re, t0.im]
    let sum_hi: __m128d = _mm256_extractf128_pd(sum, 1); // [t2.re, t2.im]
    let dif_lo: __m128d = _mm256_extractf128_pd(dif, 0); // [t1.re, t1.im]
    let dif_hi: __m128d = _mm256_extractf128_pd(dif, 1); // [t3.re, t3.im]

    // Stage 2 twiddle: W_4^1 = -i (forward) or +i (inverse) applied to t3.
    // Forward:  (re, im) * (-i) = (im, -re)  → swap lanes then negate lane 1.
    // Inverse:  (re, im) * (+i) = (-im, re)  → swap lanes then negate lane 0.
    // _mm_permute_pd with imm8=0b01: result[0]=a[1]=t3.im, result[1]=a[0]=t3.re.
    let perm = _mm_permute_pd(dif_hi, 0b01); // [t3.im, t3.re]
                                             // _mm_set_pd(e1, e0): e0→lane0, e1→lane1.
    let t3_tw = if inverse {
        // [-t3.im, t3.re]: negate lane 0 → XOR sign bit at lane 0.
        _mm_xor_pd(perm, _mm_set_pd(0.0f64, -0.0f64))
    } else {
        // [t3.im, -t3.re]: negate lane 1 → XOR sign bit at lane 1.
        _mm_xor_pd(perm, _mm_set_pd(-0.0f64, 0.0f64))
    };

    // Stage 2 final DFT-2 butterflies.
    let t0_plus_t2 = _mm_add_pd(sum_lo, sum_hi); // out[0]
    let t0_minus_t2 = _mm_sub_pd(sum_lo, sum_hi); // out[2]
    let t1_plus_t3tw = _mm_add_pd(dif_lo, t3_tw); // out[1]
    let t1_minus_t3tw = _mm_sub_pd(dif_lo, t3_tw); // out[3]

    // Pack [out[0], out[1]] and [out[2], out[3]] into two __m256d for 2-store.
    let out01 = _mm256_insertf128_pd(_mm256_castpd128_pd256(t0_plus_t2), t1_plus_t3tw, 1);
    let out23 = _mm256_insertf128_pd(_mm256_castpd128_pd256(t0_minus_t2), t1_minus_t3tw, 1);
    _mm256_storeu_pd(data.as_mut_ptr().cast::<f64>(), out01);
    _mm256_storeu_pd(data.as_mut_ptr().add(2).cast::<f64>(), out23);
}

/// AVX+FMA-accelerated in-place DFT-8 for `Complex64`.
///
/// Full SIMD pipeline:
/// 1. Gather even/odd sub-arrays (8 scalar reads).
/// 2. `dft4_avx_fma_64` on each (vectorised DFT-4).
/// 3. AVX twiddle W_8^k via `cmul2_64` (2 packed complex mults).
/// 4. Butterfly combine (4 AVX add/sub stores).
///
/// Twiddle table (forward), packed as 2×Complex64 per register:
/// - `tw01 = [W_8^0, W_8^1] = [1, SQ2O2·(1−i)]`
/// - `tw23 = [W_8^2, W_8^3] = [−i, SQ2O2·(−1−i)]`
///
/// # Safety
/// Caller must ensure `target_feature = "avx"` and `target_feature = "fma"`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline]
pub unsafe fn dft8_avx_fma_64(data: &mut [Complex64; 8], inverse: bool) {
    // Step 1: gather even/odd sub-arrays.
    let mut even = [data[0], data[2], data[4], data[6]];
    let mut odd = [data[1], data[3], data[5], data[7]];

    // Step 2: vectorised DFT-4 on each sub-array.
    dft4_avx_fma_64(&mut even, inverse);
    dft4_avx_fma_64(&mut odd, inverse);

    // Step 3: AVX twiddle W_8^k via cmul2_64.
    // Twiddle constants (packed 2×Complex64):
    //   Forward: W^0=1+0i, W^1=SQ2O2−i·SQ2O2, W^2=0−i, W^3=−SQ2O2−i·SQ2O2
    //   Inverse: W^0=1+0i, W^{-1}=SQ2O2+i·SQ2O2, W^{-2}=0+i, W^{-3}=−SQ2O2+i·SQ2O2
    // _mm256_setr_pd(e0,e1,e2,e3): e0→lane0 (lowest addr), e3→lane3.
    const SQ2O2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let odd01 = _mm256_loadu_pd(odd.as_ptr().cast::<f64>());
    let odd23 = _mm256_loadu_pd(odd.as_ptr().add(2).cast::<f64>());
    let (tw01, tw23) = if inverse {
        (
            _mm256_setr_pd(1.0, 0.0, SQ2O2, SQ2O2),
            _mm256_setr_pd(0.0, 1.0, -SQ2O2, SQ2O2),
        )
    } else {
        (
            _mm256_setr_pd(1.0, 0.0, SQ2O2, -SQ2O2),
            _mm256_setr_pd(0.0, -1.0, -SQ2O2, -SQ2O2),
        )
    };
    let ot01 = cmul2_64(odd01, tw01);
    let ot23 = cmul2_64(odd23, tw23);

    // Step 4: AVX butterfly combine: data[0..4] = even ± ot, data[4..8] = even ∓ ot.
    let ev01 = _mm256_loadu_pd(even.as_ptr().cast::<f64>());
    let ev23 = _mm256_loadu_pd(even.as_ptr().add(2).cast::<f64>());
    _mm256_storeu_pd(data.as_mut_ptr().cast::<f64>(), _mm256_add_pd(ev01, ot01));
    _mm256_storeu_pd(
        data.as_mut_ptr().add(2).cast::<f64>(),
        _mm256_add_pd(ev23, ot23),
    );
    _mm256_storeu_pd(
        data.as_mut_ptr().add(4).cast::<f64>(),
        _mm256_sub_pd(ev01, ot01),
    );
    _mm256_storeu_pd(
        data.as_mut_ptr().add(6).cast::<f64>(),
        _mm256_sub_pd(ev23, ot23),
    );
}

