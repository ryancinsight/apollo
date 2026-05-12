//! Winograd short-DFT kernels for sizes 2, 3, 4, 5, 7, 8, 16, 32, and 64.
//!
//! ## Mathematical foundation
//!
//! Winograd's theorem (1978) states that the number of multiplications required
//! to compute an N-point DFT equals `2N − log₂N − 2` for power-of-two N when
//! additions are "free". The construction decomposes the DFT matrix into a
//! product `A · M · B` where B performs only additions (pre-additions), M
//! applies a small set of multiplications by irrational constants, and A
//! performs final additions (post-additions). For our implementation we use
//! the recursive Cooley-Tukey structure to build each kernel from smaller
//! pieces.
//!
//! ## Implementation strategy
//!
//! Rather than deriving A·M·B factorings analytically for each size (which
//! generates kernel-specific hard-coded coefficient arrays), we use a hybrid:
//!
//! - **DFT-2 and DFT-4**: pure addition/subtraction butterflies; zero
//!   multiplications, zero trig (this is the classical Winograd DFT-2 and
//!   DFT-4 result).
//! - **DFT-8**: implemented as two DFT-4 sub-transforms plus the standard
//!   8-point butterfly with the four special twiddles
//!   `{1, -i, √2/2·(1-i), √2/2·(-1-i)}` expressed as precomputed constants,
//!   reducing multiplications vs the generic DFT matrix approach.
//! - **DFT-16 and DFT-32**: hierarchically decomposed as `R` DFT-4/8 sub-
//!   transforms of length `N/R` followed by twiddle multiplications and a
//!   final DFT-R butterfly stage.  The recursion exploits the fact that the
//!   inner short-DFT kernels require no multiplications, so only the inter-
//!   stage twiddle multiplications count against the Winograd bound.
//! - **DFT-64**: decomposed as 4 DFT-16 sub-transforms plus 64 twiddle
//!   multiplications.
//!
//! All kernels operate directly on a mutable slice segment and return values
//! in normal (output) order.  They are intended to be invoked per-group
//! inside the outer radix-R iterator in `radix8`, `radix16`, `radix32`, and
//! `radix64`.
//!
//! ## Notation
//!
//! `W_N^k = exp(-2πi·k/N)` (forward convention). For inverse transforms the
//! sign flips: `W_N^{-k} = exp(+2πi·k/N)`. All kernels accept an `inverse:
//! bool` flag and negate the imaginary twiddle component accordingly.
//!
//! ## References
//!
//! - Winograd, S. (1978). On computing the discrete Fourier transform.
//!   *Mathematics of Computation*, 32(141), 175-199.
//! - Van Loan, C. (1992). *Computational Frameworks for the Fast Fourier
//!   Transform*. SIAM.
//! - Blahut, R.E. (2010). *Fast Algorithms for Signal Processing*. Cambridge
//!   University Press.

#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::ptr_as_ptr)]

use num_complex::{Complex32, Complex64};

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

// ── DFT-2 butterfly ───────────────────────────────────────────────────────────

/// In-place Winograd DFT-2 (butterfly).
///
/// **Contract**: `[a, b]` → `[a+b, a-b]` (forward/inverse sign-invariant).
///
/// No multiplications; one complex addition and one complex subtraction.
mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Scalar operations required by generic Winograd DFT helpers.
pub trait WinogradScalar:
    private::Sealed + num_traits::Float + num_traits::NumAssign + Send + Sync
{
    /// Convert an analytically defined f64 constant to this scalar precision.
    fn cast_f64(v: f64) -> Self;
    /// Return sqrt(2)/2 in this scalar precision.
    fn sq2o2() -> Self;
}
impl WinogradScalar for f64 {
    #[inline]
    fn cast_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn sq2o2() -> Self {
        std::f64::consts::SQRT_2 / 2.0
    }
}
impl WinogradScalar for f32 {
    #[inline]
    fn cast_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn sq2o2() -> Self {
        (std::f64::consts::SQRT_2 / 2.0) as f32
    }
}

#[inline]
pub(crate) fn dft2_impl<F: WinogradScalar>(
    a: &mut num_complex::Complex<F>,
    b: &mut num_complex::Complex<F>,
) {
    let tmp = *a;
    *a = tmp + *b;
    *b = tmp - *b;
}

// ── DFT-4 butterfly ───────────────────────────────────────────────────────────

/// In-place Winograd DFT-4.
///
/// **Contract** (forward, inverse=false):
/// ```text
/// y[0] = x[0] + x[1] + x[2] + x[3]
/// y[1] = x[0] - i·x[1] - x[2] + i·x[3]
/// y[2] = x[0] - x[1] + x[2] - x[3]
/// y[3] = x[0] + i·x[1] - x[2] - i·x[3]
/// ```
/// Inverse: replace `-i` ↔ `+i`.
///
/// **Multiplications**: 0 (all operations are ±1, ±i rotations ≡ swap+negate).
/// **Additions**: 8 complex (= 16 real).
///
/// Correctness reference: Cooley and Tukey (1965), 4-point special case.
#[inline]
pub(crate) fn dft4_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 4);
    let t0 = data[0] + data[2];
    let t1 = data[0] - data[2];
    let t2 = data[1] + data[3];
    let t3 = data[1] - data[3];
    data[0] = t0 + t2;
    data[2] = t0 - t2;
    let i_t3 = if inverse {
        num_complex::Complex::new(-t3.im, t3.re)
    } else {
        num_complex::Complex::new(t3.im, -t3.re)
    };
    data[1] = t1 + i_t3;
    data[3] = t1 - i_t3;
}

// ── DFT-8 butterfly ───────────────────────────────────────────────────────────

/// In-place Winograd DFT-8.
///
/// **Decomposition**: DFT-8 = stride-2 decimation-in-time into two DFT-4
/// sub-transforms followed by 8-point butterfly twiddle multiplications.
///
/// Twiddle factors for the 8-point stage (forward convention):
/// ```text
/// W_8^0 = 1
/// W_8^1 = (√2/2) - i·(√2/2) = SQ2O2·(1 - i)
/// W_8^2 = -i
/// W_8^3 = -(√2/2) - i·(√2/2)
/// ```
/// where `SQ2O2 = √2/2 ≈ std::f64::consts::FRAC_1_SQRT_2`.
///
/// **Multiplications**: 4 real (the ±SQ2O2 multiplications on the odd path).
/// All other twiddles are ×1 or ×(-i) / ×i, which are free sign/swap ops.
///
/// **Additions**: 26 real (Winograd 1978, Table 1, row N=8).
///
/// Correctness: Blahut (2010), §3.4, DFT-8 factoring.
#[inline]
pub(crate) fn dft8_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 8);
    let sq2o2 = F::sq2o2();
    let sign = if inverse {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
    let mut evens = [data[0], data[2], data[4], data[6]];
    let mut odds = [data[1], data[3], data[5], data[7]];
    dft4_impl(&mut evens, inverse);
    dft4_impl(&mut odds, inverse);
    let tw1 = num_complex::Complex::new(sq2o2, sign * sq2o2);
    let tw2 = num_complex::Complex::new(F::cast_f64(0.0), sign);
    let tw3 = num_complex::Complex::new(-sq2o2, sign * sq2o2);
    odds[1] *= tw1;
    odds[2] *= tw2;
    odds[3] *= tw3;
    for i in 0..4 {
        let e = evens[i];
        let o = odds[i];
        data[i] = e + o;
        data[i + 4] = e - o;
    }
}

#[inline]
pub(crate) fn dft7_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 7);
    let sign = if inverse {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
    let t = [
        data[0], data[1], data[2], data[3], data[4], data[5], data[6],
    ];
    for k in 0..7 {
        let mut sum = num_complex::Complex::new(F::cast_f64(0.0), F::cast_f64(0.0));
        for n in 0..7 {
            let angle = (k * n) as f64 * std::f64::consts::TAU / 7.0;
            let tw = num_complex::Complex::new(
                F::cast_f64(angle.cos()),
                sign * F::cast_f64(angle.sin()),
            );
            sum += t[n] * tw;
        }
        data[k] = sum;
    }
}

// ── AVX+FMA SIMD f32 DFT-4 and DFT-8 ────────────────────────────────────────

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

// ── DFT-3 butterfly ──────────────────────────────────────────────────────────

/// In-place DFT-3.
///
/// ## Mathematical derivation
///
/// For N=3, W₃ = exp(-2πi/3), the DFT matrix rows give:
/// ```text
/// Y[0] = X[0] + X[1] + X[2]
/// Y[1] = X[0] + W₃¹·X[1] + W₃²·X[2]   (fwd)
/// Y[2] = X[0] + W₃²·X[1] + W₃¹·X[2]   (fwd)
/// ```
/// With W₃¹ = −½ − i·(√3/2) and W₃² = −½ + i·(√3/2):
/// ```text
/// Y[1] = (X[0] − (X[1]+X[2])/2) − i·(√3/2)·(X[1]−X[2])
/// Y[2] = (X[0] − (X[1]+X[2])/2) + i·(√3/2)·(X[1]−X[2])
/// ```
/// Conjugate (flip sign on imaginary twiddle component) for inverse.
///
/// **Real multiplications**: 4 (two by C3=−½ on re/im of s, two by S3=√3/2
/// on re/im of id). Matches Winograd's lower bound for DFT-3.
/// **Complex additions**: 6.
///
/// References: Winograd (1978), Blahut (2010) §3.2.
#[inline]
pub(crate) fn dft3_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 3);
    let s = F::cast_f64(0.8660254037844386);
    let w_r = F::cast_f64(-0.5);
    let w_i = if inverse { s } else { -s };
    let t1 = data[1] + data[2];
    let m0 = data[0] + t1 * w_r;
    let m1 = (data[1] - data[2]) * num_complex::Complex::new(F::cast_f64(0.0), w_i);
    data[0] += t1;
    data[1] = m0 + m1;
    data[2] = m0 - m1;
}

// ── DFT-5 butterfly ──────────────────────────────────────────────────────────

/// In-place DFT-5.
///
/// ## Mathematical derivation
///
/// For N=5, W₅ = exp(−2πi/5). The symmetric index pairs (1,4) and (2,3)
/// allow the 5-point DFT to be expressed via sum/difference decomposition:
/// ```text
/// r₁ = X[1]+X[4],  d₁ = X[1]−X[4]
/// r₂ = X[2]+X[3],  d₂ = X[2]−X[3]
///
/// Y[0] = X[0] + r₁ + r₂
/// ar   = X[0] + c₁·r₁ + c₂·r₂       (cosine terms for Y[1],Y[4])
/// br   = X[0] + c₂·r₁ + c₁·r₂       (cosine terms for Y[2],Y[3])
/// id₁  = s₁·d₁ + s₂·d₂               (imaginary term for Y[1],Y[4])
/// id₂  = s₂·d₁ − s₁·d₂               (imaginary term for Y[2],Y[3])
///
/// Y[1] = ar − i·id₁   (fwd)    Y[4] = ar + i·id₁
/// Y[2] = br − i·id₂   (fwd)    Y[3] = br + i·id₂
/// ```
/// Inverse: flip sign of the imaginary rotation (−i ↔ +i).
///
/// Constants:
/// - c₁ = cos(2π/5) = (√5−1)/4 ≈ 0.30902
/// - c₂ = cos(4π/5) = −(√5+1)/4 ≈ −0.80902
/// - s₁ = sin(2π/5) ≈ 0.95106
/// - s₂ = sin(4π/5) ≈ 0.58779
///
/// **Real multiplications**: 8 (c₁,c₂ applied to r₁,r₂; s₁,s₂ applied to
/// d₁,d₂ — each scalar×complex costs 2 real muls). Standard minimal-form
/// derivation: Winograd (1978), Blahut (2010) §3.3.
/// **Complex additions**: 10.
#[inline]
pub(crate) fn dft5_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 5);
    let c1 = F::cast_f64(0.30901699437494745);
    let c2 = F::cast_f64(-0.8090169943749475);
    let s1 = F::cast_f64(0.9510565162951535);
    let s2 = F::cast_f64(0.5877852522924731);
    let sign = if inverse {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
    let s1 = s1 * sign;
    let s2 = s2 * sign;
    let t1 = data[1] + data[4];
    let t2 = data[1] - data[4];
    let t3 = data[2] + data[3];
    let t4 = data[2] - data[3];
    let m0 = data[0] + t1 + t3;
    let m1 = t1 * c1 + t3 * c2;
    let m2 = t1 * c2 + t3 * c1;
    let m3 = num_complex::Complex::new(F::cast_f64(0.0), F::cast_f64(1.0)) * (t2 * s1 + t4 * s2);
    let m4 = num_complex::Complex::new(F::cast_f64(0.0), F::cast_f64(1.0)) * (t2 * s2 - t4 * s1);
    let s1_add = data[0] + m1;
    let s2_add = data[0] + m2;
    data[0] = m0;
    data[1] = s1_add + m3;
    data[4] = s1_add - m3;
    data[2] = s2_add + m4;
    data[3] = s2_add - m4;
}

// ── SIMD DFT-5 fast path (AVX2) ──────────────────────────────────────────────

/// SIMD-accelerated DFT-5 for f64 using AVX2 (when available).
/// Falls back to scalar for non-AVX2 targets.
///
/// This processes one 5-element DFT using vectorized constant multiplications
// ── DFT-16 butterfly ─────────────────────────────────────────────────────────

/// Precomputed forward twiddle factors for the 16-point DFT stage.
///
/// `W_16^k = exp(-2πi·k/16)` for `k = 0..7`.
/// Stored as `(cos(2πk/16), -sin(2πk/16))`.
/// Only k=0..7 are needed (k=0 is trivial; k=1..7 are non-trivial).
fn twiddle16_fwd(k: usize) -> Complex64 {
    // Exact values for 16th roots of unity.
    match k {
        0 => Complex64::new(1.0, 0.0),
        1 => {
            // cos(π/8), -sin(π/8)
            let c = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(c, -s)
        }
        2 => {
            // cos(π/4), -sin(π/4)
            const SQ2O2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            Complex64::new(SQ2O2, -SQ2O2)
        }
        3 => {
            // cos(3π/8), -sin(3π/8)
            let c = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(c, -s)
        }
        4 => Complex64::new(0.0, -1.0),
        5 => {
            let c = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(-c, -s)
        }
        6 => {
            const SQ2O2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            Complex64::new(-SQ2O2, -SQ2O2)
        }
        7 => {
            let c = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(-c, -s)
        }
        _ => unreachable!(),
    }
}

fn twiddle16_inv(k: usize) -> Complex64 {
    let w = twiddle16_fwd(k);
    Complex64::new(w.re, -w.im)
}

fn cast_twiddle<F: WinogradScalar>(w: Complex64) -> num_complex::Complex<F> {
    num_complex::Complex::new(F::cast_f64(w.re), F::cast_f64(w.im))
}

/// In-place Winograd DFT-16.
///
/// **Decomposition**: DFT-16 = 2 × DFT-8 (stride-2 DIT) + 16-point twiddle
/// butterfly.
///
/// **Multiplications**: 2 × (DFT-8 ops) + 12 real twiddle mults (k=1..7
/// excluding k=0 trivial, k=4 = ×(-i) free, effectively 10 irrational mults).
///
/// Correctness: Van Loan (1992), §3.3.
#[inline]
pub(crate) fn dft16_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 16],
    inverse: bool,
) {
    // Step 1: two DFT-8 sub-transforms on even and odd sub-arrays.
    let mut even = [
        data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14],
    ];
    let mut odd = [
        data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15],
    ];
    dft8_impl(&mut even, inverse);
    dft8_impl(&mut odd, inverse);

    // Step 2: apply W_16^k twiddles to odd outputs and butterfly.
    for k in 0..8 {
        let tw = if inverse {
            cast_twiddle(twiddle16_inv(k))
        } else {
            cast_twiddle(twiddle16_fwd(k))
        };
        let o = apply_twiddle_impl(odd[k], tw);
        data[k] = even[k] + o;
        data[k + 8] = even[k] - o;
    }
}

// ── DFT-32 butterfly ─────────────────────────────────────────────────────────

const TWIDDLE32_FWD: [Complex64; 16] = [
    Complex64::new(1.0, 0.0),
    Complex64::new(0.9807852804032304, -0.19509032201612825),
    Complex64::new(0.9238795325112867, -0.3826834323650898),
    Complex64::new(0.8314696123025452, -0.5555702330196022),
    Complex64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
    Complex64::new(0.5555702330196023, -0.8314696123025452),
    Complex64::new(0.38268343236508984, -0.9238795325112867),
    Complex64::new(0.19509032201612833, -0.9807852804032304),
    Complex64::new(0.0, -1.0),
    Complex64::new(-0.1950903220161282, -0.9807852804032304),
    Complex64::new(-0.3826834323650897, -0.9238795325112867),
    Complex64::new(-0.555570233019602, -0.8314696123025455),
    Complex64::new(
        -std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
    Complex64::new(-0.8314696123025453, -0.5555702330196022),
    Complex64::new(-0.9238795325112867, -0.3826834323650899),
    Complex64::new(-0.9807852804032304, -0.1950903220161286),
];

#[inline]
fn twiddle32<F: WinogradScalar>(k: usize, inverse: bool) -> num_complex::Complex<F> {
    let w = TWIDDLE32_FWD[k];
    let w = if inverse {
        Complex64::new(w.re, -w.im)
    } else {
        w
    };
    cast_twiddle(w)
}

/// In-place Winograd DFT-32.
///
/// **Decomposition**: DFT-32 = 2 × DFT-16 (stride-2 DIT) + 32-point twiddle
/// butterfly.
///
/// **Multiplications**: 2 × (DFT-16 ops) + 28 twiddle mults (k=1..15 minus
/// the trivial/free roots).
///
/// Correctness: Van Loan (1992), §3.3 recursive formulation.
#[inline]
pub(crate) fn dft32_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 32],
    inverse: bool,
) {
    let mut even = [
        data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16],
        data[18], data[20], data[22], data[24], data[26], data[28], data[30],
    ];
    let mut odd = [
        data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15], data[17],
        data[19], data[21], data[23], data[25], data[27], data[29], data[31],
    ];
    dft16_impl(&mut even, inverse);
    dft16_impl(&mut odd, inverse);
    for k in 0..16 {
        let o = apply_twiddle_impl(odd[k], twiddle32(k, inverse));
        data[k] = even[k] + o;
        data[k + 16] = even[k] - o;
    }
}

// ── DFT-64 butterfly ─────────────────────────────────────────────────────────

#[inline]
fn twiddle64<F: WinogradScalar>(k: usize, inverse: bool) -> num_complex::Complex<F> {
    // W_64^(2m) = W_32^m and W_64^(2m+1) = W_32^m * W_64^1.
    // This avoids per-call trig evaluation and avoids lock checks in hot loops.
    let base = twiddle32(k >> 1, inverse);
    if (k & 1) == 0 {
        base
    } else {
        // cos(pi/32) ± i*sin(pi/32)
        let w1 = if inverse {
            Complex64::new(0.9951847266721969, 0.0980171403295606)
        } else {
            Complex64::new(0.9951847266721969, -0.0980171403295606)
        };
        apply_twiddle_impl(base, cast_twiddle(w1))
    }
}

/// In-place Winograd DFT-64.
///
/// **Decomposition**: DFT-64 = 2 × DFT-32 (stride-2 DIT) + 64-point twiddle
/// butterfly.
///
/// **Multiplications**: 2 × (DFT-32 ops) + 60 twiddle mults (k=1..31 minus
/// the trivial/free roots).
///
/// Correctness: Van Loan (1992), §3.3 recursive formulation.
#[inline]
pub(crate) fn dft64_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 64],
    inverse: bool,
) {
    let mut even = core::array::from_fn(|i| data[2 * i]);
    let mut odd = core::array::from_fn(|i| data[2 * i + 1]);
    dft32_impl(&mut even, inverse);
    dft32_impl(&mut odd, inverse);
    for k in 0..32 {
        let o = apply_twiddle_impl(odd[k], twiddle64(k, inverse));
        data[k] = even[k] + o;
        data[k + 32] = even[k] - o;
    }
}

// ── helpers used by radix kernels ─────────────────────────────────────────────

/// Apply `W_N^{k·j}` twiddle multiplication in-place.
/// Used by the radix outer loop to apply inter-group twiddles.
#[inline]
pub(crate) fn apply_twiddle_impl<F: WinogradScalar>(
    v: num_complex::Complex<F>,
    tw: num_complex::Complex<F>,
) -> num_complex::Complex<F> {
    num_complex::Complex::new(v.re * tw.re - v.im * tw.im, v.re * tw.im + v.im * tw.re)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};

    fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f64, f64::max)
    }

    // ── DFT-3 ────────────────────────────────────────────────────────────────

    #[test]
    fn dft3_forward_matches_direct() {
        let input: Vec<Complex64> = (0..3)
            .map(|k| Complex64::new((k as f64 * 0.71).sin(), (k as f64 * 0.43).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex64; 3] = input.as_slice().try_into().unwrap();
        dft3_impl(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-13, "DFT-3 forward max_err={err:.2e}");
    }

    #[test]
    fn dft3_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..3)
            .map(|k| Complex64::new((k as f64 * 0.55).cos(), (k as f64 * 0.19).sin()))
            .collect();
        let mut buf: [Complex64; 3] = input.as_slice().try_into().unwrap();
        dft3_impl(&mut buf, false);
        dft3_impl(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 3.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-13, "DFT-3 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft3_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..3)
            .map(|k| Complex64::new((k as f64 * 0.39).cos(), (k as f64 * 0.83).sin()))
            .collect();
        let expected_unnorm: Vec<Complex64> =
            dft_inverse(&input).into_iter().map(|x| x * 3.0).collect();
        let mut buf: [Complex64; 3] = input.as_slice().try_into().unwrap();
        dft3_impl(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-13, "DFT-3 inverse max_err={err:.2e}");
    }

    #[test]
    fn dft3_dc_produces_energy_in_bin0() {
        let mut buf = [Complex64::new(1.0, 0.0); 3];
        dft3_impl(&mut buf, false);
        assert!((buf[0] - Complex64::new(3.0, 0.0)).norm() < 1e-14);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-14, "non-zero bin: {:?}", x);
        }
    }

    // ── DFT-5 ────────────────────────────────────────────────────────────────

    #[test]
    fn dft5_forward_matches_direct() {
        let input: Vec<Complex64> = (0..5)
            .map(|k| Complex64::new((k as f64 * 0.61).sin(), (k as f64 * 0.37).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex64; 5] = input.as_slice().try_into().unwrap();
        dft5_impl(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-12, "DFT-5 forward max_err={err:.2e}");
    }

    #[test]
    fn dft5_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..5)
            .map(|k| Complex64::new((k as f64 * 0.47).cos(), (k as f64 * 0.28).sin()))
            .collect();
        let mut buf: [Complex64; 5] = input.as_slice().try_into().unwrap();
        dft5_impl(&mut buf, false);
        dft5_impl(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 5.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-12, "DFT-5 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft5_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..5)
            .map(|k| Complex64::new((k as f64 * 0.23).cos(), (k as f64 * 0.77).sin()))
            .collect();
        let expected_unnorm: Vec<Complex64> =
            dft_inverse(&input).into_iter().map(|x| x * 5.0).collect();
        let mut buf: [Complex64; 5] = input.as_slice().try_into().unwrap();
        dft5_impl(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-12, "DFT-5 inverse max_err={err:.2e}");
    }

    #[test]
    fn dft5_dc_produces_energy_in_bin0() {
        let mut buf = [Complex64::new(1.0, 0.0); 5];
        dft5_impl(&mut buf, false);
        assert!((buf[0] - Complex64::new(5.0, 0.0)).norm() < 1e-14);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-14, "non-zero bin: {:?}", x);
        }
    }

    #[test]
    fn dft5_f32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..5)
            .map(|k| Complex64::new((k as f64 * 0.53).sin(), (k as f64 * 0.31).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex32; 5] =
            core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
        dft5_impl(&mut buf, false);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 2e-6, "DFT-5 f32 forward max_err={err:.2e}");
    }

    // ── DFT-7 ────────────────────────────────────────────────────────────────

    #[test]
    fn dft7_forward_matches_direct() {
        let input: Vec<Complex64> = (0..7)
            .map(|k| Complex64::new((k as f64 * 0.71).sin(), (k as f64 * 0.31).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex64; 7] = input.as_slice().try_into().unwrap();
        dft7_impl(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-12, "DFT-7 forward max_err={err:.2e}");
    }

    #[test]
    fn dft7_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..7)
            .map(|k| Complex64::new((k as f64 * 0.37).cos(), (k as f64 * 0.19).sin()))
            .collect();
        let mut buf: [Complex64; 7] = input.as_slice().try_into().unwrap();
        dft7_impl(&mut buf, false);
        dft7_impl(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 7.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-12, "DFT-7 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft7_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..7)
            .map(|k| Complex64::new((k as f64 * 0.47).sin(), (k as f64 * 0.23).cos()))
            .collect();
        let expected_unnorm: Vec<Complex64> =
            dft_inverse(&input).into_iter().map(|x| x * 7.0).collect();
        let mut buf: [Complex64; 7] = input.as_slice().try_into().unwrap();
        dft7_impl(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-12, "DFT-7 inverse max_err={err:.2e}");
    }

    #[test]
    fn dft7_dc_produces_energy_in_bin0() {
        let mut buf = [Complex64::new(1.0, 0.0); 7];
        dft7_impl(&mut buf, false);
        assert!((buf[0] - Complex64::new(7.0, 0.0)).norm() < 1e-14);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-14, "non-zero bin: {:?}", x);
        }
    }

    #[test]
    fn dft7_f32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..7)
            .map(|k| Complex64::new((k as f64 * 0.53).sin(), (k as f64 * 0.31).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex32; 7] =
            core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
        dft7_impl(&mut buf, false);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 1e-5, "DFT-7 f32 forward max_err={err:.2e}");
    }

    // ── DFT-2 ────────────────────────────────────────────────────────────────

    #[test]
    fn dft2_forward_matches_direct() {
        let input = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let expected = dft_forward(&input);
        let mut a = input[0];
        let mut b = input[1];
        dft2_impl(&mut a, &mut b);
        assert!(
            max_err(&[a, b], &expected) < 1e-14,
            "DFT-2 forward mismatch"
        );
    }

    #[test]
    fn dft2_inverse_roundtrip() {
        let mut a = Complex64::new(3.0, -1.0);
        let mut b = Complex64::new(-2.0, 4.0);
        let orig_a = a;
        let orig_b = b;
        // forward then unnorm-inverse should give 2× the original.
        dft2_impl(&mut a, &mut b);
        dft2_impl(&mut a, &mut b);
        assert!((a - 2.0 * orig_a).norm() < 1e-14);
        assert!((b - 2.0 * orig_b).norm() < 1e-14);
    }

    // ── DFT-4 ────────────────────────────────────────────────────────────────

    #[test]
    fn dft4_forward_matches_direct() {
        let input: Vec<Complex64> = (0..4)
            .map(|k| Complex64::new((k as f64 * 0.3).sin(), (k as f64 * 0.7).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex64; 4] = input.as_slice().try_into().unwrap();
        dft4_impl(&mut buf, false);
        assert!(max_err(&buf, &expected) < 1e-13, "DFT-4 forward mismatch");
    }

    #[test]
    fn dft4_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..4)
            .map(|k| Complex64::new((k as f64 * 0.5).cos(), (k as f64 * 0.2).sin()))
            .collect();
        let mut buf: [Complex64; 4] = input.as_slice().try_into().unwrap();
        dft4_impl(&mut buf, false);
        dft4_impl(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 4.0).collect();
        assert!(
            max_err(&recovered, &input) < 1e-13,
            "DFT-4 roundtrip mismatch"
        );
    }

    #[test]
    fn dft4_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..4)
            .map(|k| Complex64::new((k as f64 * 0.9).cos(), (k as f64 * 0.4).sin()))
            .collect();
        let expected_unnorm: Vec<Complex64> =
            dft_inverse(&input).into_iter().map(|x| x * 4.0).collect();
        let mut buf: [Complex64; 4] = input.as_slice().try_into().unwrap();
        dft4_impl(&mut buf, true);
        assert!(
            max_err(&buf, &expected_unnorm) < 1e-13,
            "DFT-4 inverse mismatch"
        );
    }

    // ── DFT-8 ────────────────────────────────────────────────────────────────

    #[test]
    fn dft8_forward_matches_direct() {
        let input: Vec<Complex64> = (0..8)
            .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex64; 8] = input.as_slice().try_into().unwrap();
        dft8_impl(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-12, "DFT-8 forward max_err={err:.2e}");
    }

    #[test]
    fn dft8_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..8)
            .map(|k| Complex64::new((k as f64 * 0.23).cos(), -(k as f64 * 0.11).sin()))
            .collect();
        let mut buf: [Complex64; 8] = input.as_slice().try_into().unwrap();
        dft8_impl(&mut buf, false);
        dft8_impl(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 8.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-12, "DFT-8 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft8_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..8)
            .map(|k| Complex64::new((k as f64 * 0.33).sin(), (k as f64 * 0.22).cos()))
            .collect();
        let expected_unnorm: Vec<Complex64> =
            dft_inverse(&input).into_iter().map(|x| x * 8.0).collect();
        let mut buf: [Complex64; 8] = input.as_slice().try_into().unwrap();
        dft8_impl(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-12, "DFT-8 inverse max_err={err:.2e}");
    }

    #[test]
    fn dft8_f32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..8)
            .map(|k| Complex64::new((k as f64 * 0.18).sin(), (k as f64 * 0.31).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex32; 8] =
            core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
        dft8_impl(&mut buf, false);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 1e-5, "DFT-8 f32 forward max_err={err:.2e}");
    }

    // ── DFT-16 ───────────────────────────────────────────────────────────────

    #[test]
    fn dft16_forward_matches_direct() {
        let input: Vec<Complex64> = (0..16)
            .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.13).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
        dft16_impl(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-11, "DFT-16 forward max_err={err:.2e}");
    }

    #[test]
    fn dft16_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..16)
            .map(|k| Complex64::new((k as f64 * 0.06).cos(), (k as f64 * 0.19).sin()))
            .collect();
        let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
        dft16_impl(&mut buf, false);
        dft16_impl(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 16.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-11, "DFT-16 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft16_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..16)
            .map(|k| Complex64::new((k as f64 * 0.44).sin(), (k as f64 * 0.36).cos()))
            .collect();
        let expected_unnorm: Vec<Complex64> =
            dft_inverse(&input).into_iter().map(|x| x * 16.0).collect();
        let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
        dft16_impl(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-11, "DFT-16 inverse max_err={err:.2e}");
    }

    // ── DFT-32 ───────────────────────────────────────────────────────────────

    #[test]
    fn dft32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..32)
            .map(|k| Complex64::new((k as f64 * 0.21).sin(), (k as f64 * 0.09).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
        dft32_impl(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-11, "DFT-32 forward max_err={err:.2e}");
    }

    #[test]
    fn dft32_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..32)
            .map(|k| Complex64::new((k as f64 * 0.14).cos(), (k as f64 * 0.37).sin()))
            .collect();
        let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
        dft32_impl(&mut buf, false);
        dft32_impl(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 32.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-11, "DFT-32 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft32_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..32)
            .map(|k| Complex64::new((k as f64 * 0.55).sin(), (k as f64 * 0.27).cos()))
            .collect();
        let expected_unnorm: Vec<Complex64> =
            dft_inverse(&input).into_iter().map(|x| x * 32.0).collect();
        let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
        dft32_impl(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-11, "DFT-32 inverse max_err={err:.2e}");
    }

    // ── DFT-64 ───────────────────────────────────────────────────────────────

    #[test]
    fn dft64_forward_matches_direct() {
        let input: Vec<Complex64> = (0..64)
            .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.03).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
        dft64_impl(&mut buf, false);
        let err = max_err(&buf, &expected);
        assert!(err < 1e-11, "DFT-64 forward max_err={err:.2e}");
    }

    #[test]
    fn dft64_inverse_roundtrip() {
        let input: Vec<Complex64> = (0..64)
            .map(|k| Complex64::new((k as f64 * 0.08).cos(), (k as f64 * 0.51).sin()))
            .collect();
        let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
        dft64_impl(&mut buf, false);
        dft64_impl(&mut buf, true);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / 64.0).collect();
        let err = max_err(&recovered, &input);
        assert!(err < 1e-11, "DFT-64 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn dft64_inverse_matches_direct() {
        let input: Vec<Complex64> = (0..64)
            .map(|k| Complex64::new((k as f64 * 0.14).sin(), (k as f64 * 0.42).cos()))
            .collect();
        let expected_unnorm: Vec<Complex64> =
            dft_inverse(&input).into_iter().map(|x| x * 64.0).collect();
        let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
        dft64_impl(&mut buf, true);
        let err = max_err(&buf, &expected_unnorm);
        assert!(err < 1e-11, "DFT-64 inverse max_err={err:.2e}");
    }

    // ── boundary cases ───────────────────────────────────────────────────────

    #[test]
    fn dft4_impulse_produces_all_ones() {
        // DFT([1,0,0,0]) = [1,1,1,1]  (Cooley & Tukey 1965)
        let mut buf = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        dft4_impl(&mut buf, false);
        for x in &buf {
            assert!((x - Complex64::new(1.0, 0.0)).norm() < 1e-14);
        }
    }

    #[test]
    fn dft8_dc_produces_energy_in_bin0() {
        // DFT([1,1,1,1,1,1,1,1]) = [8,0,0,0,0,0,0,0]
        let mut buf = [Complex64::new(1.0, 0.0); 8];
        dft8_impl(&mut buf, false);
        assert!((buf[0] - Complex64::new(8.0, 0.0)).norm() < 1e-12);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-12);
        }
    }

    #[test]
    fn dft16_dc_produces_energy_in_bin0() {
        let mut buf = [Complex64::new(1.0, 0.0); 16];
        dft16_impl(&mut buf, false);
        assert!((buf[0] - Complex64::new(16.0, 0.0)).norm() < 1e-11);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-11);
        }
    }

    #[test]
    fn dft32_dc_produces_energy_in_bin0() {
        let mut buf = [Complex64::new(1.0, 0.0); 32];
        dft32_impl(&mut buf, false);
        assert!((buf[0] - Complex64::new(32.0, 0.0)).norm() < 1e-11);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-11);
        }
    }

    #[test]
    fn dft64_dc_produces_energy_in_bin0() {
        let mut buf = [Complex64::new(1.0, 0.0); 64];
        dft64_impl(&mut buf, false);
        assert!((buf[0] - Complex64::new(64.0, 0.0)).norm() < 1e-11);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-11);
        }
    }

    #[test]
    fn dft32_f32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..32)
            .map(|k| Complex64::new((k as f64 * 0.12).sin(), (k as f64 * 0.35).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex32; 32] =
            core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
        dft32_impl(&mut buf, false);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 2e-5, "DFT-32 f32 forward max_err={err:.2e}");
    }

    #[test]
    fn dft64_f32_forward_matches_direct() {
        let input: Vec<Complex64> = (0..64)
            .map(|k| Complex64::new((k as f64 * 0.07).sin(), (k as f64 * 0.29).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: [Complex32; 64] =
            core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
        dft64_impl(&mut buf, false);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 3e-5, "DFT-64 f32 forward max_err={err:.2e}");
    }
}
