use num_complex::Complex;
/// AVX2+FMA flat Stockham pass for radix-4 (f64).
///
/// Invoked ONCE PER STAGE (not per group) by `flat_stockham_fused` in `core.rs`.
/// This eliminates the per-group `#[target_feature]` function-call overhead that
/// caused the previous AVX2 attempt to regress (275 K cycles overhead for 13,770
/// per-group calls). With the flat loop, the `#[target_feature]` call frame is
/// amortized over all `g_count` groups: O(8) calls per transform for M=20736.
///
/// # Safety
/// Caller must verify AVX2 and FMA support before calling.
///
/// # Addressing
///
/// ## prev_len == 1  (first radix-4 pass, no twiddles)
/// - stride = g_count (= n/4 for r=4)
/// - arm k block: src[k*stride .. (k+1)*stride]
/// - Adjacent groups g,g+1 share a 256-bit AVX2 load: `_mm256_loadu_pd(&src[k*stride + g])`
/// - Outputs g and g+1 are contiguous 4-element blocks in dst.
///
/// ## prev_len >= 2  (later radix-4 passes, twiddles required)
/// - stride = g_count * prev_len = n/4 (constant for r=4)
/// - Within group g, adjacent columns j,j+1 are contiguous in both src and dst.
/// - Loads: `_mm256_loadu_pd(&src[k*stride + g*prev_len + j])` ← 2 complex at once
/// - Twiddle loads: `_mm256_loadu_pd(&tw[(k-1)*prev_len + j])` ← adjacent in table
/// - Stores: `_mm256_storeu_pd(&dst_block[j + k*prev_len])` ← 2 complex contiguous
///
/// # DFT-4 butterfly (forward)
/// t0 = a0+a2, t1 = a0-a2, t2 = a1+a3, t3 = a1-a3
/// b0 = t0+t2,  b1 = t1-i·t3,  b2 = t0-t2,  b3 = t1+i·t3
/// (inverse: swap the ±i sign)
use std::arch::x86_64::*;

// ─────────────────────────────────────────────────────────────────────────────
// f64 AVX2+FMA flat Stockham pass for radix-3.
//
// DFT-3 butterfly (from dft3_impl):
//   sum = x1 + x2,  diff = x1 - x2
//   b0  = x0 + sum
//   m0  = x0 + sum * (-0.5)          [fused: fmadd(sum, -0.5, x0)]
//   m1  = rot_neg_i(diff) * s        [forward;  s = √3/2]
//         rot_pos_i(diff) * s        [inverse]
//   b1  = m0 + m1
//   b2  = m0 - m1
//
// Cost: 4 real multiplies, 6 complex adds — minimal for radix-3.
//
// ## Addressing (prev_len == 1, no twiddles)
//   stride = g_count = n/3.
//   2 adjacent groups per __m256d: _mm256_loadu_pd(&src[k*stride + g]).
//   Stores: [b0g0,b1g0] via _mm256_set_m128d; then b2g0 via _mm_storeu_pd.
//
// ## Addressing (prev_len >= 2, twiddles required)
//   stride = g_count * prev_len = n/3.
//   2 adjacent columns per __m256d; twiddle layout matches r=4:
//     arm 1 at tw[0*prev_len + j], arm 2 at tw[1*prev_len + j].
// ─────────────────────────────────────────────────────────────────────────────

/// Flat radix-3 Stockham pass in AVX2+FMA for f64. Processes all `g_count` groups.
///
/// `tw` has `(R-1) × prev_len = 2 × prev_len` entries:
///   arm 1 at `[0*prev_len .. 1*prev_len)`, arm 2 at `[1*prev_len .. 2*prev_len)`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn flat_pass_r3_f64(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    prev_len: usize,
    g_count: usize,
    stage_chunk: usize,
    tw: &[Complex<f64>],
    pointwise: Option<&[Complex<f64>]>,
    inverse: bool,
) {
    let stride = g_count * prev_len; // = n/3 for r=3
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        // No twiddles (W^0 = 1). 2 groups per AVX2 iteration.
        let mut g = 0usize;
        while g + 1 < g_count {
            let a0 = _mm256_loadu_pd(src_ptr.add(0 * stride + g) as *const f64);
            let a1 = _mm256_loadu_pd(src_ptr.add(1 * stride + g) as *const f64);
            let a2 = _mm256_loadu_pd(src_ptr.add(2 * stride + g) as *const f64);

            // b0 = x0+sum, b1 = m0+m1, b2 = m0-m1; returned as (b1, b2, b0).
            let wr = _mm256_set1_pd(-0.5);
            let s = _mm256_set1_pd(0.8660254037844386_f64);
            let sum = _mm256_add_pd(a1, a2);
            let diff = _mm256_sub_pd(a1, a2);
            let b0 = _mm256_add_pd(a0, sum);
            let m0 = _mm256_fmadd_pd(sum, wr, a0);
            let m1_dir = if inverse {
                rot_pos_i(diff)
            } else {
                rot_neg_i(diff)
            };
            let m1 = _mm256_mul_pd(m1_dir, s);
            let b1 = _mm256_add_pd(m0, m1);
            let b2 = _mm256_sub_pd(m0, m1);

            // b0 = [b0g0, b0g1], b1 = [b1g0, b1g1], b2 = [b2g0, b2g1].
            // Store group g: [b0g0, b1g0] as 256 bits at dst[g*3], b2g0 at dst[g*3+2].
            // Store group g+1: [b0g1, b1g1] at dst[(g+1)*3], b2g1 at dst[(g+1)*3+2].
            let b0_lo = _mm256_castpd256_pd128(b0);
            let b1_lo = _mm256_castpd256_pd128(b1);
            let b2_lo = _mm256_castpd256_pd128(b2);
            let b0_hi = _mm256_extractf128_pd(b0, 1);
            let b1_hi = _mm256_extractf128_pd(b1, 1);
            let b2_hi = _mm256_extractf128_pd(b2, 1);

            // Group g
            let dg = dst_ptr.add(g * 3);
            _mm256_storeu_pd(dg as *mut f64, _mm256_set_m128d(b1_lo, b0_lo)); // [b0g0, b1g0]
            _mm_storeu_pd(dg.add(2) as *mut f64, b2_lo); // b2g0

            // Group g+1
            let dg1 = dst_ptr.add((g + 1) * 3);
            _mm256_storeu_pd(dg1 as *mut f64, _mm256_set_m128d(b1_hi, b0_hi));
            _mm_storeu_pd(dg1.add(2) as *mut f64, b2_hi);

            g += 2;
        }
        // Scalar tail
        if g < g_count {
            let a0s = *src_ptr.add(0 * stride + g);
            let a1s = *src_ptr.add(1 * stride + g);
            let a2s = *src_ptr.add(2 * stride + g);
            scalar_dft3(a0s, a1s, a2s, dst_ptr.add(g * 3), inverse);
        }
    } else {
        // prev_len >= 2: twiddles applied.
        //
        // Width-4 sub-loop (j += 4, prev_len >= 4): processes 2 independent
        // butterfly chains per iteration, each on 2 adjacent complex-f64 columns.
        // Uses up to 14 YMM registers (6 data × 2 + 2 twiddle × 2 = 14),
        // enabling dual-FMA-unit parallelism on OOO CPUs.
        // Falls through to the width-2 loop for the remaining 0–3 columns.
        //
        // Hoist loop-invariant constants: wr/s are constant for all groups,
        // and tw_ptr is the base pointer that never changes.
        let wr = _mm256_set1_pd(-0.5);
        let s = _mm256_set1_pd(0.8660254037844386_f64);
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;

            // ── Width-4 loop ──────────────────────────────────────────────────
            while j + 3 < prev_len {
                let off0a = 0 * stride + src_base + j;
                let off1a = 1 * stride + src_base + j;
                let off2a = 2 * stride + src_base + j;
                // lo: columns j, j+1
                let a0_lo = _mm256_loadu_pd(src_ptr.add(off0a) as *const f64);
                let a1_lo = _mm256_loadu_pd(src_ptr.add(off1a) as *const f64);
                let a2_lo = _mm256_loadu_pd(src_ptr.add(off2a) as *const f64);
                // hi: columns j+2, j+3
                let a0_hi = _mm256_loadu_pd(src_ptr.add(off0a + 2) as *const f64);
                let a1_hi_raw = _mm256_loadu_pd(src_ptr.add(off1a + 2) as *const f64);
                let a2_hi_raw = _mm256_loadu_pd(src_ptr.add(off2a + 2) as *const f64);
                // Twiddles for lo/hi (tw_ptr hoisted outside loops):
                let tw1_lo = _mm256_loadu_pd(tw_ptr.add(0 * prev_len + j) as *const f64);
                let tw2_lo = _mm256_loadu_pd(tw_ptr.add(1 * prev_len + j) as *const f64);
                let tw1_hi = _mm256_loadu_pd(tw_ptr.add(0 * prev_len + j + 2) as *const f64);
                let tw2_hi = _mm256_loadu_pd(tw_ptr.add(1 * prev_len + j + 2) as *const f64);
                // Prefetch twiddle data 2 iterations ahead for L1 cache.
                // Only beneficial when prev_len is large enough to have cache pressure.
                // Cast to *const i8 as required by _mm_prefetch intrinsic.
                if prev_len >= 16 {
                    _mm_prefetch(tw_ptr.add(0 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(1 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                }
                // Apply twiddles:
                let a1_lo_tw = cmul(a1_lo, tw1_lo);
                let a2_lo_tw = cmul(a2_lo, tw2_lo);
                let a1_hi = cmul(a1_hi_raw, tw1_hi);
                let a2_hi = cmul(a2_hi_raw, tw2_hi);
                // DFT-3 chain lo (independent of hi):
                let sum_lo = _mm256_add_pd(a1_lo_tw, a2_lo_tw);
                let diff_lo = _mm256_sub_pd(a1_lo_tw, a2_lo_tw);
                let b0_lo = _mm256_add_pd(a0_lo, sum_lo);
                let m0_lo = _mm256_fmadd_pd(sum_lo, wr, a0_lo);
                let m1_dir_lo = if inverse {
                    rot_pos_i(diff_lo)
                } else {
                    rot_neg_i(diff_lo)
                };
                let m1_lo = _mm256_mul_pd(m1_dir_lo, s);
                let b1_lo = _mm256_add_pd(m0_lo, m1_lo);
                let b2_lo = _mm256_sub_pd(m0_lo, m1_lo);
                // DFT-3 chain hi (independent of lo):
                let sum_hi = _mm256_add_pd(a1_hi, a2_hi);
                let diff_hi = _mm256_sub_pd(a1_hi, a2_hi);
                let b0_hi = _mm256_add_pd(a0_hi, sum_hi);
                let m0_hi = _mm256_fmadd_pd(sum_hi, wr, a0_hi);
                let m1_dir_hi = if inverse {
                    rot_pos_i(diff_hi)
                } else {
                    rot_neg_i(diff_hi)
                };
                let m1_hi = _mm256_mul_pd(m1_dir_hi, s);
                let b1_hi = _mm256_add_pd(m0_hi, m1_hi);
                let b2_hi = _mm256_sub_pd(m0_hi, m1_hi);
                // Stores (6 __m256d):
                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_pd(dp.add(j + 0 * prev_len) as *mut f64, b0_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 0 * prev_len) as *mut f64, b0_hi);
                _mm256_storeu_pd(dp.add(j + 1 * prev_len) as *mut f64, b1_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 1 * prev_len) as *mut f64, b1_hi);
                _mm256_storeu_pd(dp.add(j + 2 * prev_len) as *mut f64, b2_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 2 * prev_len) as *mut f64, b2_hi);
                j += 4;
            }

            // ── Width-2 loop (handles remaining 0–3 columns) ─────────────────
            while j + 1 < prev_len {
                let off0 = 0 * stride + src_base + j;
                let off1 = 1 * stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let a0 = _mm256_loadu_pd(src_ptr.add(off0) as *const f64);
                let a1_raw = _mm256_loadu_pd(src_ptr.add(off1) as *const f64);
                let a2_raw = _mm256_loadu_pd(src_ptr.add(off2) as *const f64);
                // Twiddles (tw_ptr hoisted outside loops): arm 1 at tw[0*prev_len+j], arm 2 at tw[1*prev_len+j].
                let tw1 = _mm256_loadu_pd(tw_ptr.add(0 * prev_len + j) as *const f64);
                let tw2 = _mm256_loadu_pd(tw_ptr.add(1 * prev_len + j) as *const f64);
                let a1 = cmul(a1_raw, tw1);
                let a2 = cmul(a2_raw, tw2);

                let sum = _mm256_add_pd(a1, a2);
                let diff = _mm256_sub_pd(a1, a2);
                let b0 = _mm256_add_pd(a0, sum);
                let m0 = _mm256_fmadd_pd(sum, wr, a0);
                let m1_dir = if inverse {
                    rot_pos_i(diff)
                } else {
                    rot_neg_i(diff)
                };
                let m1 = _mm256_mul_pd(m1_dir, s);
                let b1 = _mm256_add_pd(m0, m1);
                let b2 = _mm256_sub_pd(m0, m1);

                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_pd(dp.add(j + 0 * prev_len) as *mut f64, b0);
                _mm256_storeu_pd(dp.add(j + 1 * prev_len) as *mut f64, b1);
                _mm256_storeu_pd(dp.add(j + 2 * prev_len) as *mut f64, b2);
                j += 2;
            }
            if j < prev_len {
                let src_j = src_base + j;
                let a0s = *src_ptr.add(0 * stride + src_j);
                let mut a1s = *src_ptr.add(1 * stride + src_j);
                let mut a2s = *src_ptr.add(2 * stride + src_j);
                if j > 0 {
                    let tw1 = *tw.as_ptr().add(0 * prev_len + j);
                    let tw2 = *tw.as_ptr().add(1 * prev_len + j);
                    a1s = Complex {
                        re: a1s.re * tw1.re - a1s.im * tw1.im,
                        im: a1s.re * tw1.im + a1s.im * tw1.re,
                    };
                    a2s = Complex {
                        re: a2s.re * tw2.re - a2s.im * tw2.im,
                        im: a2s.re * tw2.im + a2s.im * tw2.re,
                    };
                }
                // Column-major output: arm k at dp[j + k*prev_len].
                let dp = dst_ptr.add(dst_base);
                let s: f64 = 0.8660254037844386;
                let wr: f64 = -0.5;
                let sum_re = a1s.re + a2s.re;
                let sum_im = a1s.im + a2s.im;
                let diff_re = a1s.re - a2s.re;
                let diff_im = a1s.im - a2s.im;
                let m0_re = a0s.re + sum_re * wr;
                let m0_im = a0s.im + sum_im * wr;
                let (m1_re, m1_im) = if inverse {
                    (-diff_im * s, diff_re * s)
                } else {
                    (diff_im * s, -diff_re * s)
                };
                *dp.add(j + 0 * prev_len) = Complex {
                    re: a0s.re + sum_re,
                    im: a0s.im + sum_im,
                };
                *dp.add(j + 1 * prev_len) = Complex {
                    re: m0_re + m1_re,
                    im: m0_im + m1_im,
                };
                *dp.add(j + 2 * prev_len) = Complex {
                    re: m0_re - m1_re,
                    im: m0_im - m1_im,
                };
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f64(dst_ptr, pw.as_ptr(), dst.len());
    }
}

/// Scalar DFT-3 butterfly: reads a0,a1,a2 and writes b0,b1,b2 to `out[0..3]`.
#[inline]
unsafe fn scalar_dft3(
    a0: Complex<f64>,
    a1: Complex<f64>,
    a2: Complex<f64>,
    out: *mut Complex<f64>,
    inverse: bool,
) {
    let s: f64 = 0.8660254037844386;
    let wr: f64 = -0.5;
    let sum_re = a1.re + a2.re;
    let sum_im = a1.im + a2.im;
    let diff_re = a1.re - a2.re;
    let diff_im = a1.im - a2.im;
    let m0_re = a0.re + sum_re * wr;
    let m0_im = a0.im + sum_im * wr;
    let (m1_re, m1_im) = if inverse {
        (-diff_im * s, diff_re * s)
    } else {
        (diff_im * s, -diff_re * s)
    };
    *out.add(0) = Complex {
        re: a0.re + sum_re,
        im: a0.im + sum_im,
    };
    *out.add(1) = Complex {
        re: m0_re + m1_re,
        im: m0_im + m1_im,
    };
    *out.add(2) = Complex {
        re: m0_re - m1_re,
        im: m0_im - m1_im,
    };
}

/// Complex multiply: (a_re + i·a_im) × (b_re + i·b_im) for 2 complex-f64 pairs.
///
/// Uses the fmaddsub pattern:
///   a_re = broadcast_re(a), a_im = broadcast_im(a), b_sw = swap_re_im(b)
///   result = fmaddsub(a_re, b, a_im * b_sw)
///          = [a_re·b_re - a_im·b_im, a_re·b_im + a_im·b_re, ...]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn cmul(a: __m256d, b: __m256d) -> __m256d {
    let a_re = _mm256_permute_pd(a, 0b0000); // [re0, re0, re1, re1]
    let a_im = _mm256_permute_pd(a, 0b1111); // [im0, im0, im1, im1]
    let b_sw = _mm256_permute_pd(b, 0b0101); // [im0, re0, im1, re1]
    _mm256_fmaddsub_pd(a_re, b, _mm256_mul_pd(a_im, b_sw))
}

/// Multiply v by -i: (re+i·im) → (im, -re).
/// permute(v, 0b0101) = [im0, re0, im1, re1]; XOR negates re positions (1, 3).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn rot_neg_i(v: __m256d) -> __m256d {
    // sign mask: negate elements 1,3 (the `re` positions after permute)
    // _mm256_set_pd(e3, e2, e1, e0): e0 at position 0
    let sign = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    _mm256_xor_pd(_mm256_permute_pd(v, 0b0101), sign)
}

/// Multiply v by +i: (re+i·im) → (-im, re).
/// permute(v, 0b0101) = [im0, re0, im1, re1]; XOR negates im positions (0, 2).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn rot_pos_i(v: __m256d) -> __m256d {
    let sign = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
    _mm256_xor_pd(_mm256_permute_pd(v, 0b0101), sign)
}

/// Radix-4 butterfly for 2 complex-f64 pairs simultaneously.
/// Returns (b0, b1, b2, b3) given arms (a0, a1, a2, a3).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft4_f64(
    a0: __m256d,
    a1: __m256d,
    a2: __m256d,
    a3: __m256d,
    inverse: bool,
) -> (__m256d, __m256d, __m256d, __m256d) {
    let t0 = _mm256_add_pd(a0, a2);
    let t1 = _mm256_sub_pd(a0, a2);
    let t2 = _mm256_add_pd(a1, a3);
    let t3 = _mm256_sub_pd(a1, a3);
    if inverse {
        let r = rot_pos_i(t3); // +i·t3
        (
            _mm256_add_pd(t0, t2),
            _mm256_add_pd(t1, r),
            _mm256_sub_pd(t0, t2),
            _mm256_sub_pd(t1, r),
        )
    } else {
        let r = rot_neg_i(t3); // -i·t3
        (
            _mm256_add_pd(t0, t2),
            _mm256_add_pd(t1, r),
            _mm256_sub_pd(t0, t2),
            _mm256_sub_pd(t1, r),
        )
    }
}

/// Apply pointwise frequency-domain multiply to dst (inline after butterfly).
/// `pw` covers `dst[0..dst.len()]` element-wise. Used on the final stage only.
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_pointwise_f64(dst: *mut Complex<f64>, pw: *const Complex<f64>, len: usize) {
    let mut i = 0usize;
    while i + 1 < len {
        let d = _mm256_loadu_pd(dst.add(i) as *const f64);
        let p = _mm256_loadu_pd(pw.add(i) as *const f64);
        _mm256_storeu_pd(dst.add(i) as *mut f64, cmul(d, p));
        i += 2;
    }
    if i < len {
        let d0 = *dst.add(i);
        let p0 = *pw.add(i);
        *dst.add(i) = Complex {
            re: d0.re * p0.re - d0.im * p0.im,
            im: d0.re * p0.im + d0.im * p0.re,
        };
    }
}

/// Flat radix-4 Stockham pass in AVX2+FMA for f64. Processes all `g_count` groups.
///
/// `src` and `dst` are the full n-element arrays for this pass (length n = g_count × stage_chunk).
/// `tw` is the twiddle slice for this stage: `(R-1) × prev_len` entries, arm k at `[(k-1)*prev_len..]`.
///
/// # Layout invariants
/// - stage_chunk = 4 × prev_len
/// - stride = g_count × prev_len = n / 4  (constant for r=4 regardless of prev_len)
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn flat_pass_r4_f64(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    prev_len: usize,
    g_count: usize,
    stage_chunk: usize,
    tw: &[Complex<f64>],
    pointwise: Option<&[Complex<f64>]>,
    inverse: bool,
) {
    let stride = g_count * prev_len; // = n/4 for r=4
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        // No twiddles (j=0 always, W^0=1). Process 2 groups per AVX2 iteration.
        // arm k block starts at src_ptr + k*stride.
        // Groups g and g+1 are adjacent: _mm256_loadu_pd at &src[k*stride + g].
        let mut g = 0usize;
        while g + 1 < g_count {
            let a0 = _mm256_loadu_pd(src_ptr.add(0 * stride + g) as *const f64);
            let a1 = _mm256_loadu_pd(src_ptr.add(1 * stride + g) as *const f64);
            let a2 = _mm256_loadu_pd(src_ptr.add(2 * stride + g) as *const f64);
            let a3 = _mm256_loadu_pd(src_ptr.add(3 * stride + g) as *const f64);

            let (b0, b1, b2, b3) = dft4_f64(a0, a1, a2, a3, inverse);

            // b0 = [group_g.arm0, group_g+1.arm0] — extract lo/hi 128-bit lanes.
            // Group g output: dst[g*4 .. g*4+4]; group g+1: dst[(g+1)*4 .. (g+1)*4+4].
            let d0 = dst_ptr.add(g * 4);
            _mm_storeu_pd(d0.add(0) as *mut f64, _mm256_castpd256_pd128(b0));
            _mm_storeu_pd(d0.add(1) as *mut f64, _mm256_castpd256_pd128(b1));
            _mm_storeu_pd(d0.add(2) as *mut f64, _mm256_castpd256_pd128(b2));
            _mm_storeu_pd(d0.add(3) as *mut f64, _mm256_castpd256_pd128(b3));
            let d1 = dst_ptr.add((g + 1) * 4);
            _mm_storeu_pd(d1.add(0) as *mut f64, _mm256_extractf128_pd(b0, 1));
            _mm_storeu_pd(d1.add(1) as *mut f64, _mm256_extractf128_pd(b1, 1));
            _mm_storeu_pd(d1.add(2) as *mut f64, _mm256_extractf128_pd(b2, 1));
            _mm_storeu_pd(d1.add(3) as *mut f64, _mm256_extractf128_pd(b3, 1));

            g += 2;
        }
        // Scalar tail (odd g_count).
        if g < g_count {
            let src_base = g;
            let d = dst_ptr.add(g * 4);
            let a0s = *src_ptr.add(0 * stride + src_base);
            let a1s = *src_ptr.add(1 * stride + src_base);
            let a2s = *src_ptr.add(2 * stride + src_base);
            let a3s = *src_ptr.add(3 * stride + src_base);
            let t0 = a0s + a2s;
            let t1 = a0s - a2s;
            let t2 = a1s + a3s;
            let t3 = a1s - a3s;
            let (it3_re, it3_im) = if inverse {
                (-t3.im, t3.re)
            } else {
                (t3.im, -t3.re)
            };
            let it3 = Complex {
                re: it3_re,
                im: it3_im,
            };
            *d.add(0) = t0 + t2;
            *d.add(1) = t1 + it3;
            *d.add(2) = t0 - t2;
            *d.add(3) = t1 - it3;
        }
    } else {
        // prev_len >= 2: twiddles applied.
        //
        // Width-4 sub-loop (j += 4, prev_len >= 4): 2 independent DFT-4 chains
        // per iteration (lo = cols j,j+1; hi = cols j+2,j+3). Each chain uses
        // 4 arms × 1 __m256d + 3 twiddles × 1 __m256d = 7 YMM registers.
        // Both chains together use 14 YMM registers (fits x86's 16 YMM file),
        // enabling dual-FMA-unit parallelism on OOO CPUs.
        // Falls through to width-2 loop for remaining 0–3 columns.
        //
        // Hoist invariants from inner loops: tw_ptr never changes, and wr/s
        // are constant for the DFT-3 butterfly.
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;

            let mut j = 0usize;

            // ── Width-4 loop ──────────────────────────────────────────────────
            while j + 3 < prev_len {
                let off0 = 0 * stride + src_base + j;
                let off1 = 1 * stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;
                // lo: cols j, j+1
                let a0_lo = _mm256_loadu_pd(src_ptr.add(off0) as *const f64);
                let a1_lo_raw = _mm256_loadu_pd(src_ptr.add(off1) as *const f64);
                let a2_lo_raw = _mm256_loadu_pd(src_ptr.add(off2) as *const f64);
                let a3_lo_raw = _mm256_loadu_pd(src_ptr.add(off3) as *const f64);
                // hi: cols j+2, j+3
                let a0_hi = _mm256_loadu_pd(src_ptr.add(off0 + 2) as *const f64);
                let a1_hi_raw = _mm256_loadu_pd(src_ptr.add(off1 + 2) as *const f64);
                let a2_hi_raw = _mm256_loadu_pd(src_ptr.add(off2 + 2) as *const f64);
                let a3_hi_raw = _mm256_loadu_pd(src_ptr.add(off3 + 2) as *const f64);
                // Twiddles (tw_ptr hoisted outside loop):
                let tw1_lo = _mm256_loadu_pd(tw_ptr.add(0 * prev_len + j) as *const f64);
                let tw2_lo = _mm256_loadu_pd(tw_ptr.add(1 * prev_len + j) as *const f64);
                let tw3_lo = _mm256_loadu_pd(tw_ptr.add(2 * prev_len + j) as *const f64);
                let tw1_hi = _mm256_loadu_pd(tw_ptr.add(0 * prev_len + j + 2) as *const f64);
                let tw2_hi = _mm256_loadu_pd(tw_ptr.add(1 * prev_len + j + 2) as *const f64);
                let tw3_hi = _mm256_loadu_pd(tw_ptr.add(2 * prev_len + j + 2) as *const f64);
                // Prefetch twiddle data 2 iterations ahead for L1 cache.
                // This helps for large prev_len where cache lines may be evicted.
                // _MM_HINT_T0 loads into L1 dcache, which is what we want for data we'll use shortly.
                // Cast to *const i8 as required by _mm_prefetch intrinsic.
                if prev_len >= 16 {
                    _mm_prefetch(tw_ptr.add(0 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(1 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(2 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                }
                // Apply twiddles to arms 1,2,3 for lo/hi chains:
                let a1_lo = cmul(a1_lo_raw, tw1_lo);
                let a2_lo = cmul(a2_lo_raw, tw2_lo);
                let a3_lo = cmul(a3_lo_raw, tw3_lo);
                let a1_hi = cmul(a1_hi_raw, tw1_hi);
                let a2_hi = cmul(a2_hi_raw, tw2_hi);
                let a3_hi = cmul(a3_hi_raw, tw3_hi);
                // DFT-4 lo chain (independent of hi):
                let (b0_lo, b1_lo, b2_lo, b3_lo) = dft4_f64(a0_lo, a1_lo, a2_lo, a3_lo, inverse);
                // DFT-4 hi chain (OOO-parallel with lo):
                let (b0_hi, b1_hi, b2_hi, b3_hi) = dft4_f64(a0_hi, a1_hi, a2_hi, a3_hi, inverse);
                // Stores (8 __m256d):
                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_pd(dp.add(j + 0 * prev_len) as *mut f64, b0_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 0 * prev_len) as *mut f64, b0_hi);
                _mm256_storeu_pd(dp.add(j + 1 * prev_len) as *mut f64, b1_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 1 * prev_len) as *mut f64, b1_hi);
                _mm256_storeu_pd(dp.add(j + 2 * prev_len) as *mut f64, b2_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 2 * prev_len) as *mut f64, b2_hi);
                _mm256_storeu_pd(dp.add(j + 3 * prev_len) as *mut f64, b3_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 3 * prev_len) as *mut f64, b3_hi);
                j += 4;
            }

            // ── Width-2 loop (handles remaining 0–3 columns) ─────────────────
            while j + 1 < prev_len {
                let off0 = 0 * stride + src_base + j;
                let off1 = 1 * stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;

                let a0 = _mm256_loadu_pd(src_ptr.add(off0) as *const f64);
                let mut a1 = _mm256_loadu_pd(src_ptr.add(off1) as *const f64);
                let mut a2 = _mm256_loadu_pd(src_ptr.add(off2) as *const f64);
                let mut a3 = _mm256_loadu_pd(src_ptr.add(off3) as *const f64);

                // Apply twiddles for arms 1, 2, 3. Arm k twiddles at tw[(k-1)*prev_len + j].
                // tw[0] is 1.0 + i*0.0, so col 0 is correctly multiplied by 1 while col 1 gets W^1.
                // tw_ptr hoisted outside loops above — use it here for consistency.
                let tw1 = _mm256_loadu_pd(tw_ptr.add(0 * prev_len + j) as *const f64);
                let tw2 = _mm256_loadu_pd(tw_ptr.add(1 * prev_len + j) as *const f64);
                let tw3 = _mm256_loadu_pd(tw_ptr.add(2 * prev_len + j) as *const f64);
                a1 = cmul(a1, tw1);
                a2 = cmul(a2, tw2);
                a3 = cmul(a3, tw3);

                let (b0, b1, b2, b3) = dft4_f64(a0, a1, a2, a3, inverse);

                // dst_block[j + k*prev_len]: cols j,j+1 for arm k are contiguous.
                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_pd(dp.add(j + 0 * prev_len) as *mut f64, b0);
                _mm256_storeu_pd(dp.add(j + 1 * prev_len) as *mut f64, b1);
                _mm256_storeu_pd(dp.add(j + 2 * prev_len) as *mut f64, b2);
                _mm256_storeu_pd(dp.add(j + 3 * prev_len) as *mut f64, b3);

                j += 2;
            }

            // Scalar tail for odd prev_len.
            if j < prev_len {
                let src_base_j = src_base + j;
                let a0s = *src_ptr.add(0 * stride + src_base_j);
                let mut a1s = *src_ptr.add(1 * stride + src_base_j);
                let mut a2s = *src_ptr.add(2 * stride + src_base_j);
                let mut a3s = *src_ptr.add(3 * stride + src_base_j);
                if j > 0 {
                    let tw1 = *tw.as_ptr().add(0 * prev_len + j);
                    let tw2 = *tw.as_ptr().add(1 * prev_len + j);
                    let tw3 = *tw.as_ptr().add(2 * prev_len + j);
                    a1s = Complex {
                        re: a1s.re * tw1.re - a1s.im * tw1.im,
                        im: a1s.re * tw1.im + a1s.im * tw1.re,
                    };
                    a2s = Complex {
                        re: a2s.re * tw2.re - a2s.im * tw2.im,
                        im: a2s.re * tw2.im + a2s.im * tw2.re,
                    };
                    a3s = Complex {
                        re: a3s.re * tw3.re - a3s.im * tw3.im,
                        im: a3s.re * tw3.im + a3s.im * tw3.re,
                    };
                }
                let t0 = a0s + a2s;
                let t1 = a0s - a2s;
                let t2 = a1s + a3s;
                let t3 = a1s - a3s;
                let (it3_re, it3_im) = if inverse {
                    (-t3.im, t3.re)
                } else {
                    (t3.im, -t3.re)
                };
                let it3 = Complex {
                    re: it3_re,
                    im: it3_im,
                };
                let dp = dst_ptr.add(dst_base);
                *dp.add(j + 0 * prev_len) = t0 + t2;
                *dp.add(j + 1 * prev_len) = t1 + it3;
                *dp.add(j + 2 * prev_len) = t0 - t2;
                *dp.add(j + 3 * prev_len) = t1 - it3;
            }
        }
    }

    // Pointwise frequency-domain multiply (convolution spectrum).
    // Applied only on the last stage; for r=4 in M=20736 this is never reached
    // since the last stage is r=3. Implemented for correctness in all callers.
    if let Some(pw) = pointwise {
        apply_pointwise_f64(dst_ptr, pw.as_ptr(), dst.len());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// f32 AVX2+FMA flat radix-4 pass
//
// Each __m256 holds 8 f32 = 4 Complex<f32>. Layout: [re0,im0,re1,im1,re2,im2,re3,im3].
//
// ## prev_len == 1
//   Process 4 groups per AVX2 iteration.
//   arm k elements at src[k*stride + g..+4] — 4 contiguous Complex<f32> → __m256.
//
// ## prev_len >= 4
//   Process 4 adjacent columns per AVX2 iteration.
//   Loads: src[k*stride + g*prev_len + j..+4].
//
// ## prev_len in {2,3}
//   Process 2 adjacent columns per 128-bit iteration (via __m128 loads/stores).
// ─────────────────────────────────────────────────────────────────────────────

/// Complex multiply for 4 Complex<f32> pairs simultaneously (8 f32 = __m256).
///
/// Layout: [re0,im0,re1,im1,re2,im2,re3,im3].
/// moveldup/movehdup broadcast re/im; permute(0xB1) swaps re↔im within each pair.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn cmul_f32(a: __m256, b: __m256) -> __m256 {
    let a_re = _mm256_moveldup_ps(a); // [re0,re0,re1,re1,re2,re2,re3,re3]
    let a_im = _mm256_movehdup_ps(a); // [im0,im0,im1,im1,im2,im2,im3,im3]
    let b_sw = _mm256_permute_ps(b, 0xB1); // swap re↔im: [im0,re0,im1,re1,...]
    _mm256_fmaddsub_ps(a_re, b, _mm256_mul_ps(a_im, b_sw))
}

/// Multiply 4 Complex<f32> by -i: (re+i·im) → (im, -re).
/// permute(0xB1) swaps pairs → [im,re,...]; XOR negates re positions (1,3,5,7).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn rot_neg_i_f32(v: __m256) -> __m256 {
    // After permute(0xB1): positions [0,2,4,6] = im, positions [1,3,5,7] = re.
    // Negate positions 1,3,5,7 (the re positions): sign = [0,-0,0,-0,0,-0,0,-0].
    let sign = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
    _mm256_xor_ps(_mm256_permute_ps(v, 0xB1), sign)
}

/// Multiply 4 Complex<f32> by +i: (re+i·im) → (-im, re).
/// permute(0xB1) swaps pairs → [im,re,...]; XOR negates im positions (0,2,4,6).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn rot_pos_i_f32(v: __m256) -> __m256 {
    // Negate positions 0,2,4,6 (the im positions after permute).
    let sign = _mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    _mm256_xor_ps(_mm256_permute_ps(v, 0xB1), sign)
}

/// Radix-4 butterfly for 4 Complex<f32> simultaneously.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft4_f32(
    a0: __m256,
    a1: __m256,
    a2: __m256,
    a3: __m256,
    inverse: bool,
) -> (__m256, __m256, __m256, __m256) {
    let t0 = _mm256_add_ps(a0, a2);
    let t1 = _mm256_sub_ps(a0, a2);
    let t2 = _mm256_add_ps(a1, a3);
    let t3 = _mm256_sub_ps(a1, a3);
    if inverse {
        let r = rot_pos_i_f32(t3);
        (
            _mm256_add_ps(t0, t2),
            _mm256_add_ps(t1, r),
            _mm256_sub_ps(t0, t2),
            _mm256_sub_ps(t1, r),
        )
    } else {
        let r = rot_neg_i_f32(t3);
        (
            _mm256_add_ps(t0, t2),
            _mm256_add_ps(t1, r),
            _mm256_sub_ps(t0, t2),
            _mm256_sub_ps(t1, r),
        )
    }
}

/// Radix-4 butterfly for 2 Complex<f32> simultaneously (128-bit lane pair).
///
/// Used when prev_len is 2 or 3 (less than 4) to process 2 columns at a time.
/// Each __m128 holds [re0,im0,re1,im1].
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft4_f32_128(
    a0: __m128,
    a1: __m128,
    a2: __m128,
    a3: __m128,
    inverse: bool,
) -> (__m128, __m128, __m128, __m128) {
    let t0 = _mm_add_ps(a0, a2);
    let t1 = _mm_sub_ps(a0, a2);
    let t2 = _mm_add_ps(a1, a3);
    let t3 = _mm_sub_ps(a1, a3);
    // rot_neg_i / rot_pos_i for __m128: permute(0xB1) + XOR sign
    if inverse {
        let perm = _mm_permute_ps(t3, 0xB1); // [im0,re0,im1,re1]
        let sign = _mm_set_ps(0.0, -0.0, 0.0, -0.0); // negate pos 1,3 (re positions)
        let r = _mm_xor_ps(perm, sign);
        (
            _mm_add_ps(t0, t2),
            _mm_add_ps(t1, r),
            _mm_sub_ps(t0, t2),
            _mm_sub_ps(t1, r),
        )
    } else {
        let perm = _mm_permute_ps(t3, 0xB1); // [im0,re0,im1,re1]
        let sign = _mm_set_ps(-0.0, 0.0, -0.0, 0.0); // negate pos 0,2 (im positions)
        let r = _mm_xor_ps(perm, sign);
        (
            _mm_add_ps(t0, t2),
            _mm_add_ps(t1, r),
            _mm_sub_ps(t0, t2),
            _mm_sub_ps(t1, r),
        )
    }
}

/// Apply pointwise multiply to f32 dst.
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_pointwise_f32(dst: *mut Complex<f32>, pw: *const Complex<f32>, len: usize) {
    let mut i = 0usize;
    while i + 3 < len {
        let d = _mm256_loadu_ps(dst.add(i) as *const f32);
        let p = _mm256_loadu_ps(pw.add(i) as *const f32);
        _mm256_storeu_ps(dst.add(i) as *mut f32, cmul_f32(d, p));
        i += 4;
    }
    // 2-wide tail
    while i + 1 < len {
        let d = _mm_loadu_ps(dst.add(i) as *const f32);
        let p = _mm_loadu_ps(pw.add(i) as *const f32);
        // cmul_f32 via 128-bit: moveldup/movehdup equivalents
        let d_re = _mm_moveldup_ps(d);
        let d_im = _mm_movehdup_ps(d);
        let p_sw = _mm_permute_ps(p, 0xB1);
        let result = _mm_fmaddsub_ps(d_re, p, _mm_mul_ps(d_im, p_sw));
        _mm_storeu_ps(dst.add(i) as *mut f32, result);
        i += 2;
    }
    if i < len {
        let d = *dst.add(i);
        let p = *pw.add(i);
        *dst.add(i) = Complex {
            re: d.re * p.re - d.im * p.im,
            im: d.re * p.im + d.im * p.re,
        };
    }
}

/// cmul for 2 Complex<f32> via __m128.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn cmul_f32_128(a: __m128, b: __m128) -> __m128 {
    let a_re = _mm_moveldup_ps(a);
    let a_im = _mm_movehdup_ps(a);
    let b_sw = _mm_permute_ps(b, 0xB1);
    _mm_fmaddsub_ps(a_re, b, _mm_mul_ps(a_im, b_sw))
}

/// Flat radix-4 Stockham pass in AVX2+FMA for f32. Processes all `g_count` groups.
///
/// `src` and `dst` are the full n-element arrays for this pass.
/// `tw` is the twiddle slice for this stage: `(R-1) × prev_len` entries.
///
/// # Layout invariants
/// - stage_chunk = 4 × prev_len
/// - stride = g_count × prev_len = n / 4  (constant for r=4)
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn flat_pass_r4_f32(
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    prev_len: usize,
    g_count: usize,
    stage_chunk: usize,
    tw: &[Complex<f32>],
    pointwise: Option<&[Complex<f32>]>,
    inverse: bool,
) {
    let stride = g_count * prev_len; // = n/4 for r=4
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        // No twiddles. Process 4 groups per AVX2 iteration.
        // arm k block: src[k*stride .. (k+1)*stride]. Adjacent groups are contiguous.
        //
        // Transpose output using 64-bit (f64-width) unpack:
        //   Each Complex<f32> = 8 bytes = 1 f64-equivalent.
        //   b0 = [b0g0, b0g1, b0g2, b0g3] in 64-bit lanes.
        //   _mm_unpacklo_pd(as_d(b0_lo), as_d(b1_lo)) = [b0g0, b1g0] → group-g arm-01 pair.
        //   _mm_unpacklo_pd(as_d(b2_lo), as_d(b3_lo)) = [b2g0, b3g0] → group-g arm-23 pair.
        //   _mm256_set_m128 combines both → dst[g*4 .. g*4+4] in one 256-bit store.
        let mut g = 0usize;
        while g + 3 < g_count {
            let a0 = _mm256_loadu_ps(src_ptr.add(0 * stride + g) as *const f32);
            let a1 = _mm256_loadu_ps(src_ptr.add(1 * stride + g) as *const f32);
            let a2 = _mm256_loadu_ps(src_ptr.add(2 * stride + g) as *const f32);
            let a3 = _mm256_loadu_ps(src_ptr.add(3 * stride + g) as *const f32);

            let (b0, b1, b2, b3) = dft4_f32(a0, a1, a2, a3, inverse);

            // Split each result into lo/hi 128-bit halves (groups g,g+1 / g+2,g+3).
            let b0_lo = _mm_castps_pd(_mm256_castps256_ps128(b0)); // [b0g0, b0g1]
            let b1_lo = _mm_castps_pd(_mm256_castps256_ps128(b1)); // [b1g0, b1g1]
            let b2_lo = _mm_castps_pd(_mm256_castps256_ps128(b2)); // [b2g0, b2g1]
            let b3_lo = _mm_castps_pd(_mm256_castps256_ps128(b3)); // [b3g0, b3g1]
            let b0_hi = _mm_castps_pd(_mm256_extractf128_ps(b0, 1)); // [b0g2, b0g3]
            let b1_hi = _mm_castps_pd(_mm256_extractf128_ps(b1, 1));
            let b2_hi = _mm_castps_pd(_mm256_extractf128_ps(b2, 1));
            let b3_hi = _mm_castps_pd(_mm256_extractf128_ps(b3, 1));

            // Group g: arms 0-3 = [b0g0,b1g0] ++ [b2g0,b3g0]
            let g0_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0_lo, b1_lo));
            let g0_23 = _mm_castpd_ps(_mm_unpacklo_pd(b2_lo, b3_lo));
            _mm256_storeu_ps(
                dst_ptr.add(g * 4) as *mut f32,
                _mm256_set_m128(g0_23, g0_01),
            );

            // Group g+1: arms 0-3 = [b0g1,b1g1] ++ [b2g1,b3g1]
            let g1_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0_lo, b1_lo));
            let g1_23 = _mm_castpd_ps(_mm_unpackhi_pd(b2_lo, b3_lo));
            _mm256_storeu_ps(
                dst_ptr.add((g + 1) * 4) as *mut f32,
                _mm256_set_m128(g1_23, g1_01),
            );

            // Group g+2
            let g2_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0_hi, b1_hi));
            let g2_23 = _mm_castpd_ps(_mm_unpacklo_pd(b2_hi, b3_hi));
            _mm256_storeu_ps(
                dst_ptr.add((g + 2) * 4) as *mut f32,
                _mm256_set_m128(g2_23, g2_01),
            );

            // Group g+3
            let g3_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0_hi, b1_hi));
            let g3_23 = _mm_castpd_ps(_mm_unpackhi_pd(b2_hi, b3_hi));
            _mm256_storeu_ps(
                dst_ptr.add((g + 3) * 4) as *mut f32,
                _mm256_set_m128(g3_23, g3_01),
            );

            g += 4;
        }
        // 2-group tail (handles when g_count mod 4 >= 2).
        if g + 1 < g_count {
            // Load 2 groups (128 bits = 4 f32 = 2 Complex<f32>) from each arm.
            let a0 = _mm_loadu_ps(src_ptr.add(0 * stride + g) as *const f32);
            let a1 = _mm_loadu_ps(src_ptr.add(1 * stride + g) as *const f32);
            let a2 = _mm_loadu_ps(src_ptr.add(2 * stride + g) as *const f32);
            let a3 = _mm_loadu_ps(src_ptr.add(3 * stride + g) as *const f32);
            let (b0, b1, b2, b3) = dft4_f32_128(a0, a1, a2, a3, inverse);
            // b0 = [b0g0.re, b0g0.im, b0g1.re, b0g1.im] as __m128
            let b0d = _mm_castps_pd(b0); // [b0g0, b0g1]
            let b1d = _mm_castps_pd(b1);
            let b2d = _mm_castps_pd(b2);
            let b3d = _mm_castps_pd(b3);
            let g0_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0d, b1d)); // [b0g0, b1g0]
            let g0_23 = _mm_castpd_ps(_mm_unpacklo_pd(b2d, b3d)); // [b2g0, b3g0]
            _mm256_storeu_ps(
                dst_ptr.add(g * 4) as *mut f32,
                _mm256_set_m128(g0_23, g0_01),
            );
            let g1_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0d, b1d));
            let g1_23 = _mm_castpd_ps(_mm_unpackhi_pd(b2d, b3d));
            _mm256_storeu_ps(
                dst_ptr.add((g + 1) * 4) as *mut f32,
                _mm256_set_m128(g1_23, g1_01),
            );
            g += 2;
        }
        // Scalar tail for remaining 0 or 1 groups.
        while g < g_count {
            let a0s = *src_ptr.add(0 * stride + g);
            let a1s = *src_ptr.add(1 * stride + g);
            let a2s = *src_ptr.add(2 * stride + g);
            let a3s = *src_ptr.add(3 * stride + g);
            let t0 = a0s + a2s;
            let t1 = a0s - a2s;
            let t2 = a1s + a3s;
            let t3 = a1s - a3s;
            let (it3_re, it3_im) = if inverse {
                (-t3.im, t3.re)
            } else {
                (t3.im, -t3.re)
            };
            let it3 = Complex {
                re: it3_re,
                im: it3_im,
            };
            let d = dst_ptr.add(g * 4);
            *d.add(0) = t0 + t2;
            *d.add(1) = t1 + it3;
            *d.add(2) = t0 - t2;
            *d.add(3) = t1 - it3;
            g += 1;
        }
    } else if prev_len >= 4 {
        // Process 4 adjacent columns per AVX2 iteration, 1 group at a time.
        // Twiddle layout matches f64: arm k at tw[(k-1)*prev_len + j].
        // Hoist tw_ptr outside group loop since it never changes.
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;

            let mut j = 0usize;
            while j + 3 < prev_len {
                let off0 = 0 * stride + src_base + j;
                let off1 = 1 * stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;

                let a0 = _mm256_loadu_ps(src_ptr.add(off0) as *const f32);
                let mut a1 = _mm256_loadu_ps(src_ptr.add(off1) as *const f32);
                let mut a2 = _mm256_loadu_ps(src_ptr.add(off2) as *const f32);
                let mut a3 = _mm256_loadu_ps(src_ptr.add(off3) as *const f32);

                let tw1 = _mm256_loadu_ps(tw_ptr.add(0 * prev_len + j) as *const f32);
                let tw2 = _mm256_loadu_ps(tw_ptr.add(1 * prev_len + j) as *const f32);
                let tw3 = _mm256_loadu_ps(tw_ptr.add(2 * prev_len + j) as *const f32);
                // Prefetch twiddle data 2 iterations ahead for L1 cache.
                // Only beneficial when prev_len is large enough to have cache pressure.
                // Cast to *const i8 as required by _mm_prefetch intrinsic.
                if prev_len >= 16 {
                    _mm_prefetch(tw_ptr.add(0 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(1 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(2 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                }
                a1 = cmul_f32(a1, tw1);
                a2 = cmul_f32(a2, tw2);
                a3 = cmul_f32(a3, tw3);

                let (b0, b1, b2, b3) = dft4_f32(a0, a1, a2, a3, inverse);

                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_ps(dp.add(j + 0 * prev_len) as *mut f32, b0);
                _mm256_storeu_ps(dp.add(j + 1 * prev_len) as *mut f32, b1);
                _mm256_storeu_ps(dp.add(j + 2 * prev_len) as *mut f32, b2);
                _mm256_storeu_ps(dp.add(j + 3 * prev_len) as *mut f32, b3);

                j += 4;
            }
            // 2-column tail
            while j + 1 < prev_len {
                let off0 = 0 * stride + src_base + j;
                let off1 = 1 * stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;
                let a0 = _mm_loadu_ps(src_ptr.add(off0) as *const f32);
                let mut a1 = _mm_loadu_ps(src_ptr.add(off1) as *const f32);
                let mut a2 = _mm_loadu_ps(src_ptr.add(off2) as *const f32);
                let mut a3 = _mm_loadu_ps(src_ptr.add(off3) as *const f32);
                let tw1 = _mm_loadu_ps(tw_ptr.add(0 * prev_len + j) as *const f32);
                let tw2 = _mm_loadu_ps(tw_ptr.add(1 * prev_len + j) as *const f32);
                let tw3 = _mm_loadu_ps(tw_ptr.add(2 * prev_len + j) as *const f32);
                a1 = cmul_f32_128(a1, tw1);
                a2 = cmul_f32_128(a2, tw2);
                a3 = cmul_f32_128(a3, tw3);
                let (b0, b1, b2, b3) = dft4_f32_128(a0, a1, a2, a3, inverse);
                let dp = dst_ptr.add(dst_base);
                _mm_storeu_ps(dp.add(j + 0 * prev_len) as *mut f32, b0);
                _mm_storeu_ps(dp.add(j + 1 * prev_len) as *mut f32, b1);
                _mm_storeu_ps(dp.add(j + 2 * prev_len) as *mut f32, b2);
                _mm_storeu_ps(dp.add(j + 3 * prev_len) as *mut f32, b3);
                j += 2;
            }
            // Scalar tail
            if j < prev_len {
                let src_j = src_base + j;
                let a0s = *src_ptr.add(0 * stride + src_j);
                let mut a1s = *src_ptr.add(1 * stride + src_j);
                let mut a2s = *src_ptr.add(2 * stride + src_j);
                let mut a3s = *src_ptr.add(3 * stride + src_j);
                let tw1 = *tw_ptr.add(0 * prev_len + j);
                let tw2 = *tw_ptr.add(1 * prev_len + j);
                let tw3 = *tw_ptr.add(2 * prev_len + j);
                a1s = Complex {
                    re: a1s.re * tw1.re - a1s.im * tw1.im,
                    im: a1s.re * tw1.im + a1s.im * tw1.re,
                };
                a2s = Complex {
                    re: a2s.re * tw2.re - a2s.im * tw2.im,
                    im: a2s.re * tw2.im + a2s.im * tw2.re,
                };
                a3s = Complex {
                    re: a3s.re * tw3.re - a3s.im * tw3.im,
                    im: a3s.re * tw3.im + a3s.im * tw3.re,
                };
                let t0 = a0s + a2s;
                let t1 = a0s - a2s;
                let t2 = a1s + a3s;
                let t3 = a1s - a3s;
                let (it3_re, it3_im) = if inverse {
                    (-t3.im, t3.re)
                } else {
                    (t3.im, -t3.re)
                };
                let it3 = Complex {
                    re: it3_re,
                    im: it3_im,
                };
                let dp = dst_ptr.add(dst_base);
                *dp.add(j + 0 * prev_len) = t0 + t2;
                *dp.add(j + 1 * prev_len) = t1 + it3;
                *dp.add(j + 2 * prev_len) = t0 - t2;
                *dp.add(j + 3 * prev_len) = t1 - it3;
            }
        }
    } else {
        // prev_len in {2, 3}: 2-column __m128 path, same structure as f64 prev_len>=2.
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;
            while j + 1 < prev_len {
                let off0 = 0 * stride + src_base + j;
                let off1 = 1 * stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;
                let a0 = _mm_loadu_ps(src_ptr.add(off0) as *const f32);
                let mut a1 = _mm_loadu_ps(src_ptr.add(off1) as *const f32);
                let mut a2 = _mm_loadu_ps(src_ptr.add(off2) as *const f32);
                let mut a3 = _mm_loadu_ps(src_ptr.add(off3) as *const f32);
                let tw1 = _mm_loadu_ps(tw.as_ptr().add(0 * prev_len + j) as *const f32);
                let tw2 = _mm_loadu_ps(tw.as_ptr().add(1 * prev_len + j) as *const f32);
                let tw3 = _mm_loadu_ps(tw.as_ptr().add(2 * prev_len + j) as *const f32);
                a1 = cmul_f32_128(a1, tw1);
                a2 = cmul_f32_128(a2, tw2);
                a3 = cmul_f32_128(a3, tw3);
                let (b0, b1, b2, b3) = dft4_f32_128(a0, a1, a2, a3, inverse);
                let dp = dst_ptr.add(dst_base);
                _mm_storeu_ps(dp.add(j + 0 * prev_len) as *mut f32, b0);
                _mm_storeu_ps(dp.add(j + 1 * prev_len) as *mut f32, b1);
                _mm_storeu_ps(dp.add(j + 2 * prev_len) as *mut f32, b2);
                _mm_storeu_ps(dp.add(j + 3 * prev_len) as *mut f32, b3);
                j += 2;
            }
            if j < prev_len {
                let src_j = src_base + j;
                let a0s = *src_ptr.add(0 * stride + src_j);
                let mut a1s = *src_ptr.add(1 * stride + src_j);
                let mut a2s = *src_ptr.add(2 * stride + src_j);
                let mut a3s = *src_ptr.add(3 * stride + src_j);
                if j > 0 {
                    let tw1 = *tw.as_ptr().add(0 * prev_len + j);
                    let tw2 = *tw.as_ptr().add(1 * prev_len + j);
                    let tw3 = *tw.as_ptr().add(2 * prev_len + j);
                    a1s = Complex {
                        re: a1s.re * tw1.re - a1s.im * tw1.im,
                        im: a1s.re * tw1.im + a1s.im * tw1.re,
                    };
                    a2s = Complex {
                        re: a2s.re * tw2.re - a2s.im * tw2.im,
                        im: a2s.re * tw2.im + a2s.im * tw2.re,
                    };
                    a3s = Complex {
                        re: a3s.re * tw3.re - a3s.im * tw3.im,
                        im: a3s.re * tw3.im + a3s.im * tw3.re,
                    };
                }
                let t0 = a0s + a2s;
                let t1 = a0s - a2s;
                let t2 = a1s + a3s;
                let t3 = a1s - a3s;
                let (it3_re, it3_im) = if inverse {
                    (-t3.im, t3.re)
                } else {
                    (t3.im, -t3.re)
                };
                let it3 = Complex {
                    re: it3_re,
                    im: it3_im,
                };
                let dp = dst_ptr.add(dst_base);
                *dp.add(j + 0 * prev_len) = t0 + t2;
                *dp.add(j + 1 * prev_len) = t1 + it3;
                *dp.add(j + 2 * prev_len) = t0 - t2;
                *dp.add(j + 3 * prev_len) = t1 - it3;
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f32(dst_ptr, pw.as_ptr(), dst.len());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// f32 AVX2+FMA flat Stockham pass for radix-3.
//
// DFT-3 butterfly applied to 4 Complex<f32> simultaneously (__m256 / 8 f32).
// Each Complex<f32> = 64 bits — same 64-bit-slot trick as f64 for scatter-stores.
//
// ## prev_len == 1: process 4 groups per __m256; 2 groups per __m128 tail.
// ## prev_len >= 4: process 4 adjacent columns per __m256; 2 per __m128 tail.
// ## prev_len in {2,3}: 2-column __m128 loop.
// ─────────────────────────────────────────────────────────────────────────────

/// Flat radix-3 Stockham pass in AVX2+FMA for f32. Processes all `g_count` groups.
///
/// `tw` has `2 × prev_len` entries: arm 1 at `[0*prev_len..)`, arm 2 at `[1*prev_len..)`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn flat_pass_r3_f32(
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    prev_len: usize,
    g_count: usize,
    stage_chunk: usize,
    tw: &[Complex<f32>],
    pointwise: Option<&[Complex<f32>]>,
    inverse: bool,
) {
    let stride = g_count * prev_len;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let wr_f32 = _mm256_set1_ps(-0.5_f32);
    let s_f32 = _mm256_set1_ps(0.8660254037844386_f32); // √3/2
    let wr_128 = _mm_set1_ps(-0.5_f32);
    let s_128 = _mm_set1_ps(0.8660254037844386_f32);

    // DFT3 for 4 Complex<f32> via __m256.
    // Returns (b0, b1, b2).
    macro_rules! dft3_256 {
        ($a0:expr, $a1:expr, $a2:expr) => {{
            let sum = _mm256_add_ps($a1, $a2);
            let diff = _mm256_sub_ps($a1, $a2);
            let b0 = _mm256_add_ps($a0, sum);
            let m0 = _mm256_fmadd_ps(sum, wr_f32, $a0);
            let m1 = _mm256_mul_ps(
                if inverse {
                    rot_pos_i_f32(diff)
                } else {
                    rot_neg_i_f32(diff)
                },
                s_f32,
            );
            (b0, _mm256_add_ps(m0, m1), _mm256_sub_ps(m0, m1))
        }};
    }

    // DFT3 for 2 Complex<f32> via __m128.
    macro_rules! dft3_128 {
        ($a0:expr, $a1:expr, $a2:expr) => {{
            let sum = _mm_add_ps($a1, $a2);
            let diff = _mm_sub_ps($a1, $a2);
            let b0 = _mm_add_ps($a0, sum);
            let m0 = _mm_fmadd_ps(sum, wr_128, $a0);
            let perm = _mm_permute_ps(diff, 0xB1);
            let sign_fwd = _mm_set_ps(-0.0, 0.0, -0.0, 0.0); // negate im after permute (forward)
            let sign_inv = _mm_set_ps(0.0, -0.0, 0.0, -0.0); // negate im for inverse
            let m1_dir = _mm_xor_ps(perm, if inverse { sign_inv } else { sign_fwd });
            let m1 = _mm_mul_ps(m1_dir, s_128);
            (b0, _mm_add_ps(m0, m1), _mm_sub_ps(m0, m1))
        }};
    }

    if prev_len == 1 {
        // 4-group AVX2 loop.
        let mut g = 0usize;
        while g + 3 < g_count {
            let a0 = _mm256_loadu_ps(src_ptr.add(0 * stride + g) as *const f32);
            let a1 = _mm256_loadu_ps(src_ptr.add(1 * stride + g) as *const f32);
            let a2 = _mm256_loadu_ps(src_ptr.add(2 * stride + g) as *const f32);
            let (b0, b1, b2) = dft3_256!(a0, a1, a2);

            // Scatter to dst[g*3..g*3+3], ..., dst[(g+3)*3..].
            // Treat each Complex<f32> as a 64-bit slot; use unpack_pd to interleave.
            let b0_lo_d = _mm_castps_pd(_mm256_castps256_ps128(b0)); // [b0g0,b0g1]
            let b1_lo_d = _mm_castps_pd(_mm256_castps256_ps128(b1));
            let b2_lo_d = _mm_castps_pd(_mm256_castps256_ps128(b2));
            let b0_hi_d = _mm_castps_pd(_mm256_extractf128_ps(b0, 1)); // [b0g2,b0g3]
            let b1_hi_d = _mm_castps_pd(_mm256_extractf128_ps(b1, 1));
            let b2_hi_d = _mm_castps_pd(_mm256_extractf128_ps(b2, 1));

            // Group g: [b0g0,b1g0] + b2g0
            let g0_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0_lo_d, b1_lo_d));
            _mm_storeu_ps(dst_ptr.add(g * 3) as *mut f32, g0_01);
            _mm_store_sd(dst_ptr.add(g * 3 + 2) as *mut f64, b2_lo_d);

            // Group g+1: [b0g1,b1g1] + b2g1
            let g1_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0_lo_d, b1_lo_d));
            _mm_storeu_ps(dst_ptr.add((g + 1) * 3) as *mut f32, g1_01);
            _mm_store_sd(
                dst_ptr.add((g + 1) * 3 + 2) as *mut f64,
                _mm_unpackhi_pd(b2_lo_d, b2_lo_d),
            );

            // Group g+2
            let g2_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0_hi_d, b1_hi_d));
            _mm_storeu_ps(dst_ptr.add((g + 2) * 3) as *mut f32, g2_01);
            _mm_store_sd(dst_ptr.add((g + 2) * 3 + 2) as *mut f64, b2_hi_d);

            // Group g+3
            let g3_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0_hi_d, b1_hi_d));
            _mm_storeu_ps(dst_ptr.add((g + 3) * 3) as *mut f32, g3_01);
            _mm_store_sd(
                dst_ptr.add((g + 3) * 3 + 2) as *mut f64,
                _mm_unpackhi_pd(b2_hi_d, b2_hi_d),
            );

            g += 4;
        }
        // 2-group __m128 tail
        while g + 1 < g_count {
            let a0 = _mm_loadu_ps(src_ptr.add(0 * stride + g) as *const f32);
            let a1 = _mm_loadu_ps(src_ptr.add(1 * stride + g) as *const f32);
            let a2 = _mm_loadu_ps(src_ptr.add(2 * stride + g) as *const f32);
            let (b0, b1, b2) = dft3_128!(a0, a1, a2);
            let b0d = _mm_castps_pd(b0);
            let b1d = _mm_castps_pd(b1);
            let b2d = _mm_castps_pd(b2);
            let g0_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0d, b1d));
            _mm_storeu_ps(dst_ptr.add(g * 3) as *mut f32, g0_01);
            _mm_store_sd(dst_ptr.add(g * 3 + 2) as *mut f64, b2d);
            let g1_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0d, b1d));
            _mm_storeu_ps(dst_ptr.add((g + 1) * 3) as *mut f32, g1_01);
            _mm_store_sd(
                dst_ptr.add((g + 1) * 3 + 2) as *mut f64,
                _mm_unpackhi_pd(b2d, b2d),
            );
            g += 2;
        }
        // Scalar tail
        while g < g_count {
            let a0s = *src_ptr.add(0 * stride + g);
            let a1s = *src_ptr.add(1 * stride + g);
            let a2s = *src_ptr.add(2 * stride + g);
            scalar_dft3_f32(a0s, a1s, a2s, dst_ptr.add(g * 3), inverse);
            g += 1;
        }
    } else if prev_len >= 4 {
        // 4-column AVX2 loop, 1 group at a time.
        // Hoist tw_ptr outside group loop since it never changes.
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;
            while j + 3 < prev_len {
                let off0 = 0 * stride + src_base + j;
                let off1 = 1 * stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let a0 = _mm256_loadu_ps(src_ptr.add(off0) as *const f32);
                let tw1 = _mm256_loadu_ps(tw_ptr.add(0 * prev_len + j) as *const f32);
                let tw2 = _mm256_loadu_ps(tw_ptr.add(1 * prev_len + j) as *const f32);
                let a1 = cmul_f32(_mm256_loadu_ps(src_ptr.add(off1) as *const f32), tw1);
                let a2 = cmul_f32(_mm256_loadu_ps(src_ptr.add(off2) as *const f32), tw2);
                // Prefetch twiddle data 2 iterations ahead for L1 cache.
                // Only beneficial when prev_len is large enough to have cache pressure.
                // Cast to *const i8 as required by _mm_prefetch intrinsic.
                if prev_len >= 16 {
                    _mm_prefetch(tw_ptr.add(0 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(1 * prev_len + j + 8) as *const i8, _MM_HINT_T0);
                }
                let (b0, b1, b2) = dft3_256!(a0, a1, a2);
                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_ps(dp.add(j + 0 * prev_len) as *mut f32, b0);
                _mm256_storeu_ps(dp.add(j + 1 * prev_len) as *mut f32, b1);
                _mm256_storeu_ps(dp.add(j + 2 * prev_len) as *mut f32, b2);
                j += 4;
            }
            // 2-column __m128 tail
            while j + 1 < prev_len {
                let off0 = 0 * stride + src_base + j;
                let off1 = 1 * stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let tw1 = _mm_loadu_ps(tw_ptr.add(0 * prev_len + j) as *const f32);
                let tw2 = _mm_loadu_ps(tw_ptr.add(1 * prev_len + j) as *const f32);
                let a0 = _mm_loadu_ps(src_ptr.add(off0) as *const f32);
                let a1 = cmul_f32_128(_mm_loadu_ps(src_ptr.add(off1) as *const f32), tw1);
                let a2 = cmul_f32_128(_mm_loadu_ps(src_ptr.add(off2) as *const f32), tw2);
                let (b0, b1, b2) = dft3_128!(a0, a1, a2);
                let dp = dst_ptr.add(dst_base);
                _mm_storeu_ps(dp.add(j + 0 * prev_len) as *mut f32, b0);
                _mm_storeu_ps(dp.add(j + 1 * prev_len) as *mut f32, b1);
                _mm_storeu_ps(dp.add(j + 2 * prev_len) as *mut f32, b2);
                j += 2;
            }
            if j < prev_len {
                scalar_dft3_f32_col(
                    src_ptr, dst_ptr, stride, dst_base, src_base, j, prev_len, tw, inverse,
                );
            }
        }
    } else {
        // prev_len in {2, 3}: 2-column __m128 loop.
        // Hoist tw_ptr outside group loop since it never changes.
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;
            while j + 1 < prev_len {
                let off0 = 0 * stride + src_base + j;
                let off1 = 1 * stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let tw1 = _mm_loadu_ps(tw_ptr.add(0 * prev_len + j) as *const f32);
                let tw2 = _mm_loadu_ps(tw_ptr.add(1 * prev_len + j) as *const f32);
                let a0 = _mm_loadu_ps(src_ptr.add(off0) as *const f32);
                let a1 = cmul_f32_128(_mm_loadu_ps(src_ptr.add(off1) as *const f32), tw1);
                let a2 = cmul_f32_128(_mm_loadu_ps(src_ptr.add(off2) as *const f32), tw2);
                let (b0, b1, b2) = dft3_128!(a0, a1, a2);
                let dp = dst_ptr.add(dst_base);
                _mm_storeu_ps(dp.add(j + 0 * prev_len) as *mut f32, b0);
                _mm_storeu_ps(dp.add(j + 1 * prev_len) as *mut f32, b1);
                _mm_storeu_ps(dp.add(j + 2 * prev_len) as *mut f32, b2);
                j += 2;
            }
            if j < prev_len {
                scalar_dft3_f32_col(
                    src_ptr, dst_ptr, stride, dst_base, src_base, j, prev_len, tw, inverse,
                );
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f32(dst_ptr, pw.as_ptr(), dst.len());
    }
}

/// Scalar DFT-3 for f32, sequential output (prev_len==1 case).
#[inline]
unsafe fn scalar_dft3_f32(
    a0: Complex<f32>,
    a1: Complex<f32>,
    a2: Complex<f32>,
    out: *mut Complex<f32>,
    inverse: bool,
) {
    let s: f32 = 0.8660254_f32;
    let wr: f32 = -0.5_f32;
    let sum_re = a1.re + a2.re;
    let sum_im = a1.im + a2.im;
    let diff_re = a1.re - a2.re;
    let diff_im = a1.im - a2.im;
    let m0_re = a0.re + sum_re * wr;
    let m0_im = a0.im + sum_im * wr;
    let (m1_re, m1_im) = if inverse {
        (-diff_im * s, diff_re * s)
    } else {
        (diff_im * s, -diff_re * s)
    };
    *out.add(0) = Complex {
        re: a0.re + sum_re,
        im: a0.im + sum_im,
    };
    *out.add(1) = Complex {
        re: m0_re + m1_re,
        im: m0_im + m1_im,
    };
    *out.add(2) = Complex {
        re: m0_re - m1_re,
        im: m0_im - m1_im,
    };
}

/// Scalar DFT-3 column tail for f32, column-major output (prev_len>=2 case).
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn scalar_dft3_f32_col(
    src_ptr: *const Complex<f32>,
    dst_ptr: *mut Complex<f32>,
    stride: usize,
    dst_base: usize,
    src_base: usize,
    j: usize,
    prev_len: usize,
    tw: &[Complex<f32>],
    inverse: bool,
) {
    let src_j = src_base + j;
    let a0s = *src_ptr.add(0 * stride + src_j);
    let mut a1s = *src_ptr.add(1 * stride + src_j);
    let mut a2s = *src_ptr.add(2 * stride + src_j);
    if j > 0 {
        let tw1 = *tw.as_ptr().add(0 * prev_len + j);
        let tw2 = *tw.as_ptr().add(1 * prev_len + j);
        a1s = Complex {
            re: a1s.re * tw1.re - a1s.im * tw1.im,
            im: a1s.re * tw1.im + a1s.im * tw1.re,
        };
        a2s = Complex {
            re: a2s.re * tw2.re - a2s.im * tw2.im,
            im: a2s.re * tw2.im + a2s.im * tw2.re,
        };
    }
    let s: f32 = 0.8660254_f32;
    let wr: f32 = -0.5_f32;
    let sum_re = a1s.re + a2s.re;
    let sum_im = a1s.im + a2s.im;
    let diff_re = a1s.re - a2s.re;
    let diff_im = a1s.im - a2s.im;
    let m0_re = a0s.re + sum_re * wr;
    let m0_im = a0s.im + sum_im * wr;
    let (m1_re, m1_im) = if inverse {
        (-diff_im * s, diff_re * s)
    } else {
        (diff_im * s, -diff_re * s)
    };
    let dp = dst_ptr.add(dst_base);
    *dp.add(j + 0 * prev_len) = Complex {
        re: a0s.re + sum_re,
        im: a0s.im + sum_im,
    };
    *dp.add(j + 1 * prev_len) = Complex {
        re: m0_re + m1_re,
        im: m0_im + m1_im,
    };
    *dp.add(j + 2 * prev_len) = Complex {
        re: m0_re - m1_re,
        im: m0_im - m1_im,
    };
}
