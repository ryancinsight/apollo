use super::fixed::{cmul_pair32, cmul_vec32, store_complex32_high, store_complex32_low};
use num_complex::Complex32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple32_quarter_groups_two_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
    third_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castpd_ps, _mm256_castps256_ps128, _mm256_castps_pd,
        _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_permute2f128_pd, _mm256_set1_ps,
        _mm256_set_ps, _mm256_sub_ps, _mm_storeu_ps,
    };

    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, radix << 4);
    debug_assert_eq!(dst.len(), n);
    debug_assert!(radix >= 1);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();

    for j in 0..radix {
        let w1 = *first_ptr.add(j);
        let w2a = *second_ptr.add(j);
        let w2b = *second_ptr.add(j + radix);
        let w3a = *third_ptr.add(j);
        let w3b = *third_ptr.add(j + radix);
        let w3c = *third_ptr.add(j + 2 * radix);
        let w3d = *third_ptr.add(j + 3 * radix);

        let w1r = _mm256_set1_ps(w1.re);
        let w1i = _mm256_set1_ps(w1.im);
        let w2ar = _mm256_set1_ps(w2a.re);
        let w2ai = _mm256_set1_ps(w2a.im);
        let w2br = _mm256_set1_ps(w2b.re);
        let w2bi = _mm256_set1_ps(w2b.im);

        let src_base = j * 16;
        let y01 = _mm256_loadu_ps(src_ptr.add(src_base).cast::<f32>());
        let y23 = _mm256_loadu_ps(src_ptr.add(src_base + 4).cast::<f32>());
        let y45 = _mm256_loadu_ps(src_ptr.add(src_base + 8).cast::<f32>());
        let y67 = _mm256_loadu_ps(src_ptr.add(src_base + 12).cast::<f32>());

        let x45 = cmul_vec32(w1r, w1i, y45);
        let x67 = cmul_vec32(w1r, w1i, y67);

        let s01 = _mm256_add_ps(y01, x45);
        let s23 = _mm256_add_ps(y23, x67);
        let d01 = _mm256_sub_ps(y01, x45);
        let d23 = _mm256_sub_ps(y23, x67);

        let t23 = cmul_vec32(w2ar, w2ai, s23);
        let u23 = cmul_vec32(w2br, w2bi, d23);

        let p01 = _mm256_add_ps(s01, t23);
        let p45 = _mm256_sub_ps(s01, t23);
        let p23 = _mm256_add_ps(d01, u23);
        let p67 = _mm256_sub_ps(d01, u23);

        let p02 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(p01),
            _mm256_castps_pd(p23),
            0x20,
        ));
        let p13 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(p01),
            _mm256_castps_pd(p23),
            0x31,
        ));

        let w3ab_r = _mm256_set_ps(
            w3b.re, w3b.re, w3b.re, w3b.re, w3a.re, w3a.re, w3a.re, w3a.re,
        );
        let w3ab_i = _mm256_set_ps(
            w3b.im, w3b.im, w3b.im, w3b.im, w3a.im, w3a.im, w3a.im, w3a.im,
        );
        let q01 = cmul_vec32(w3ab_r, w3ab_i, p13);

        let p46 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(p45),
            _mm256_castps_pd(p67),
            0x20,
        ));
        let p57 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(p45),
            _mm256_castps_pd(p67),
            0x31,
        ));

        let w3cd_r = _mm256_set_ps(
            w3d.re, w3d.re, w3d.re, w3d.re, w3c.re, w3c.re, w3c.re, w3c.re,
        );
        let w3cd_i = _mm256_set_ps(
            w3d.im, w3d.im, w3d.im, w3d.im, w3c.im, w3c.im, w3c.im, w3c.im,
        );
        let q23 = cmul_vec32(w3cd_r, w3cd_i, p57);

        let out02 = _mm256_add_ps(p02, q01);
        let out13 = _mm256_sub_ps(p02, q01);
        let out46 = _mm256_add_ps(p46, q23);
        let out57 = _mm256_sub_ps(p46, q23);

        let out_base = j * 2;
        _mm_storeu_ps(
            dst_ptr.add(out_base).cast::<f32>(),
            _mm256_castps256_ps128(out02),
        );
        _mm_storeu_ps(
            dst_ptr.add(eighth_n + out_base).cast::<f32>(),
            _mm256_extractf128_ps(out02, 1),
        );
        _mm_storeu_ps(
            dst_ptr.add(half_n + out_base).cast::<f32>(),
            _mm256_castps256_ps128(out13),
        );
        _mm_storeu_ps(
            dst_ptr.add(half_n + eighth_n + out_base).cast::<f32>(),
            _mm256_extractf128_ps(out13, 1),
        );

        _mm_storeu_ps(
            dst_ptr.add(quarter_n + out_base).cast::<f32>(),
            _mm256_castps256_ps128(out46),
        );
        _mm_storeu_ps(
            dst_ptr.add(quarter_n + eighth_n + out_base).cast::<f32>(),
            _mm256_extractf128_ps(out46, 1),
        );
        _mm_storeu_ps(
            dst_ptr.add(half_n + quarter_n + out_base).cast::<f32>(),
            _mm256_castps256_ps128(out57),
        );
        _mm_storeu_ps(
            dst_ptr
                .add(half_n + quarter_n + eighth_n + out_base)
                .cast::<f32>(),
            _mm256_extractf128_ps(out57, 1),
        );
    }
}

/// Fuses a f32 three-stage Stockham suffix when `groups == 4`.
///
/// With `G = 4`, each digit `j` owns exactly eight contiguous source values
/// `x0..x7`, and `quarter_groups = 1`. The scalar `stage_triple32` recurrence
/// therefore stores one complex result into each eighth-band at offset `j`.
/// This leaf evaluates the same DAG using 256-bit AVX registers by processing
/// two consecutive `j` values per iteration:
///
/// Lower 128 bits carry the data and twiddles for `j`, upper 128 bits for `j+1`.
/// Within each 128-bit lane the butterfly DAG is identical:
///
/// `s = x0..x3 + w1*x4..x7`, `d = x0..x3 - w1*x4..x7`;
/// `p0,p1,p4,p5` are produced from the `s` half and `w2a`;
/// `p2,p3,p6,p7` are produced from the `d` half and `w2b`;
/// the stage-3 products consume only `p1,p3,p5,p7`.
///
/// `_mm256_shuffle_ps` applies the same within-lane permutation to both halves
/// simultaneously, so `[p0,p2]`/`[p1,p3]` are extracted for both `j` values
/// with a single instruction. An SSE scalar tail handles odd `radix`.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple32_quarter_groups_one_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
    third_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castps128_ps256, _mm256_castps256_ps128, _mm256_extractf128_ps,
        _mm256_insertf128_ps, _mm256_set_ps, _mm256_shuffle_ps, _mm256_sub_ps, _mm_add_ps,
        _mm_loadu_ps, _mm_set1_ps, _mm_set_ps, _mm_shuffle_ps, _mm_sub_ps,
    };

    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, radix << 3);
    debug_assert_eq!(dst.len(), n);
    debug_assert!(radix >= 1);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= 2 * radix);
    debug_assert!(third_twiddles.len() >= 4 * radix);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();

    // 256-bit path: pack j and j+1 into lower/upper 128-bit halves of each YMM.
    // Each j owns 8 Complex32 = 16 f32 = 1 cache line; two adjacent j's are loaded
    // with 8 XMM loads assembled into 4 YMM registers via insertf128.
    let vector_end = radix & !1;
    let mut j = 0usize;
    while j < vector_end {
        let w1_j0 = *first_ptr.add(j);
        let w1_j1 = *first_ptr.add(j + 1);
        let w2a_j0 = *second_ptr.add(j);
        let w2a_j1 = *second_ptr.add(j + 1);
        let w2b_j0 = *second_ptr.add(j + radix);
        let w2b_j1 = *second_ptr.add(j + 1 + radix);
        let w3a_j0 = *third_ptr.add(j);
        let w3a_j1 = *third_ptr.add(j + 1);
        let w3b_j0 = *third_ptr.add(j + radix);
        let w3b_j1 = *third_ptr.add(j + 1 + radix);
        let w3c_j0 = *third_ptr.add(j + 2 * radix);
        let w3c_j1 = *third_ptr.add(j + 1 + 2 * radix);
        let w3d_j0 = *third_ptr.add(j + 3 * radix);
        let w3d_j1 = *third_ptr.add(j + 1 + 3 * radix);

        // Load: src[j*8..j*8+8] into lower 128, src[(j+1)*8..(j+1)*8+8] into upper 128.
        let src_base = j * 8;
        let lo01 = _mm_loadu_ps(src_ptr.add(src_base).cast::<f32>());
        let lo23 = _mm_loadu_ps(src_ptr.add(src_base + 2).cast::<f32>());
        let lo45 = _mm_loadu_ps(src_ptr.add(src_base + 4).cast::<f32>());
        let lo67 = _mm_loadu_ps(src_ptr.add(src_base + 6).cast::<f32>());
        let hi01 = _mm_loadu_ps(src_ptr.add(src_base + 8).cast::<f32>());
        let hi23 = _mm_loadu_ps(src_ptr.add(src_base + 10).cast::<f32>());
        let hi45 = _mm_loadu_ps(src_ptr.add(src_base + 12).cast::<f32>());
        let hi67 = _mm_loadu_ps(src_ptr.add(src_base + 14).cast::<f32>());
        let ymm01 = _mm256_insertf128_ps(_mm256_castps128_ps256(lo01), hi01, 1);
        let ymm23 = _mm256_insertf128_ps(_mm256_castps128_ps256(lo23), hi23, 1);
        let ymm45 = _mm256_insertf128_ps(_mm256_castps128_ps256(lo45), hi45, 1);
        let ymm67 = _mm256_insertf128_ps(_mm256_castps128_ps256(lo67), hi67, 1);

        // Twiddle broadcast: elements [0..3] = j0 broadcast, elements [4..7] = j1 broadcast.
        // Each Complex32 occupies 2 f32 slots, so each twiddle scalar is replicated twice per lane.
        let ymm_w1r = _mm256_set_ps(
            w1_j1.re, w1_j1.re, w1_j1.re, w1_j1.re, w1_j0.re, w1_j0.re, w1_j0.re, w1_j0.re,
        );
        let ymm_w1i = _mm256_set_ps(
            w1_j1.im, w1_j1.im, w1_j1.im, w1_j1.im, w1_j0.im, w1_j0.im, w1_j0.im, w1_j0.im,
        );
        let ymm_w2ar = _mm256_set_ps(
            w2a_j1.re, w2a_j1.re, w2a_j1.re, w2a_j1.re, w2a_j0.re, w2a_j0.re, w2a_j0.re, w2a_j0.re,
        );
        let ymm_w2ai = _mm256_set_ps(
            w2a_j1.im, w2a_j1.im, w2a_j1.im, w2a_j1.im, w2a_j0.im, w2a_j0.im, w2a_j0.im, w2a_j0.im,
        );
        let ymm_w2br = _mm256_set_ps(
            w2b_j1.re, w2b_j1.re, w2b_j1.re, w2b_j1.re, w2b_j0.re, w2b_j0.re, w2b_j0.re, w2b_j0.re,
        );
        let ymm_w2bi = _mm256_set_ps(
            w2b_j1.im, w2b_j1.im, w2b_j1.im, w2b_j1.im, w2b_j0.im, w2b_j0.im, w2b_j0.im, w2b_j0.im,
        );

        let x45 = cmul_vec32(ymm_w1r, ymm_w1i, ymm45);
        let x67 = cmul_vec32(ymm_w1r, ymm_w1i, ymm67);

        let s01 = _mm256_add_ps(ymm01, x45);
        let s23 = _mm256_add_ps(ymm23, x67);
        let d01 = _mm256_sub_ps(ymm01, x45);
        let d23 = _mm256_sub_ps(ymm23, x67);

        let t23 = cmul_vec32(ymm_w2ar, ymm_w2ai, s23);
        let u23 = cmul_vec32(ymm_w2br, ymm_w2bi, d23);
        let p01 = _mm256_add_ps(s01, t23);
        let p45 = _mm256_sub_ps(s01, t23);
        let p23 = _mm256_add_ps(d01, u23);
        let p67 = _mm256_sub_ps(d01, u23);

        // Within-lane shuffle: same imm8 applied independently to j0 (low) and j1 (high).
        // 0b1110_1110 = select [a[2],a[3],b[2],b[3]] per lane → extracts p1 and p3.
        // 0b0100_0100 = select [a[0],a[1],b[0],b[1]] per lane → extracts p0 and p2.
        let p13 = _mm256_shuffle_ps::<0b1110_1110>(p01, p23);
        let p02 = _mm256_shuffle_ps::<0b0100_0100>(p01, p23);

        // w3ab broadcast: elements [0,1] = w3a for j0's p1, [2,3] = w3b for j0's p3,
        //                 [4,5] = w3a for j1's p1, [6,7] = w3b for j1's p3.
        let ymm_w3ab_r = _mm256_set_ps(
            w3b_j1.re, w3b_j1.re, w3a_j1.re, w3a_j1.re, w3b_j0.re, w3b_j0.re, w3a_j0.re, w3a_j0.re,
        );
        let ymm_w3ab_i = _mm256_set_ps(
            w3b_j1.im, w3b_j1.im, w3a_j1.im, w3a_j1.im, w3b_j0.im, w3b_j0.im, w3a_j0.im, w3a_j0.im,
        );
        let q01 = cmul_vec32(ymm_w3ab_r, ymm_w3ab_i, p13);
        let out02 = _mm256_add_ps(p02, q01);
        let out13 = _mm256_sub_ps(p02, q01);

        let p57 = _mm256_shuffle_ps::<0b1110_1110>(p45, p67);
        let p46 = _mm256_shuffle_ps::<0b0100_0100>(p45, p67);
        let ymm_w3cd_r = _mm256_set_ps(
            w3d_j1.re, w3d_j1.re, w3c_j1.re, w3c_j1.re, w3d_j0.re, w3d_j0.re, w3c_j0.re, w3c_j0.re,
        );
        let ymm_w3cd_i = _mm256_set_ps(
            w3d_j1.im, w3d_j1.im, w3c_j1.im, w3c_j1.im, w3d_j0.im, w3d_j0.im, w3c_j0.im, w3c_j0.im,
        );
        let q23 = cmul_vec32(ymm_w3cd_r, ymm_w3cd_i, p57);
        let out46 = _mm256_add_ps(p46, q23);
        let out57 = _mm256_sub_ps(p46, q23);

        // Unpack lower (j0) and upper (j1) halves and store each complex to its band.
        let lo02 = _mm256_castps256_ps128(out02);
        let hi02 = _mm256_extractf128_ps(out02, 1);
        store_complex32_low(dst_ptr.add(j), lo02);
        store_complex32_low(dst_ptr.add(j + 1), hi02);
        store_complex32_high(dst_ptr.add(eighth_n + j), lo02);
        store_complex32_high(dst_ptr.add(eighth_n + j + 1), hi02);

        let lo13 = _mm256_castps256_ps128(out13);
        let hi13 = _mm256_extractf128_ps(out13, 1);
        store_complex32_low(dst_ptr.add(half_n + j), lo13);
        store_complex32_low(dst_ptr.add(half_n + j + 1), hi13);
        store_complex32_high(dst_ptr.add(half_n + eighth_n + j), lo13);
        store_complex32_high(dst_ptr.add(half_n + eighth_n + j + 1), hi13);

        let lo46 = _mm256_castps256_ps128(out46);
        let hi46 = _mm256_extractf128_ps(out46, 1);
        store_complex32_low(dst_ptr.add(quarter_n + j), lo46);
        store_complex32_low(dst_ptr.add(quarter_n + j + 1), hi46);
        store_complex32_high(dst_ptr.add(quarter_n + eighth_n + j), lo46);
        store_complex32_high(dst_ptr.add(quarter_n + eighth_n + j + 1), hi46);

        let lo57 = _mm256_castps256_ps128(out57);
        let hi57 = _mm256_extractf128_ps(out57, 1);
        store_complex32_low(dst_ptr.add(half_n + quarter_n + j), lo57);
        store_complex32_low(dst_ptr.add(half_n + quarter_n + j + 1), hi57);
        store_complex32_high(dst_ptr.add(half_n + quarter_n + eighth_n + j), lo57);
        store_complex32_high(dst_ptr.add(half_n + quarter_n + eighth_n + j + 1), hi57);

        j += 2;
    }

    // SSE scalar tail for odd radix (at most one iteration).
    while j < radix {
        let w1 = *first_ptr.add(j);
        let w2a = *second_ptr.add(j);
        let w2b = *second_ptr.add(j + radix);
        let w3a = *third_ptr.add(j);
        let w3b = *third_ptr.add(j + radix);
        let w3c = *third_ptr.add(j + 2 * radix);
        let w3d = *third_ptr.add(j + 3 * radix);

        let w1r = _mm_set1_ps(w1.re);
        let w1i = _mm_set1_ps(w1.im);
        let src_base = j * 8;
        let x01 = _mm_loadu_ps(src_ptr.add(src_base).cast::<f32>());
        let x23 = _mm_loadu_ps(src_ptr.add(src_base + 2).cast::<f32>());
        let x45 = cmul_pair32(
            w1r,
            w1i,
            _mm_loadu_ps(src_ptr.add(src_base + 4).cast::<f32>()),
        );
        let x67 = cmul_pair32(
            w1r,
            w1i,
            _mm_loadu_ps(src_ptr.add(src_base + 6).cast::<f32>()),
        );

        let s01 = _mm_add_ps(x01, x45);
        let s23 = _mm_add_ps(x23, x67);
        let d01 = _mm_sub_ps(x01, x45);
        let d23 = _mm_sub_ps(x23, x67);

        let t23 = cmul_pair32(_mm_set1_ps(w2a.re), _mm_set1_ps(w2a.im), s23);
        let u23 = cmul_pair32(_mm_set1_ps(w2b.re), _mm_set1_ps(w2b.im), d23);
        let p01 = _mm_add_ps(s01, t23);
        let p45 = _mm_sub_ps(s01, t23);
        let p23 = _mm_add_ps(d01, u23);
        let p67 = _mm_sub_ps(d01, u23);

        let p13 = _mm_shuffle_ps::<0b1110_1110>(p01, p23);
        let q01 = cmul_pair32(
            _mm_set_ps(w3b.re, w3b.re, w3a.re, w3a.re),
            _mm_set_ps(w3b.im, w3b.im, w3a.im, w3a.im),
            p13,
        );
        let p02 = _mm_shuffle_ps::<0b0100_0100>(p01, p23);
        let out02 = _mm_add_ps(p02, q01);
        let out13 = _mm_sub_ps(p02, q01);
        store_complex32_low(dst_ptr.add(j), out02);
        store_complex32_low(dst_ptr.add(half_n + j), out13);
        store_complex32_high(dst_ptr.add(eighth_n + j), out02);
        store_complex32_high(dst_ptr.add(half_n + eighth_n + j), out13);

        let p57 = _mm_shuffle_ps::<0b1110_1110>(p45, p67);
        let q23 = cmul_pair32(
            _mm_set_ps(w3d.re, w3d.re, w3c.re, w3c.re),
            _mm_set_ps(w3d.im, w3d.im, w3c.im, w3c.im),
            p57,
        );
        let p46 = _mm_shuffle_ps::<0b0100_0100>(p45, p67);
        let out46 = _mm_add_ps(p46, q23);
        let out57 = _mm_sub_ps(p46, q23);
        store_complex32_low(dst_ptr.add(quarter_n + j), out46);
        store_complex32_low(dst_ptr.add(half_n + quarter_n + j), out57);
        store_complex32_high(dst_ptr.add(quarter_n + eighth_n + j), out46);
        store_complex32_high(dst_ptr.add(half_n + quarter_n + eighth_n + j), out57);

        j += 1;
    }
}
