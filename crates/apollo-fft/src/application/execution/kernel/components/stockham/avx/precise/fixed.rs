use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn cmul_vec_precise(
    w_re: std::arch::x86_64::__m256d,
    w_im: std::arch::x86_64::__m256d,
    value: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{_mm256_fmaddsub_pd, _mm256_mul_pd, _mm256_permute_pd};
    _mm256_fmaddsub_pd(
        w_re,
        value,
        _mm256_mul_pd(w_im, _mm256_permute_pd::<0b0101>(value)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_cmul_by_pair_twiddle(
    value: std::arch::x86_64::__m256d,
    twiddle: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::_mm256_permute_pd;

    let wr = _mm256_permute_pd::<0b0000>(twiddle);
    let wi = _mm256_permute_pd::<0b1111>(twiddle);
    cmul_vec_precise(wr, wi, value)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn avx_rotate_quarter_turn(
    value: std::arch::x86_64::__m256d,
    sign_mask: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{_mm256_permute_pd, _mm256_xor_pd};

    _mm256_xor_pd(_mm256_permute_pd::<0b0101>(value), sign_mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_butterfly2_pair(
    a: std::arch::x86_64::__m256d,
    b: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 2] {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_sub_pd};

    [_mm256_add_pd(a, b), _mm256_sub_pd(a, b)]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_butterfly4_pair(
    rows: [std::arch::x86_64::__m256d; 4],
    quarter_turn_mask: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 4] {
    let [mid0, mid2] = avx_butterfly2_pair(rows[0], rows[2]);
    let [mid1, mid3] = avx_butterfly2_pair(rows[1], rows[3]);
    let mid3 = avx_rotate_quarter_turn(mid3, quarter_turn_mask);
    let [out0, out1] = avx_butterfly2_pair(mid0, mid1);
    let [out2, out3] = avx_butterfly2_pair(mid2, mid3);

    [out0, out2, out1, out3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_butterfly8_pair(
    rows: [std::arch::x86_64::__m256d; 8],
    quarter_turn_mask: std::arch::x86_64::__m256d,
    w1_re: std::arch::x86_64::__m256d,
    w1_im: std::arch::x86_64::__m256d,
    w3_re: std::arch::x86_64::__m256d,
    w3_im: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 8] {
    let mid0 = avx_butterfly4_pair([rows[0], rows[2], rows[4], rows[6]], quarter_turn_mask);
    let mut mid1 = avx_butterfly4_pair([rows[1], rows[3], rows[5], rows[7]], quarter_turn_mask);
    mid1[1] = cmul_vec_precise(w1_re, w1_im, mid1[1]);
    mid1[2] = avx_rotate_quarter_turn(mid1[2], quarter_turn_mask);
    mid1[3] = cmul_vec_precise(w3_re, w3_im, mid1[3]);

    let [out0, out1] = avx_butterfly2_pair(mid0[0], mid1[0]);
    let [out2, out3] = avx_butterfly2_pair(mid0[1], mid1[1]);
    let [out4, out5] = avx_butterfly2_pair(mid0[2], mid1[2]);
    let [out6, out7] = avx_butterfly2_pair(mid0[3], mid1[3]);

    [out0, out2, out4, out6, out1, out3, out5, out7]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_transpose8_pairs(
    rows: [std::arch::x86_64::__m256d; 8],
) -> [std::arch::x86_64::__m256d; 8] {
    use std::arch::x86_64::_mm256_permute2f128_pd;

    [
        _mm256_permute2f128_pd(rows[0], rows[1], 0x20),
        _mm256_permute2f128_pd(rows[2], rows[3], 0x20),
        _mm256_permute2f128_pd(rows[4], rows[5], 0x20),
        _mm256_permute2f128_pd(rows[6], rows[7], 0x20),
        _mm256_permute2f128_pd(rows[0], rows[1], 0x31),
        _mm256_permute2f128_pd(rows[2], rows[3], 0x31),
        _mm256_permute2f128_pd(rows[4], rows[5], 0x31),
        _mm256_permute2f128_pd(rows[6], rows[7], 0x31),
    ]
}

#[inline]
unsafe fn twiddle_len64_const_unchecked<const EXPONENT: usize>(
    twiddle_ptr: *const Complex64,
) -> Complex64 {
    debug_assert!(EXPONENT < 64);
    match EXPONENT {
        0 => Complex64::new(1.0, 0.0),
        1..=31 => *twiddle_ptr.add(31 + EXPONENT),
        32 => Complex64::new(-1.0, 0.0),
        _ => {
            let w = *twiddle_ptr.add(31 + EXPONENT - 32);
            Complex64::new(-w.re, -w.im)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_twiddle_len64_pair_const<
    const FIRST_EXPONENT: usize,
    const SECOND_EXPONENT: usize,
>(
    twiddle_ptr: *const Complex64,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::_mm256_set_pd;

    let first = twiddle_len64_const_unchecked::<FIRST_EXPONENT>(twiddle_ptr);
    let second = twiddle_len64_const_unchecked::<SECOND_EXPONENT>(twiddle_ptr);
    _mm256_set_pd(second.im, second.re, first.im, first.re)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_apply_len64_twiddles_for_column_pair<const COLUMN_PAIR: usize>(
    mid: &mut [std::arch::x86_64::__m256d; 8],
    twiddle_ptr: *const Complex64,
) {
    match COLUMN_PAIR {
        0 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len64_pair_const::<0, 1>(twiddle_ptr));
            mid[2] =
                avx_cmul_by_pair_twiddle(mid[2], avx_twiddle_len64_pair_const::<0, 2>(twiddle_ptr));
            mid[3] =
                avx_cmul_by_pair_twiddle(mid[3], avx_twiddle_len64_pair_const::<0, 3>(twiddle_ptr));
            mid[4] =
                avx_cmul_by_pair_twiddle(mid[4], avx_twiddle_len64_pair_const::<0, 4>(twiddle_ptr));
            mid[5] =
                avx_cmul_by_pair_twiddle(mid[5], avx_twiddle_len64_pair_const::<0, 5>(twiddle_ptr));
            mid[6] =
                avx_cmul_by_pair_twiddle(mid[6], avx_twiddle_len64_pair_const::<0, 6>(twiddle_ptr));
            mid[7] =
                avx_cmul_by_pair_twiddle(mid[7], avx_twiddle_len64_pair_const::<0, 7>(twiddle_ptr));
        }
        1 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len64_pair_const::<2, 3>(twiddle_ptr));
            mid[2] =
                avx_cmul_by_pair_twiddle(mid[2], avx_twiddle_len64_pair_const::<4, 6>(twiddle_ptr));
            mid[3] =
                avx_cmul_by_pair_twiddle(mid[3], avx_twiddle_len64_pair_const::<6, 9>(twiddle_ptr));
            mid[4] = avx_cmul_by_pair_twiddle(
                mid[4],
                avx_twiddle_len64_pair_const::<8, 12>(twiddle_ptr),
            );
            mid[5] = avx_cmul_by_pair_twiddle(
                mid[5],
                avx_twiddle_len64_pair_const::<10, 15>(twiddle_ptr),
            );
            mid[6] = avx_cmul_by_pair_twiddle(
                mid[6],
                avx_twiddle_len64_pair_const::<12, 18>(twiddle_ptr),
            );
            mid[7] = avx_cmul_by_pair_twiddle(
                mid[7],
                avx_twiddle_len64_pair_const::<14, 21>(twiddle_ptr),
            );
        }
        2 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len64_pair_const::<4, 5>(twiddle_ptr));
            mid[2] = avx_cmul_by_pair_twiddle(
                mid[2],
                avx_twiddle_len64_pair_const::<8, 10>(twiddle_ptr),
            );
            mid[3] = avx_cmul_by_pair_twiddle(
                mid[3],
                avx_twiddle_len64_pair_const::<12, 15>(twiddle_ptr),
            );
            mid[4] = avx_cmul_by_pair_twiddle(
                mid[4],
                avx_twiddle_len64_pair_const::<16, 20>(twiddle_ptr),
            );
            mid[5] = avx_cmul_by_pair_twiddle(
                mid[5],
                avx_twiddle_len64_pair_const::<20, 25>(twiddle_ptr),
            );
            mid[6] = avx_cmul_by_pair_twiddle(
                mid[6],
                avx_twiddle_len64_pair_const::<24, 30>(twiddle_ptr),
            );
            mid[7] = avx_cmul_by_pair_twiddle(
                mid[7],
                avx_twiddle_len64_pair_const::<28, 35>(twiddle_ptr),
            );
        }
        3 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len64_pair_const::<6, 7>(twiddle_ptr));
            mid[2] = avx_cmul_by_pair_twiddle(
                mid[2],
                avx_twiddle_len64_pair_const::<12, 14>(twiddle_ptr),
            );
            mid[3] = avx_cmul_by_pair_twiddle(
                mid[3],
                avx_twiddle_len64_pair_const::<18, 21>(twiddle_ptr),
            );
            mid[4] = avx_cmul_by_pair_twiddle(
                mid[4],
                avx_twiddle_len64_pair_const::<24, 28>(twiddle_ptr),
            );
            mid[5] = avx_cmul_by_pair_twiddle(
                mid[5],
                avx_twiddle_len64_pair_const::<30, 35>(twiddle_ptr),
            );
            mid[6] = avx_cmul_by_pair_twiddle(
                mid[6],
                avx_twiddle_len64_pair_const::<36, 42>(twiddle_ptr),
            );
            mid[7] = avx_cmul_by_pair_twiddle(
                mid[7],
                avx_twiddle_len64_pair_const::<42, 49>(twiddle_ptr),
            );
        }
        _ => unreachable!("length-64 column pair index must be in 0..4"),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn fixed_len64_first_phase_column_pair<const COLUMN_PAIR: usize>(
    data_ptr: *mut Complex64,
    scratch_ptr: *mut Complex64,
    twiddle_ptr: *const Complex64,
    quarter_turn_mask: std::arch::x86_64::__m256d,
    w1_re: std::arch::x86_64::__m256d,
    w1_im: std::arch::x86_64::__m256d,
    w3_re: std::arch::x86_64::__m256d,
    w3_im: std::arch::x86_64::__m256d,
) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};

    let column = COLUMN_PAIR << 1;
    let rows = [
        _mm256_loadu_pd(data_ptr.add(column).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 8).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 16).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 24).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 32).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 40).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 48).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 56).cast::<f64>()),
    ];
    let mut mid = avx_butterfly8_pair(rows, quarter_turn_mask, w1_re, w1_im, w3_re, w3_im);
    avx_apply_len64_twiddles_for_column_pair::<COLUMN_PAIR>(&mut mid, twiddle_ptr);
    let transposed = avx_transpose8_pairs(mid);
    let base = COLUMN_PAIR << 4;
    _mm256_storeu_pd(scratch_ptr.add(base).cast::<f64>(), transposed[0]);
    _mm256_storeu_pd(scratch_ptr.add(base + 2).cast::<f64>(), transposed[1]);
    _mm256_storeu_pd(scratch_ptr.add(base + 4).cast::<f64>(), transposed[2]);
    _mm256_storeu_pd(scratch_ptr.add(base + 6).cast::<f64>(), transposed[3]);
    _mm256_storeu_pd(scratch_ptr.add(base + 8).cast::<f64>(), transposed[4]);
    _mm256_storeu_pd(scratch_ptr.add(base + 10).cast::<f64>(), transposed[5]);
    _mm256_storeu_pd(scratch_ptr.add(base + 12).cast::<f64>(), transposed[6]);
    _mm256_storeu_pd(scratch_ptr.add(base + 14).cast::<f64>(), transposed[7]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn fixed_len64_second_phase_column_pair<const COLUMN_PAIR: usize>(
    data_ptr: *mut Complex64,
    scratch_ptr: *mut Complex64,
    quarter_turn_mask: std::arch::x86_64::__m256d,
    w1_re: std::arch::x86_64::__m256d,
    w1_im: std::arch::x86_64::__m256d,
    w3_re: std::arch::x86_64::__m256d,
    w3_im: std::arch::x86_64::__m256d,
) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};

    let column = COLUMN_PAIR << 1;
    let rows = [
        _mm256_loadu_pd(scratch_ptr.add(column).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 8).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 16).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 24).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 32).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 40).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 48).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 56).cast::<f64>()),
    ];
    let mid = avx_butterfly8_pair(rows, quarter_turn_mask, w1_re, w1_im, w3_re, w3_im);
    _mm256_storeu_pd(data_ptr.add(column).cast::<f64>(), mid[0]);
    _mm256_storeu_pd(data_ptr.add(column + 8).cast::<f64>(), mid[1]);
    _mm256_storeu_pd(data_ptr.add(column + 16).cast::<f64>(), mid[2]);
    _mm256_storeu_pd(data_ptr.add(column + 24).cast::<f64>(), mid[3]);
    _mm256_storeu_pd(data_ptr.add(column + 32).cast::<f64>(), mid[4]);
    _mm256_storeu_pd(data_ptr.add(column + 40).cast::<f64>(), mid[5]);
    _mm256_storeu_pd(data_ptr.add(column + 48).cast::<f64>(), mid[6]);
    _mm256_storeu_pd(data_ptr.add(column + 56).cast::<f64>(), mid[7]);
}

/// Fixed-size length-64 f64 AVX/FMA base butterfly.
///
/// The kernel evaluates a 64-point DFT by the Cooley-Tukey identity with
/// `N = 8 * 8`.  Let the input be `x[8r + c]`, where `0 <= r,c < 8`.
/// Phase one computes
///
/// `A[p,c] = Σ_r x[8r+c] W_8^{pr}`,
///
/// then multiplies by `W_64^{pc}` and stores the packed transpose in `scratch`.
/// Phase two computes
///
/// `X[8q+p] = Σ_c A[p,c] W_64^{pc} W_8^{qc}`,
///
/// because `W_8^{qc} = W_64^{8qc}`.  Therefore the total exponent is
/// `c(p + 8q) + 8rp`, exactly the mixed-radix decomposition of the DFT sum.
///
/// The vector lanes hold two adjacent matrix columns, so every butterfly is
/// two independent DFT-8 instances.  The only lane movement is the required
/// 8x2 packed transpose between the two mathematical phases; no standalone
/// global reordering pass is introduced.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn fixed_len64_precise_avx_fma(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    use std::arch::x86_64::{_mm256_set1_pd, _mm256_set_pd};

    debug_assert_eq!(data.len(), 64);
    debug_assert!(scratch.len() >= 64);
    debug_assert!(twiddles.len() >= 63);

    let twiddle_ptr = twiddles.as_ptr();
    let inverse = (*twiddle_ptr.add(2)).im > 0.0;
    let quarter_turn_mask = if inverse {
        _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
    } else {
        _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
    };
    let w1 = *twiddle_ptr.add(4);
    let w3 = *twiddle_ptr.add(6);
    let w1_re = _mm256_set1_pd(w1.re);
    let w1_im = _mm256_set1_pd(w1.im);
    let w3_re = _mm256_set1_pd(w3.re);
    let w3_im = _mm256_set1_pd(w3.im);
    let data_ptr = data.as_mut_ptr();
    let scratch_ptr = scratch.as_mut_ptr();

    fixed_len64_first_phase_column_pair::<0>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_first_phase_column_pair::<1>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_first_phase_column_pair::<2>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_first_phase_column_pair::<3>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );

    fixed_len64_second_phase_column_pair::<0>(
        data_ptr,
        scratch_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_second_phase_column_pair::<1>(
        data_ptr,
        scratch_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_second_phase_column_pair::<2>(
        data_ptr,
        scratch_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_second_phase_column_pair::<3>(
        data_ptr,
        scratch_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
}

#[inline]
unsafe fn twiddle_len32_const_unchecked<const EXPONENT: usize>(
    twiddle_ptr: *const Complex64,
) -> Complex64 {
    debug_assert!(EXPONENT < 32);
    match EXPONENT {
        0 => Complex64::new(1.0, 0.0),
        1..=15 => *twiddle_ptr.add(15 + EXPONENT),
        16 => Complex64::new(-1.0, 0.0),
        _ => {
            let w = *twiddle_ptr.add(15 + EXPONENT - 16);
            Complex64::new(-w.re, -w.im)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_twiddle_len32_pair_const<
    const FIRST_EXPONENT: usize,
    const SECOND_EXPONENT: usize,
>(
    twiddle_ptr: *const Complex64,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::_mm256_set_pd;

    let first = twiddle_len32_const_unchecked::<FIRST_EXPONENT>(twiddle_ptr);
    let second = twiddle_len32_const_unchecked::<SECOND_EXPONENT>(twiddle_ptr);
    _mm256_set_pd(second.im, second.re, first.im, first.re)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_apply_len32_twiddles_for_column_pair<const COLUMN_PAIR: usize>(
    mid: &mut [std::arch::x86_64::__m256d; 8],
    twiddle_ptr: *const Complex64,
) {
    match COLUMN_PAIR {
        0 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len32_pair_const::<0, 1>(twiddle_ptr));
            mid[2] =
                avx_cmul_by_pair_twiddle(mid[2], avx_twiddle_len32_pair_const::<0, 2>(twiddle_ptr));
            mid[3] =
                avx_cmul_by_pair_twiddle(mid[3], avx_twiddle_len32_pair_const::<0, 3>(twiddle_ptr));
            mid[4] =
                avx_cmul_by_pair_twiddle(mid[4], avx_twiddle_len32_pair_const::<0, 4>(twiddle_ptr));
            mid[5] =
                avx_cmul_by_pair_twiddle(mid[5], avx_twiddle_len32_pair_const::<0, 5>(twiddle_ptr));
            mid[6] =
                avx_cmul_by_pair_twiddle(mid[6], avx_twiddle_len32_pair_const::<0, 6>(twiddle_ptr));
            mid[7] =
                avx_cmul_by_pair_twiddle(mid[7], avx_twiddle_len32_pair_const::<0, 7>(twiddle_ptr));
        }
        1 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len32_pair_const::<2, 3>(twiddle_ptr));
            mid[2] =
                avx_cmul_by_pair_twiddle(mid[2], avx_twiddle_len32_pair_const::<4, 6>(twiddle_ptr));
            mid[3] =
                avx_cmul_by_pair_twiddle(mid[3], avx_twiddle_len32_pair_const::<6, 9>(twiddle_ptr));
            mid[4] = avx_cmul_by_pair_twiddle(
                mid[4],
                avx_twiddle_len32_pair_const::<8, 12>(twiddle_ptr),
            );
            mid[5] = avx_cmul_by_pair_twiddle(
                mid[5],
                avx_twiddle_len32_pair_const::<10, 15>(twiddle_ptr),
            );
            mid[6] = avx_cmul_by_pair_twiddle(
                mid[6],
                avx_twiddle_len32_pair_const::<12, 18>(twiddle_ptr),
            );
            mid[7] = avx_cmul_by_pair_twiddle(
                mid[7],
                avx_twiddle_len32_pair_const::<14, 21>(twiddle_ptr),
            );
        }
        _ => unreachable!("length-32 column pair index must be in 0..1"),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn fixed_len32_first_phase_column_pair<const COLUMN_PAIR: usize>(
    data_ptr: *mut Complex64,
    scratch_ptr: *mut Complex64,
    twiddle_ptr: *const Complex64,
    quarter_turn_mask: std::arch::x86_64::__m256d,
    w1_re: std::arch::x86_64::__m256d,
    w1_im: std::arch::x86_64::__m256d,
    w3_re: std::arch::x86_64::__m256d,
    w3_im: std::arch::x86_64::__m256d,
) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};

    let column = COLUMN_PAIR << 1;
    let rows = [
        _mm256_loadu_pd(data_ptr.add(column).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 4).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 8).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 12).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 16).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 20).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 24).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 28).cast::<f64>()),
    ];
    let mut mid = avx_butterfly8_pair(rows, quarter_turn_mask, w1_re, w1_im, w3_re, w3_im);
    avx_apply_len32_twiddles_for_column_pair::<COLUMN_PAIR>(&mut mid, twiddle_ptr);
    let offset = COLUMN_PAIR << 1;
    _mm256_storeu_pd(scratch_ptr.add(offset).cast::<f64>(), mid[0]);
    _mm256_storeu_pd(scratch_ptr.add(offset + 4).cast::<f64>(), mid[1]);
    _mm256_storeu_pd(scratch_ptr.add(offset + 8).cast::<f64>(), mid[2]);
    _mm256_storeu_pd(scratch_ptr.add(offset + 12).cast::<f64>(), mid[3]);
    _mm256_storeu_pd(scratch_ptr.add(offset + 16).cast::<f64>(), mid[4]);
    _mm256_storeu_pd(scratch_ptr.add(offset + 20).cast::<f64>(), mid[5]);
    _mm256_storeu_pd(scratch_ptr.add(offset + 24).cast::<f64>(), mid[6]);
    _mm256_storeu_pd(scratch_ptr.add(offset + 28).cast::<f64>(), mid[7]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn fixed_len32_second_phase_row_pair<const ROW_PAIR: usize>(
    data_ptr: *mut Complex64,
    scratch_ptr: *mut Complex64,
    quarter_turn_mask: std::arch::x86_64::__m256d,
) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_permute2f128_pd, _mm256_storeu_pd};

    let base = ROW_PAIR << 3;
    let mid0_0 = _mm256_loadu_pd(scratch_ptr.add(base).cast::<f64>());
    let mid1_0 = _mm256_loadu_pd(scratch_ptr.add(base + 2).cast::<f64>());
    let mid0_1 = _mm256_loadu_pd(scratch_ptr.add(base + 4).cast::<f64>());
    let mid1_1 = _mm256_loadu_pd(scratch_ptr.add(base + 6).cast::<f64>());

    let t0 = _mm256_permute2f128_pd(mid0_0, mid0_1, 0x20);
    let t1 = _mm256_permute2f128_pd(mid0_0, mid0_1, 0x31);
    let t2 = _mm256_permute2f128_pd(mid1_0, mid1_1, 0x20);
    let t3 = _mm256_permute2f128_pd(mid1_0, mid1_1, 0x31);

    let out = avx_butterfly4_pair([t0, t1, t2, t3], quarter_turn_mask);
    let p = ROW_PAIR << 1;
    _mm256_storeu_pd(data_ptr.add(p).cast::<f64>(), out[0]);
    _mm256_storeu_pd(data_ptr.add(p + 8).cast::<f64>(), out[1]);
    _mm256_storeu_pd(data_ptr.add(p + 16).cast::<f64>(), out[2]);
    _mm256_storeu_pd(data_ptr.add(p + 24).cast::<f64>(), out[3]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn fixed_len32_precise_avx_fma(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    use std::arch::x86_64::{_mm256_set1_pd, _mm256_set_pd};

    debug_assert_eq!(data.len(), 32);
    debug_assert!(scratch.len() >= 32);
    debug_assert!(twiddles.len() >= 31);

    let twiddle_ptr = twiddles.as_ptr();
    let inverse = (*twiddle_ptr.add(2)).im > 0.0;
    let quarter_turn_mask = if inverse {
        _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
    } else {
        _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
    };
    let w1 = *twiddle_ptr.add(4);
    let w3 = *twiddle_ptr.add(6);
    let w1_re = _mm256_set1_pd(w1.re);
    let w1_im = _mm256_set1_pd(w1.im);
    let w3_re = _mm256_set1_pd(w3.re);
    let w3_im = _mm256_set1_pd(w3.im);
    let data_ptr = data.as_mut_ptr();
    let scratch_ptr = scratch.as_mut_ptr();

    fixed_len32_first_phase_column_pair::<0>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len32_first_phase_column_pair::<1>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );

    fixed_len32_second_phase_row_pair::<0>(data_ptr, scratch_ptr, quarter_turn_mask);
    fixed_len32_second_phase_row_pair::<1>(data_ptr, scratch_ptr, quarter_turn_mask);
    fixed_len32_second_phase_row_pair::<2>(data_ptr, scratch_ptr, quarter_turn_mask);
    fixed_len32_second_phase_row_pair::<3>(data_ptr, scratch_ptr, quarter_turn_mask);
}
