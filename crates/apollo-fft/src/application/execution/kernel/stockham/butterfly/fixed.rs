#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
use num_complex::{Complex32, Complex64};
use super::hybrid::*;
use super::super::precision::*;
use super::super::avx::*;
use super::super::transform::*;
pub(crate) unsafe fn forward64_avx_with_scratch(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    if data.len() == 64 {
        fixed_len64_avx_fma(data, scratch, twiddles);
        return;
    }
    if data.len() == 4096 && twiddles.get(1).is_some_and(|w| w.im < 0.0) {
        transform_len4096_four_triples::<F64StockhamAvxFma>(data, scratch, twiddles);
        return;
    }
    transform::<F64StockhamAvxFma>(data, scratch, twiddles, None);
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn fixed_len512_avx_fma(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;
    // Pass 1: radix 1, groups 256
    for j in 0..1 {
        let w = twiddles[j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 512;
        let dst_base = j * 256;
        let mut k = 0;
        while k < 256 {
            let x0 = _mm256_loadu_pd(data.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 256 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(),
                t,
            );
            k += 2;
        }
    }
    // Pass 2: radix 2, groups 128
    for j in 0..2 {
        let w = twiddles[1 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 256;
        let dst_base = j * 128;
        let mut k = 0;
        while k < 128 {
            let x0 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + 128 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(), t);
            k += 2;
        }
    }
    // Pass 3: radix 4, groups 64
    for j in 0..4 {
        let w = twiddles[3 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 128;
        let dst_base = j * 64;
        let mut k = 0;
        while k < 64 {
            let x0 = _mm256_loadu_pd(data.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 64 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(),
                t,
            );
            k += 2;
        }
    }
    // Pass 4: radix 8, groups 32
    for j in 0..8 {
        let w = twiddles[7 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 64;
        let dst_base = j * 32;
        let mut k = 0;
        while k < 32 {
            let x0 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + 32 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(), t);
            k += 2;
        }
    }
    // Pass 5: radix 16, groups 16
    for j in 0..16 {
        let w = twiddles[15 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 32;
        let dst_base = j * 16;
        let mut k = 0;
        while k < 16 {
            let x0 = _mm256_loadu_pd(data.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 16 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(),
                t,
            );
            k += 2;
        }
    }
    // Pass 6: radix 32, groups 8
    for j in 0..32 {
        let w = twiddles[31 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 16;
        let dst_base = j * 8;
        let mut k = 0;
        while k < 8 {
            let x0 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + 8 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(), t);
            k += 2;
        }
    }
    // Pass 7: radix 64, groups 4
    for j in 0..64 {
        let w = twiddles[63 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 8;
        let dst_base = j * 4;
        let mut k = 0;
        while k < 4 {
            let x0 = _mm256_loadu_pd(data.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 4 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(),
                t,
            );
            k += 2;
        }
    }
    // Pass 8: radix 128, groups 2
    for j in 0..128 {
        let w = twiddles[127 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 4;
        let dst_base = j * 2;
        let mut k = 0;
        while k < 2 {
            let x0 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + 2 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(), t);
            k += 2;
        }
    }
    // Pass 9: radix 256, groups 1
    let mut j = 0;
    while j < 256 {
        let w0 = twiddles[255 + j];
        let w1 = twiddles[255 + j + 1];
        let w_re = _mm256_set_pd(w1.re, w1.re, w0.re, w0.re);
        let w_im = _mm256_set_pd(w1.im, w1.im, w0.im, w0.im);
        let src_base = j * 2;
        let dst_base = j;
        let d0 = _mm256_loadu_pd(data.as_ptr().add(src_base).cast::<f64>());
        let d1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 2).cast::<f64>());
        let x0 = _mm256_permute2f128_pd(d0, d1, 0x20);
        let x1 = _mm256_permute2f128_pd(d0, d1, 0x31);
        let x1_shuf = _mm256_permute_pd(x1, 0b0101);
        let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
        let s = _mm256_add_pd(x0, product);
        let t = _mm256_sub_pd(x0, product);
        _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base).cast::<f64>(), s);
        _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + 256).cast::<f64>(), t);
        j += 2;
    }
    data.copy_from_slice(scratch);
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn fixed_len512_32_avx_fma(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;
    // Pass 1: radix 1, groups 256
    for j in 0..1 {
        let w = twiddles[j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 512;
        let dst_base = j * 256;
        let mut k = 0;
        while k < 256 {
            let x0 = _mm256_loadu_ps(data.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(data.as_ptr().add(src_base + 256 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(),
                t,
            );
            k += 4;
        }
    }
    // Pass 2: radix 2, groups 128
    for j in 0..2 {
        let w = twiddles[1 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 256;
        let dst_base = j * 128;
        let mut k = 0;
        while k < 128 {
            let x0 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + 128 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(), t);
            k += 4;
        }
    }
    // Pass 3: radix 4, groups 64
    for j in 0..4 {
        let w = twiddles[3 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 128;
        let dst_base = j * 64;
        let mut k = 0;
        while k < 64 {
            let x0 = _mm256_loadu_ps(data.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(data.as_ptr().add(src_base + 64 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(),
                t,
            );
            k += 4;
        }
    }
    // Pass 4: radix 8, groups 32
    for j in 0..8 {
        let w = twiddles[7 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 64;
        let dst_base = j * 32;
        let mut k = 0;
        while k < 32 {
            let x0 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + 32 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(), t);
            k += 4;
        }
    }
    // Pass 5: radix 16, groups 16
    for j in 0..16 {
        let w = twiddles[15 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 32;
        let dst_base = j * 16;
        let mut k = 0;
        while k < 16 {
            let x0 = _mm256_loadu_ps(data.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(data.as_ptr().add(src_base + 16 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(),
                t,
            );
            k += 4;
        }
    }
    // Pass 6: radix 32, groups 8
    for j in 0..32 {
        let w = twiddles[31 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 16;
        let dst_base = j * 8;
        let mut k = 0;
        while k < 8 {
            let x0 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + 8 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(), t);
            k += 4;
        }
    }
    // Passes 7-8: radix 64 then radix 128.
    stage_pair32_quarter_groups_two_avx_fma(
        data,
        scratch,
        64,
        &twiddles[63..127],
        &twiddles[127..255],
    );
    // Pass 9: radix 256, groups 1.
    stage32_groups_one_avx_fma(scratch, data, 256, &twiddles[255..511]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len8_32_avx_fma(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;
    // Pass 1: radix 1, groups 4
    for j in 0..1 {
        let w = twiddles[j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 8;
        let dst_base = j * 4;
        let mut k = 0;
        while k < 4 {
            let x0 = _mm256_loadu_ps(data.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(data.as_ptr().add(src_base + 4 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + 4 + k).cast::<f32>(), t);
            k += 4;
        }
    }
    // Pass 2: radix 2, groups 2
    for j in 0..2 {
        let w = twiddles[1 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 4;
        let dst_base = j * 2;
        let src_vec = _mm256_loadu_ps(scratch.as_ptr().add(src_base).cast::<f32>());
        let x0 = _mm256_permute2f128_ps(src_vec, src_vec, 0x00);
        let x1 = _mm256_permute2f128_ps(src_vec, src_vec, 0x11);
        let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
        let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
        let s = _mm256_add_ps(x0, product);
        let t = _mm256_sub_ps(x0, product);
        _mm_storeu_ps(
            data.as_mut_ptr().add(dst_base).cast::<f32>(),
            _mm256_castps256_ps128(s),
        );
        _mm_storeu_ps(
            data.as_mut_ptr().add(dst_base + 4).cast::<f32>(),
            _mm256_castps256_ps128(t),
        );
    }
    // Pass 3: radix 4, groups 1
    for j in 0..4 {
        let w = twiddles[3 + j];
        let a = data[j * 2];
        let b = data[j * 2 + 1] * w;
        scratch[j] = a + b;
        scratch[j + 4] = a - b;
    }
    data.copy_from_slice(scratch);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len4_32_avx_fma(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;
    // Pass 1: radix 1, groups 2
    for j in 0..1 {
        let w = twiddles[j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 4;
        let dst_base = j * 2;
        let src_vec = _mm256_loadu_ps(data.as_ptr().add(src_base).cast::<f32>());
        let x0 = _mm256_permute2f128_ps(src_vec, src_vec, 0x00);
        let x1 = _mm256_permute2f128_ps(src_vec, src_vec, 0x11);
        let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
        let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
        let s = _mm256_add_ps(x0, product);
        let t = _mm256_sub_ps(x0, product);
        _mm_storeu_ps(
            scratch.as_mut_ptr().add(dst_base).cast::<f32>(),
            _mm256_castps256_ps128(s),
        );
        _mm_storeu_ps(
            scratch.as_mut_ptr().add(dst_base + 2).cast::<f32>(),
            _mm256_castps256_ps128(t),
        );
    }
    // Pass 2: radix 2, groups 1
    for j in 0..2 {
        let w = twiddles[1 + j];
        let a = scratch[j * 2];
        let b = scratch[j * 2 + 1] * w;
        data[j] = a + b;
        data[j + 2] = a - b;
    }
}

#[inline]
#[expect(clippy::too_many_arguments, reason = "register-blocked fused codelet")]
pub(crate) unsafe fn forward32_avx_with_scratch(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    if data.len() == 4 {
        fixed_len4_32_avx_fma(data, scratch, twiddles);
        return;
    }
    if data.len() == 8 {
        fixed_len8_32_avx_fma(data, scratch, twiddles);
        return;
    }
    if data.len() == 64 {
        fixed_len64_32_avx_fma(data, scratch, twiddles);
        return;
    }
    if data.len() == 4096 {
        transform_len4096_four_triples::<F32StockhamAvxFma>(data, scratch, twiddles);
        return;
    }
    transform::<F32StockhamAvxFma>(data, scratch, twiddles, None);
}
