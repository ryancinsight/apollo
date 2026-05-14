use super::super::backend::StockhamAvxBackend;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn stage_pair_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    radix: usize,
    first_twiddles: &[B::Complex],
    second_twiddles: &[B::Complex],
) {
    let n = src.len();
    let groups = n / (radix << 1);
    let half_groups = groups >> 1;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    let vector_end = half_groups & !(B::COMPLEX_PER_VECTOR - 1);

    for j in 0..radix {
        let w1 = first_twiddles[j];
        let w2 = second_twiddles[j];
        let w3 = second_twiddles[j + radix];
        let (w1_re, w1_im) = unsafe { B::unpack_complex(w1) };
        let (w2_re, w2_im) = unsafe { B::unpack_complex(w2) };
        let (w3_re, w3_im) = unsafe { B::unpack_complex(w3) };
        let w1r = unsafe { B::set1_real(w1_re) };
        let w1i = unsafe { B::set1_imag(w1_im) };
        let w2r = unsafe { B::set1_real(w2_re) };
        let w2i = unsafe { B::set1_imag(w2_im) };
        let w3r = unsafe { B::set1_real(w3_re) };
        let w3i = unsafe { B::set1_imag(w3_im) };
        let src_base = j * groups * 2;
        let dst_base = j * half_groups;
        let mut k = 0usize;
        while k < vector_end {
            let x0 = unsafe { B::loadu_complex(src.as_ptr().add(src_base + k)) };
            let x1 = unsafe { B::loadu_complex(src.as_ptr().add(src_base + half_groups + k)) };
            let raw_x2 = unsafe { B::loadu_complex(src.as_ptr().add(src_base + groups + k)) };
            let raw_x3 =
                unsafe { B::loadu_complex(src.as_ptr().add(src_base + groups + half_groups + k)) };
            let x2 = unsafe { B::cmul(w1r, w1i, raw_x2) };
            let x3 = unsafe { B::cmul(w1r, w1i, raw_x3) };
            let a0 = unsafe { B::add(x0, x2) };
            let a1 = unsafe { B::add(x1, x3) };
            let b0 = unsafe { B::sub(x0, x2) };
            let b1 = unsafe { B::sub(x1, x3) };
            let c0 = unsafe { B::cmul(w2r, w2i, a1) };
            let c1 = unsafe { B::cmul(w3r, w3i, b1) };
            unsafe {
                B::storeu_complex(dst.as_mut_ptr().add(dst_base + k), B::add(a0, c0));
                B::storeu_complex(dst.as_mut_ptr().add(dst_base + half_n + k), B::sub(a0, c0));
                B::storeu_complex(
                    dst.as_mut_ptr().add(dst_base + quarter_n + k),
                    B::add(b0, c1),
                );
                B::storeu_complex(
                    dst.as_mut_ptr().add(dst_base + half_n + quarter_n + k),
                    B::sub(b0, c1),
                );
            }
            k += B::COMPLEX_PER_VECTOR;
        }
        while k < half_groups {
            let x0 = src[src_base + k];
            let x1 = src[src_base + half_groups + k];
            let x2 = unsafe { B::complex_mul(src[src_base + groups + k], w1) };
            let x3 = unsafe { B::complex_mul(src[src_base + groups + half_groups + k], w1) };
            let a0 = unsafe { B::complex_add(x0, x2) };
            let a1 = unsafe { B::complex_add(x1, x3) };
            let b0 = unsafe { B::complex_sub(x0, x2) };
            let b1 = unsafe { B::complex_sub(x1, x3) };
            let c0 = unsafe { B::complex_mul(a1, w2) };
            let c1 = unsafe { B::complex_mul(b1, w3) };
            dst[dst_base + k] = unsafe { B::complex_add(a0, c0) };
            dst[dst_base + half_n + k] = unsafe { B::complex_sub(a0, c0) };
            dst[dst_base + quarter_n + k] = unsafe { B::complex_add(b0, c1) };
            dst[dst_base + half_n + quarter_n + k] = unsafe { B::complex_sub(b0, c1) };
            k += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn stage_pair_radix1_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    second_twiddles: &[B::Complex],
) {
    let n = src.len();
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    let vector_end = quarter_n & !(B::COMPLEX_PER_VECTOR - 1);
    let w3 = second_twiddles[1];
    let (w3_re, w3_im) = unsafe { B::unpack_complex(w3) };
    let w3r = unsafe { B::set1_real(w3_re) };
    let w3i = unsafe { B::set1_imag(w3_im) };

    let mut k = 0usize;
    while k < vector_end {
        let x0 = unsafe { B::loadu_complex(src.as_ptr().add(k)) };
        let x1 = unsafe { B::loadu_complex(src.as_ptr().add(quarter_n + k)) };
        let x2 = unsafe { B::loadu_complex(src.as_ptr().add(half_n + k)) };
        let x3 = unsafe { B::loadu_complex(src.as_ptr().add(half_n + quarter_n + k)) };
        let a0 = unsafe { B::add(x0, x2) };
        let a1 = unsafe { B::add(x1, x3) };
        let b0 = unsafe { B::sub(x0, x2) };
        let b1 = unsafe { B::sub(x1, x3) };
        let c1 = unsafe { B::cmul(w3r, w3i, b1) };
        unsafe {
            B::storeu_complex(dst.as_mut_ptr().add(k), B::add(a0, a1));
            B::storeu_complex(dst.as_mut_ptr().add(half_n + k), B::sub(a0, a1));
            B::storeu_complex(dst.as_mut_ptr().add(quarter_n + k), B::add(b0, c1));
            B::storeu_complex(dst.as_mut_ptr().add(half_n + quarter_n + k), B::sub(b0, c1));
        }
        k += B::COMPLEX_PER_VECTOR;
    }
    while k < quarter_n {
        let x0 = src[k];
        let x1 = src[quarter_n + k];
        let x2 = src[half_n + k];
        let x3 = src[half_n + quarter_n + k];
        let a0 = unsafe { B::complex_add(x0, x2) };
        let a1 = unsafe { B::complex_add(x1, x3) };
        let b0 = unsafe { B::complex_sub(x0, x2) };
        let b1 = unsafe { B::complex_sub(x1, x3) };
        let c1 = unsafe { B::complex_mul(b1, w3) };
        dst[k] = unsafe { B::complex_add(a0, a1) };
        dst[half_n + k] = unsafe { B::complex_sub(a0, a1) };
        dst[quarter_n + k] = unsafe { B::complex_add(b0, c1) };
        dst[half_n + quarter_n + k] = unsafe { B::complex_sub(b0, c1) };
        k += 1;
    }
}
