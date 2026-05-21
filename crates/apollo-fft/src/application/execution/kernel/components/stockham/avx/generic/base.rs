use super::super::backend::StockhamAvxBackend;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn stage_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    radix: usize,
    twiddles: &[B::Complex],
) {
    let n = src.len();
    let half_n = n >> 1;
    let groups = n / (radix << 1);

    if groups == 1 {
        unsafe { B::stage_groups_one(src, dst, radix, twiddles) };
        return;
    }

    let vector_end = groups & !(B::COMPLEX_PER_VECTOR - 1);

    for j in 0..radix {
        let w = twiddles[j];
        let (w_re, w_im) = unsafe { B::unpack_complex(w) };
        let wr = unsafe { B::set1_real(w_re) };
        let wi = unsafe { B::set1_imag(w_im) };
        let src_base = j * groups * 2;
        let dst_base = j * groups;
        let mut k = 0usize;
        while k < vector_end {
            let a = unsafe { B::loadu_complex(src.as_ptr().add(src_base + k)) };
            let b = unsafe { B::loadu_complex(src.as_ptr().add(src_base + groups + k)) };
            let product = unsafe { B::cmul(wr, wi, b) };
            unsafe {
                B::storeu_complex(dst.as_mut_ptr().add(dst_base + k), B::add(a, product));
                B::storeu_complex(
                    dst.as_mut_ptr().add(dst_base + half_n + k),
                    B::sub(a, product),
                );
            }
            k += B::COMPLEX_PER_VECTOR;
        }
        while k < groups {
            let a = src[src_base + k];
            let b = unsafe { B::complex_mul(src[src_base + groups + k], w) };
            dst[dst_base + k] = unsafe { B::complex_add(a, b) };
            dst[dst_base + half_n + k] = unsafe { B::complex_sub(a, b) };
            k += 1;
        }
    }
}
