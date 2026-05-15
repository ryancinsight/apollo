use crate::application::execution::kernel::winograd::{
    apply_twiddle_impl, dft2_impl, dft3_impl, dft4_impl, dft5_impl, dft7_impl, dft8_impl,
    WinogradScalar,
};
use num_complex::Complex;

#[inline]
pub(crate) fn apply_dft_r_impl<F: WinogradScalar>(
    data: &mut [Complex<F>],
    r: usize,
    inverse: bool,
) {
    match r {
        2 => {
            let (lo, hi) = data.split_at_mut(1);
            dft2_impl(&mut lo[0], &mut hi[0]);
        }
        3 => {
            let mut b = [data[0], data[1], data[2]];
            dft3_impl(&mut b, inverse);
            data[..3].copy_from_slice(&b);
        }
        4 => {
            let mut b = [data[0], data[1], data[2], data[3]];
            dft4_impl(&mut b, inverse);
            data[..4].copy_from_slice(&b);
        }
        5 => {
            let mut b = [data[0], data[1], data[2], data[3], data[4]];
            dft5_impl(&mut b, inverse);
            data[..5].copy_from_slice(&b);
        }
        7 => {
            let mut b = [
                data[0], data[1], data[2], data[3], data[4], data[5], data[6],
            ];
            dft7_impl(&mut b, inverse);
            data[..7].copy_from_slice(&b);
        }
        8 => {
            let mut b = [
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ];
            dft8_impl(&mut b, inverse);
            data[..8].copy_from_slice(&b);
        }
        _ => unreachable!("unsupported radix {r}"),
    }
}

use crate::application::execution::policy::ExecutionPolicy;

#[inline]
pub(crate) fn stockham_stage<F: WinogradScalar, P: ExecutionPolicy>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    r: usize,
    prev_len: usize,
    groups: usize,
    stage_len: usize,
    stage_twiddles: &[Complex<F>],
    inverse: bool,
) {
    let stride = groups * prev_len;
    P::for_each_chunk_mut_enumerated(dst, stage_len, |b, dst_block| {
        let mut buf = [Complex::new(F::cast_f64(0.0), F::cast_f64(0.0)); 8];
        let src_base = b * prev_len;
        stockham_block(
            src,
            dst_block,
            r,
            prev_len,
            stride,
            stage_twiddles,
            inverse,
            src_base,
            &mut buf,
        );
    });
}

#[inline]
fn stockham_block<F: WinogradScalar>(
    src: &[Complex<F>],
    dst_block: &mut [Complex<F>],
    r: usize,
    prev_len: usize,
    stride: usize,
    stage_twiddles: &[Complex<F>],
    inverse: bool,
    src_base: usize,
    buf: &mut [Complex<F>; 8],
) {
    for k in 0..r {
        buf[k] = *unsafe { src.get_unchecked(k * stride + src_base) };
    }
    apply_dft_r_impl(&mut buf[..r], r, inverse);
    for k in 0..r {
        *unsafe { dst_block.get_unchecked_mut(k * prev_len) } = buf[k];
    }

    for j in 1..prev_len {
        for k in 0..r {
            buf[k] = *unsafe { src.get_unchecked(k * stride + src_base + j) };
        }
        let base_tw = *unsafe { stage_twiddles.get_unchecked(j) };
        let mut tw_k = base_tw;
        for k in 1..r {
            buf[k] = apply_twiddle_impl(buf[k], tw_k);
            if k + 1 < r {
                tw_k = apply_twiddle_impl(tw_k, base_tw);
            }
        }
        apply_dft_r_impl(&mut buf[..r], r, inverse);
        for k in 0..r {
            *unsafe { dst_block.get_unchecked_mut(j + k * prev_len) } = buf[k];
        }
    }
}
