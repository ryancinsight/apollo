use super::cache::CompositeCache;
use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;
use crate::application::execution::kernel::winograd::apply_twiddle_impl;
use core::mem::MaybeUninit;
use num_complex::Complex;
use num_traits::Zero;

pub trait FusedStage {
    const R_TOTAL: usize;
    const DEPTH: usize;

    fn compute_group<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        inverse: bool,
    );
}

pub struct Radix<const R: usize>;

impl<const R: usize> FusedStage for Radix<R> {
    const R_TOTAL: usize = R;
    const DEPTH: usize = 1;

    #[inline(always)]
    fn compute_group<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        inverse: bool,
    ) {
        let stride = groups_out * prev_len;
        let src_base = b_out * prev_len;
        let stage_twiddles = twiddles[tw_idx];
        let mut j = 0;
        while j < prev_len {
            let mut buf = [Complex::<F>::zero(); R];
            let mut k = 0;
            while k < R {
                buf[k] = *unsafe { src.get_unchecked(k * stride + src_base + j) };
                k += 1;
            }
            if j > 0 {
                // stage_twiddles[j] = W_N^j; twiddle buf[k] by W_N^{j*k} via iterative multiply.
                let base_tw = *unsafe { stage_twiddles.get_unchecked(j) };
                let mut tw_k = base_tw;
                let mut k = 1;
                while k < R {
                    buf[k] = apply_twiddle_impl(buf[k], tw_k);
                    if k + 1 < R {
                        tw_k = apply_twiddle_impl(tw_k, base_tw);
                    }
                    k += 1;
                }
            }
            match R {
                2 => { let (lo, hi) = buf.split_at_mut(1); F::dft2(&mut lo[0], &mut hi[0]); }
                3 => { let ptr = buf.as_mut_ptr() as *mut [Complex<F>; 3]; F::dft3(unsafe { &mut *ptr }, inverse); }
                5 => { let ptr = buf.as_mut_ptr() as *mut [Complex<F>; 5]; F::dft5(unsafe { &mut *ptr }, inverse); }
                7 => { let ptr = buf.as_mut_ptr() as *mut [Complex<F>; 7]; F::dft7(unsafe { &mut *ptr }, inverse); }
                11 => { let ptr = buf.as_mut_ptr() as *mut [Complex<F>; 11]; F::dft11(unsafe { &mut *ptr }, inverse); }
                13 => { let ptr = buf.as_mut_ptr() as *mut [Complex<F>; 13]; if inverse { F::dft13::<true>(unsafe { &mut *ptr }) } else { F::dft13::<false>(unsafe { &mut *ptr }) } }
                17 => { let ptr = buf.as_mut_ptr() as *mut [Complex<F>; 17]; if inverse { F::dft17::<true>(unsafe { &mut *ptr }) } else { F::dft17::<false>(unsafe { &mut *ptr }) } }
                23 => { let ptr = buf.as_mut_ptr() as *mut [Complex<F>; 23]; if inverse { F::dft23::<true>(unsafe { &mut *ptr }) } else { F::dft23::<false>(unsafe { &mut *ptr }) } }
                _ => unreachable!(),
            }
            let mut k = 0;
            while k < R {
                *unsafe { dst.get_unchecked_mut(j + k * prev_len) } = buf[k];
                k += 1;
            }
            j += 1;
        }
    }
}

pub struct Compose<Inner, Outer>(std::marker::PhantomData<(Inner, Outer)>);

impl<Inner: FusedStage, Outer: FusedStage> FusedStage for Compose<Inner, Outer> {
    const R_TOTAL: usize = Inner::R_TOTAL * Outer::R_TOTAL;
    const DEPTH: usize = Inner::DEPTH + Outer::DEPTH;

    #[inline(always)]
    fn compute_group<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        inverse: bool,
    ) {
        let inner_out_len = Inner::R_TOTAL * prev_len;
        let inner_groups_out = Outer::R_TOTAL * groups_out;
        let total_mid = Outer::R_TOTAL * inner_out_len;
        // SAFETY: Inner::compute_group writes every element of each segment before
        // Outer::compute_group reads from mid. MaybeUninit avoids the 4 KiB memset
        // that [Complex::zero(); 256] would emit for f64 precision.
        let mut mid_uninit: [MaybeUninit<Complex<F>>; 256] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mid: &mut [Complex<F>] = unsafe {
            core::slice::from_raw_parts_mut(mid_uninit.as_mut_ptr().cast::<Complex<F>>(), total_mid)
        };
        let mut b_inner = 0;
        while b_inner < Outer::R_TOTAL {
            let b_inner_global = b_out + b_inner * groups_out;
            Inner::compute_group::<F>(
                src,
                &mut mid[b_inner * inner_out_len..(b_inner + 1) * inner_out_len],
                prev_len,
                b_inner_global,
                inner_groups_out,
                twiddles,
                tw_idx,
                inverse,
            );
            b_inner += 1;
        }
        Outer::compute_group::<F>(
            mid,
            dst,
            inner_out_len,
            0,
            1,
            twiddles,
            tw_idx + Inner::DEPTH,
            inverse,
        );
    }
}
