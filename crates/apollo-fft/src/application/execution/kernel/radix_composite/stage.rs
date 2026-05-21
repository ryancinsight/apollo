//! Single-stage Stockham butterfly and inner DFT-r dispatch.
//!
//! Implements the per-stage read/compute/write kernel for the mixed-radix
//! out-of-place Stockham autosort. Separated from the top-level dispatch
//! (`mod.rs`) so each file stays within the 500-line structural limit.

use num_complex::Complex;

use super::super::winograd::{
    apply_twiddle_impl, dft2_impl, dft3_impl, dft4_impl, dft5_impl, dft7_impl, dft8_impl,
    WinogradScalar,
};

use super::super::tuning::RADIX_PARALLEL_CHUNK_THRESHOLD;

use rayon::prelude::*;

// ── inner butterfly dispatchers ───────────────────────────────────────────────

#[inline]
pub(super) fn apply_dft_r_impl<F: WinogradScalar>(
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
            dft7_impl(&mut data[..7], inverse);
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

/// Single out-of-place Stockham butterfly stage.
///
/// Reads from `src` with stride `groups * prev_len`, writes to `dst` in
/// contiguous `stage_len`-element blocks. Safe to call with `src == data` and
/// `dst == scratch` or vice-versa — the two slices must not alias.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn stockham_stage<F: WinogradScalar>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    r: usize,
    prev_len: usize,
    groups: usize,
    stage_len: usize,
    stage_twiddles: &[Complex<F>],
    inverse: bool,
    use_parallel: bool,
) {
    let stride = groups * prev_len;
    if use_parallel {
        // Parallel over output blocks (b index). `src` is shared read-only.
        dst.par_chunks_mut(stage_len)
            .enumerate()
            .for_each(|(b, dst_block)| {
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
    } else {
        let mut buf = [Complex::new(F::cast_f64(0.0), F::cast_f64(0.0)); 8];
        for b in 0..groups {
            let src_base = b * prev_len;
            let dst_block = &mut dst[b * stage_len..(b + 1) * stage_len];
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
        }
    }
}

/// Process one output block `b` of size `stage_len` for a single Stockham stage.
///
/// j=0 fast path: W^0 = 1 — gather, DFT-r, scatter with no multiply.
/// j>0 path: recurrence-based twiddle application.
#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // k indexes both buf and stride-computed src/dst positions
pub(super) fn stockham_block<F: WinogradScalar>(
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
    // ── j = 0: W^0 = 1, no multiply ─────────────────────────────────────────
    for k in 0..r {
        // SAFETY: stride * k + src_base < n by loop invariant on b and stage layout.
        buf[k] = *unsafe { src.get_unchecked(k * stride + src_base) };
    }
    apply_dft_r_impl(&mut buf[..r], r, inverse);
    for k in 0..r {
        // SAFETY: k * prev_len < stage_len = prev_len * r.
        *unsafe { dst_block.get_unchecked_mut(k * prev_len) } = buf[k];
    }

    // ── j = 1..prev_len: with twiddle ────────────────────────────────────────
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

/// Parallel-threshold check reused by `composite_core_with_radices`.
#[inline]
pub(super) fn use_parallel_stage(n: usize, stage_len: usize, groups: usize) -> bool {
    n >= RADIX_PARALLEL_CHUNK_THRESHOLD && stage_len >= 512 && groups >= 4
}
