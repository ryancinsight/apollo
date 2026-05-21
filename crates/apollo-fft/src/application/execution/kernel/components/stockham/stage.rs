//! Scalar Stockham butterfly stage primitives and L1 residency thresholds.

#![allow(clippy::many_single_char_names)]

use num_complex::{Complex32, Complex64};

#[cfg(target_arch = "x86_64")]
const STOCKHAM_PRECISE_L1_RESIDENT_BYTES: usize = 32 * 1024;

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn stockham_precise_stage_is_l1_resident(n: usize) -> bool {
    n <= STOCKHAM_PRECISE_L1_RESIDENT_BYTES / (core::mem::size_of::<Complex64>() * 2)
}

/// L1 residency threshold for f32 triple-stage dispatch.
///
/// A three-stage Stockham pass reads src[0..n] and writes dst[0..n]. Both
/// buffers must fit in L1 to exploit the low-live codelet; 32 KiB gives 2048
/// `Complex32` elements across the two buffers.
#[cfg(target_arch = "x86_64")]
const STOCKHAM_REDUCED_L1_RESIDENT_BYTES: usize = 32 * 1024;

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn stockham_reduced_stage_is_l1_resident(n: usize) -> bool {
    n <= STOCKHAM_REDUCED_L1_RESIDENT_BYTES / (core::mem::size_of::<Complex32>() * 2)
}

#[inline]
pub(crate) fn stage_impl<C, const TILE_SIZE: usize>(
    src: &[C],
    dst: &mut [C],
    radix: usize,
    twiddles: &[C],
) where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    let n = src.len();
    let half_n = n >> 1;
    let groups = n / (radix << 1);

    let mut k_start = 0;
    while k_start < groups {
        let k_end = (k_start + TILE_SIZE).min(groups);
        for j in 0..radix {
            let w = twiddles[j];
            let src_base = j * groups * 2;
            let dst_base = j * groups;
            // SUBTILE-4: 4 independent butterfly pairs per iteration expose
            // 4 independent complex multiplies to the OOO execution engine.
            let k_end4 = k_start + ((k_end - k_start) / 4) * 4;
            let mut k = k_start;
            while k < k_end4 {
                let a0 = src[src_base + k];
                let a1 = src[src_base + k + 1];
                let a2 = src[src_base + k + 2];
                let a3 = src[src_base + k + 3];
                let b0 = src[src_base + groups + k] * w;
                let b1 = src[src_base + groups + k + 1] * w;
                let b2 = src[src_base + groups + k + 2] * w;
                let b3 = src[src_base + groups + k + 3] * w;
                dst[dst_base + k] = a0 + b0;
                dst[dst_base + k + 1] = a1 + b1;
                dst[dst_base + k + 2] = a2 + b2;
                dst[dst_base + k + 3] = a3 + b3;
                dst[dst_base + half_n + k] = a0 - b0;
                dst[dst_base + half_n + k + 1] = a1 - b1;
                dst[dst_base + half_n + k + 2] = a2 - b2;
                dst[dst_base + half_n + k + 3] = a3 - b3;
                k += 4;
            }
            while k < k_end {
                let a = src[src_base + k];
                let b = src[src_base + groups + k] * w;
                dst[dst_base + k] = a + b;
                dst[dst_base + half_n + k] = a - b;
                k += 1;
            }
        }
        k_start += TILE_SIZE;
    }
}
