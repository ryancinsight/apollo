//! Scalar Stockham butterfly stage primitives and L1 residency thresholds.

#![allow(clippy::many_single_char_names)]

use num_complex::{Complex32, Complex64};

#[cfg(target_arch = "x86_64")]
const STOCKHAM_F64_L1_RESIDENT_BYTES: usize = 32 * 1024;

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn stockham_f64_stage_is_l1_resident(n: usize) -> bool {
    n <= STOCKHAM_F64_L1_RESIDENT_BYTES / (core::mem::size_of::<Complex64>() * 2)
}

/// L1 residency threshold for f32 triple-stage dispatch.
///
/// A three-stage Stockham pass reads src[0..n] and writes dst[0..n]. Both
/// buffers must fit in L1 to exploit the low-live codelet; 32 KiB gives 2048
/// `Complex32` elements across the two buffers.
#[cfg(target_arch = "x86_64")]
const STOCKHAM_F32_L1_RESIDENT_BYTES: usize = 32 * 1024;

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn stockham_f32_stage_is_l1_resident(n: usize) -> bool {
    n <= STOCKHAM_F32_L1_RESIDENT_BYTES / (core::mem::size_of::<Complex32>() * 2)
}

#[inline]
pub(crate) fn stage_impl<C>(src: &[C], dst: &mut [C], radix: usize, twiddles: &[C])
where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    let n = src.len();
    let half_n = n >> 1;
    let groups = n / (radix << 1);
    for j in 0..radix {
        let w = twiddles[j];
        let src_base = j * groups * 2;
        let dst_base = j * groups;
        for k in 0..groups {
            let a = src[src_base + k];
            let b = src[src_base + groups + k] * w;
            dst[dst_base + k] = a + b;
            dst[dst_base + half_n + k] = a - b;
        }
    }
}
