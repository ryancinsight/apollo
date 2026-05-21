
use num_complex::Complex;
use super::arity::{Compose, Radix, FusedStage};
use super::cache::CompositeCache;
use crate::application::execution::policy::{SyncPolicy, ExecutionPolicy};
use crate::application::execution::kernel::tuning::FUSE_THRESHOLD;
use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;
use super::core::stockham_stage_fused;

#[inline(always)]
pub(super) fn try_fuse_combinations<F: CompositeCache + ShortWinogradScalar>(
    r1: usize,
    radices: &[usize],
    stage_idx: &mut usize,
    prev_len: &mut usize,
    src_is_data: &mut bool,
    data: &mut [Complex<F>],
    scratch: &mut [Complex<F>],
    all_twiddles: &[Complex<F>],
    stage_offsets: &[usize],
    inverse: bool,
) -> bool {
    let offset1 = stage_offsets[*stage_idx];
    let tw1 = &all_twiddles[offset1..offset1 + (r1 - 1) * (*prev_len)];




    false
}
