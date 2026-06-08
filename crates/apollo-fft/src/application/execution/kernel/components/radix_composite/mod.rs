use num_complex::Complex;

mod adaptive;
pub(crate) mod arity;
#[cfg(target_arch = "x86_64")]
mod avx2;
mod cache;
mod core;

use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;
use crate::application::execution::kernel::radix_stage::normalize_scalar;
use crate::application::execution::policy::ChunkDispatch;
pub use cache::CompositeCache;

/// Execute a fused multi-stage Stockham composite pass over all output groups.
///
/// Each output group is `prev_len * r_total` elements. `dispatch` controls
/// whether groups run sequentially or through Moirai's work-stealing scheduler.
/// `composite_fused_adaptive` handles the per-group multi-stage recursion using
/// the thread-local bump arena.
#[inline]
pub(super) fn stockham_stage_fused_adaptive<F, const INVERSE: bool>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    prev_len: usize,
    radices: &[usize],
    twiddles: &[&[Complex<F>]],
    pointwise_spectrum: Option<&[Complex<F>]>,
    dispatch: ChunkDispatch,
) where
    F: CompositeCache + ShortWinogradScalar,
{
    let r_total: usize = radices.iter().product();
    let stage_len = prev_len * r_total;
    let groups = src.len() / stage_len;
    crate::application::execution::policy::for_each_chunk_mut_enumerated(
        dst,
        stage_len,
        dispatch,
        |b, dst_block| {
            let pw = pointwise_spectrum.map(|ps| &ps[b * stage_len..(b + 1) * stage_len]);
            adaptive::composite_fused_adaptive::<F, INVERSE>(
                src, dst_block, prev_len, b, groups, radices, twiddles, pw,
            );
        },
    );
}

pub fn forward_inplace_with_radices<F: CompositeCache + ShortWinogradScalar + 'static>(
    data: &mut [Complex<F>],
    radices: &[usize],
) {
    core::composite_core_with_radices::<F, false>(data, radices, None);
}

#[inline]
pub fn forward_inplace_with_pointwise<F: CompositeCache + ShortWinogradScalar + 'static>(
    data: &mut [Complex<F>],
    radices: &[usize],
    pointwise_spectrum: &[Complex<F>],
) {
    core::composite_core_with_radices::<F, false>(data, radices, Some(pointwise_spectrum));
}

#[inline]
pub fn inverse_inplace_unnorm_with_radices<F: CompositeCache + ShortWinogradScalar + 'static>(
    data: &mut [Complex<F>],
    radices: &[usize],
) {
    core::composite_core_with_radices::<F, true>(data, radices, None);
}

#[inline]
pub fn inverse_inplace_with_radices<F: CompositeCache + ShortWinogradScalar + 'static>(
    data: &mut [Complex<F>],
    radices: &[usize],
) {
    core::composite_core_with_radices::<F, true>(data, radices, None);
    normalize_scalar(data, F::from_precise(1.0 / data.len() as f64));
}

#[cfg(test)]
mod tests;
