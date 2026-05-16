use num_complex::Complex;

mod arity;
mod butterfly;
mod cache;
mod core;

use arity::{Compose, FusedStage};
use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;
use crate::application::execution::kernel::radix_stage::normalize_inplace;
use crate::application::execution::policy::ExecutionPolicy;
pub use cache::CompositeCache;

pub(crate) type Fused2<A, B> = Compose<A, B>;
pub(crate) type Fused3<A, B, C> = Compose<A, Fused2<B, C>>;
pub(crate) type Fused4<A, B, C, D> = Compose<A, Fused3<B, C, D>>;
pub(crate) type Fused5<A, B, C, D, E> = Compose<A, Fused4<B, C, D, E>>;
pub(crate) type Fused6<A, B, C, D, E, FS> = Compose<A, Fused5<B, C, D, E, FS>>;

#[inline]
pub(crate) fn stockham_stage_fused<F: CompositeCache + ShortWinogradScalar, P: ExecutionPolicy, Node: FusedStage>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    prev_len: usize,
    twiddles: &[&[Complex<F>]],
    inverse: bool,
) {
    let final_stage_len = prev_len * Node::R_TOTAL;
    let groups_out = dst.len() / final_stage_len;

    P::for_each_chunk_mut_enumerated(dst, final_stage_len, |b_out, dst_block| {
        Node::compute_group::<F>(src, dst_block, prev_len, b_out, groups_out, twiddles, 0, inverse);
    });
}

#[inline]
pub fn forward_inplace_with_radices<F: CompositeCache + ShortWinogradScalar>(data: &mut [Complex<F>], radices: &[usize]) {
    core::composite_core_with_radices(data, false, radices);
}

#[inline]
pub fn inverse_inplace_unnorm_with_radices<F: CompositeCache + ShortWinogradScalar>(
    data: &mut [Complex<F>],
    radices: &[usize],
) {
    core::composite_core_with_radices(data, true, radices);
}

#[inline]
pub fn inverse_inplace_with_radices<F: CompositeCache + ShortWinogradScalar>(data: &mut [Complex<F>], radices: &[usize]) {
    core::composite_core_with_radices(data, true, radices);
    normalize_inplace(data, F::cast_f64(1.0 / data.len() as f64));
}

#[cfg(test)]
#[path = "../tests_radix_composite.rs"]
mod tests;
