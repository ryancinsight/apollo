//! The dup-split complex multiply both base kernels are built on.
//!
//! A free function rather than a closure in each kernel body, and that is
//! load-bearing rather than tidiness. A closure is inlined at LLVM's
//! discretion; when the surrounding body grows past its budget it declines,
//! which drops the call out of the dispatcher's `#[target_feature]` frame and
//! compiles it at baseline. That cost the instance-major kernel a fivefold
//! column pass, and moving the 128-point route off the sample-major kernel
//! shrank that module enough to flip the same decision the other way for the
//! 64-point route (gap_audit.md#across-instance-outlining).
//!
//! Neither kernel should depend on an inlining heuristic to be correct about
//! its own instruction set.

use hermes_simd::{
    Alignment, ComplexReg, ExecutionMode, LaneScalar, SimdArch, SimdKernel, SimdView,
};

/// Multiplies `v` by the twiddle at chunk `ch` of `tab`: one shuffle, one
/// multiply, one alternating FMA.
#[expect(
    clippy::inline_always,
    reason = "must fold into the dispatcher's target-feature scope; an \
              out-of-line call here compiles at baseline"
)]
#[inline(always)]
pub(crate) fn cmul_chunk<T, A, Align, Mode, Ref>(
    tab: &SimdView<'_, T, A, Align, Mode, Ref>,
    v: ComplexReg<T, A>,
    ch: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    let wr = hermes_simd::Vector::from_view_chunk(tab, ch);
    let wi = hermes_simd::Vector::from_view_chunk(tab, ch + 1);
    let vi = v.into_interleaved();
    ComplexReg::from_interleaved(vi.fmaddsub(wr, vi.swap_adjacent() * wi))
}
