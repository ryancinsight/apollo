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

/// Multiplies `v` by the twiddle at chunk `ch` of `tab`, or by its conjugate
/// when `CONJ`: one shuffle, one multiply, one alternating FMA.
///
/// The dup-split chunk pair holds `re` duplicated, then `im` duplicated, so
/// the product is `fmaddsub(v, re, swap(v) · im)`; the conjugate negates
/// `im`, and with it the rounded product the FMA adds or subtracts, which
/// swaps the alternation: `fmsubadd`. A sign flip rounds nothing, so the
/// result is bitwise the product with a conjugated table.
#[expect(
    clippy::inline_always,
    reason = "must fold into the dispatcher's target-feature scope; an \
              out-of-line call here compiles at baseline"
)]
#[inline(always)]
pub(crate) fn cmul_chunk<T, A, Align, Mode, Ref, const CONJ: bool>(
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
    let cross = vi.swap_adjacent() * wi;
    ComplexReg::from_interleaved(if CONJ {
        vi.fmsubadd(wr, cross)
    } else {
        vi.fmaddsub(wr, cross)
    })
}

/// `v` times the interleaved twiddle `w`, or times `conj(w)` when `CONJ`,
/// bitwise equal to the product with a conjugated `w` in memory.
///
/// The product is `fmaddsub(dup_even(v), w, dup_odd(v) · swap(w))`. With
/// `w` conjugated, the even lane adds the rounded cross term and the odd
/// lane computes `-(re · w.im) + cross`, which is exactly the negation of
/// `re · w.im - cross`: so `fmsubadd` then the odd lanes' sign flip, both
/// exact, give the same bits.
#[expect(
    clippy::inline_always,
    reason = "must fold into the dispatcher's target-feature scope; an \
              out-of-line call here compiles at baseline"
)]
#[inline(always)]
pub(crate) fn cmul<T, A, const CONJ: bool>(
    v: ComplexReg<T, A>,
    w: ComplexReg<T, A>,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    if CONJ {
        let vi = v.into_interleaved();
        let wi = w.into_interleaved();
        ComplexReg::from_interleaved(
            vi.dup_even()
                .fmsubadd(wi, vi.dup_odd() * wi.swap_adjacent()),
        )
        .conj()
    } else {
        v * w
    }
}
