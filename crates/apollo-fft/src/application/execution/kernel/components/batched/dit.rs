//! The time-decimated stage set: rows arrive bit-reversed, read straight
//! from the caller's interleaved rows on the first pass, and leave in
//! natural order.

use super::lane::Lane;
use super::plan::BatchedPlan;
use super::sweep;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel};

/// Section labels of the time-decimated stage set's sweeps, by sweep index;
/// the attribution probe reports them beneath `stages1`.
pub(super) const TIME_SWEEPS: [&str; 3] = ["t1", "t2", "t3"];

/// All stages of `batch` independent length-`len` transforms over planar data.
///
/// Dispatch happens once for the whole stage set, which is the placement Hermes
/// ADR 016 requires: outside the innermost loop, and never wrapping a
/// thread-spawning call.
pub(super) struct BatchedStages<'a, T> {
    re: &'a mut [T],
    im: &'a mut [T],
    tw: &'a [(T, T)],
    /// Interleaved input read by the first pass in place of the planes, or
    /// `None` when the planes already hold it. Rows of `batch` complexes as
    /// `2 * batch` reals in natural order; plane row `p` reads source row
    /// `rev(p)`, which is the row map the deinterleave pass used to apply.
    /// Reading here deletes that pass: the first stage pair is the one that
    /// loads every element exactly once, and two interleaved vector loads
    /// plus one register deinterleave replace the two plane loads.
    source: Option<&'a [T]>,
    /// Live columns per row — the loop bound.
    batch: usize,
    /// Elements per row including [`super::plane::ROW_PAD`] — the index multiplier.
    stride: usize,
    len: usize,
}

impl<T> LaneKernel<T> for BatchedStages<'_, T>
where
    T: LaneScalar + MixedRadixScalar + Lane,
{
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "large LaneKernel::call body must fold into the dispatcher's target-feature scope"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let Self {
            re,
            im,
            tw,
            source,
            batch: b,
            stride: s,
            len,
        } = self;
        // Stages ascend from 2; each sweep applies up to `SWEEP_STAGES` of
        // them per trip through the planes, two per pass while two remain
        // and then one, over tiles that stay in L1 between its passes
        // (see [`super::sweep`]). The four-step twiddle and the interleaved source
        // both ride the pass over stage 2, the one pass that loads every
        // element exactly once. The per-element operation order is the
        // single-stage one, so results are bitwise those of any other
        // grouping.
        let mut l0 = 2usize;
        for (index, stages) in sweep::sweep_lengths(len.trailing_zeros()).enumerate() {
            sect!(TIME_SWEEPS[index], {
                sweep::sweep_time(re, im, tw, source, b, s, len, l0, stages, simd);
            });
            l0 <<= stages;
        }
    }
}

/// Runs the stage set of `batch` transforms of length `plan.len` over planar
/// `re`/`im`, whose rows the caller has already bit-reversed — which the
/// driver's deinterleave does for free by writing each row to its reversed
/// position.
pub(super) fn run_batched<T>(
    re: &mut [T],
    im: &mut [T],
    plan: &BatchedPlan<T>,
    source: Option<&[T]>,
    batch: usize,
    stride: usize,
) where
    T: LaneScalar + MixedRadixScalar + Lane,
{
    hermes_simd::vectorize(BatchedStages {
        re,
        im,
        tw: &plan.tw,
        source,
        batch,
        stride,
        len: plan.len,
    });
}
