//! The time-decimated stage set: rows arrive bit-reversed, read straight
//! from the caller's interleaved rows on the first pass, and leave in
//! natural order.

use super::lane::Lane;
use super::pass::butterfly_rows;
use super::plan::BatchedPlan;
use super::radix::{Dit2, Dit4, Pair};
use super::seams::{Columns, Rows, Seams};
use super::sweep::{block_columns, spread, sweep_lengths};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

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
        for (index, stages) in sweep_lengths(len.trailing_zeros()).enumerate() {
            sect!(TIME_SWEEPS[index], {
                sweep_time(re, im, tw, source, b, s, len, l0, stages, simd);
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

/// One time-decimated sweep over stages `l0 << p` for `p < stages`.
///
/// `source` is read by the pass over stage 2 when that stage is in this
/// sweep, which is the one pass that loads every element exactly once.
#[expect(
    clippy::inline_always,
    reason = "must fold into the stage set's target-feature scope with the driver it calls"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the sweep is the loop nest of one stage set; its arguments are the set's fields plus the sweep's stage range"
)]
#[inline(always)]
pub(super) fn sweep_time<T, A>(
    re: &mut [T],
    im: &mut [T],
    tw: &[Pair<T>],
    source: Option<&[T]>,
    batch: usize,
    stride: usize,
    len: usize,
    l0: usize,
    stages: u32,
    simd: Simd<T, A>,
) where
    T: LaneScalar + Lane,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let step0 = l0 >> 1;
    let tile_rows = 1usize << stages;
    let span = tile_rows * step0;
    let groups = len / span;
    let cols = block_columns::<T>(tile_rows, lanes, batch);
    let splat = |(wr, wi): Pair<T>| (simd.splat(wr), simd.splat(wi));
    // The source rides the sweep over stage 2; its rows are read in
    // bit-reversed order straight from the caller's buffer.
    let source = source
        .filter(|_| l0 == 2)
        .map(|source| (source, len.trailing_zeros()));

    for j in 0..step0 {
        for g in 0..groups {
            let base = g * span + j;
            let mut start = 0;
            while start < batch {
                let block = Columns {
                    start,
                    end: (start + cols).min(batch),
                };
                let mut o = 0;
                while o + 2 <= stages {
                    let l = l0 << o;
                    let half = l >> 1;
                    let twx = half - 1;
                    for q in 0..tile_rows >> 2 {
                        let i0 = spread(q, o, 2);
                        let e = j + (i0 & ((1 << o) - 1)) * step0;
                        let tws = [tw[twx + e], tw[twx + half + e], tw[twx + half + e + half]];
                        let twv = tws.map(splat);
                        let rows = Rows {
                            first: base + i0 * step0,
                            step: step0 << o,
                        };
                        butterfly_rows::<T, A, Dit4, 4, 3>(
                            re,
                            im,
                            rows,
                            stride,
                            batch,
                            block,
                            &tws,
                            &twv,
                            simd,
                            Seams::time(source.filter(|_| l == 2)),
                        );
                    }
                    o += 2;
                }
                if o < stages {
                    let l = l0 << o;
                    let half = l >> 1;
                    let twx = half - 1;
                    for q in 0..tile_rows >> 1 {
                        let i0 = spread(q, o, 1);
                        let e = j + (i0 & ((1 << o) - 1)) * step0;
                        let tws = [tw[twx + e]];
                        let twv = tws.map(splat);
                        let rows = Rows {
                            first: base + i0 * step0,
                            step: step0 << o,
                        };
                        butterfly_rows::<T, A, Dit2, 2, 1>(
                            re,
                            im,
                            rows,
                            stride,
                            batch,
                            block,
                            &tws,
                            &twv,
                            simd,
                            Seams::time(source.filter(|_| l == 2)),
                        );
                    }
                }
                start = block.end;
            }
        }
    }
}
