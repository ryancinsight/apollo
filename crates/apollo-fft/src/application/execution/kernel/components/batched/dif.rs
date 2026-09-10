//! Decimation-in-frequency stage set for the planar four-step's second axis.
//!
//! ## Why a second stage set exists
//!
//! The deinterleave earns bit-reversed rows for free by writing each row to
//! `rev(row)`, which is exactly the order a decimation-in-time stage set wants.
//! The transpose between the two axes then destroys that order, so the route
//! used to run a whole pass restoring it — 4.9 to 6.2% of the route, on the
//! order of the transpose itself, and its entire content was undoing the pass
//! before it (`gap_audit.md#planar-pass-attribution`).
//!
//! Decimation in frequency inverts both ends of that. It consumes natural
//! order, which is what the transpose leaves, and produces bit-reversed order,
//! which the sink absorbs for free by reading `rev(row)` — the same trick the
//! deinterleave already uses on the source side. The repair pass has nowhere
//! left to be.
//!
//! ## Why it is a sibling rather than a parameter
//!
//! The two are different algorithms, not two configurations of one. DIT
//! multiplies the odd operand *before* the butterfly and DIF multiplies the
//! difference *after* it, so nearly every arithmetic line differs; a const
//! parameter would monomorphize to these same two bodies while making both
//! harder to read. What they genuinely share — the twiddle table, the load and
//! store helpers, the fold contract — is shared.
//!
//! The table is shared exactly: stage sub-length `l` occupies `l / 2` entries
//! at offset `l / 2 - 1`, holding `W_l^j`. DIT walks those stages upward and
//! DIF downward, over the same values.

use core::mem::size_of;

use super::fold::FourStepFold;
use super::lane::Lane;
use super::pass::butterfly_rows;
use super::plan::BatchedPlan;
use super::radix::{Dif2, Dif4, Pair};
use super::seams::{Columns, Rows, Seams, SinkRows};
use super::sink::{stage_out, DIRECT_SINK_MAX_PLANE_BYTES, STAGED_SINK_MAX_LEN};
use super::sweep::{block_columns, spread, sweep_lengths_descending};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

/// Section labels of the frequency-decimated stage set's sweeps, by sweep
/// index; reported beneath `stages2`.
pub(super) const FREQUENCY_SWEEPS: [&str; 3] = ["f1", "f2", "f3"];

/// All stages of `batch` independent length-`len` transforms over planar data,
/// decimated in frequency.
///
/// Input rows in natural order; output rows bit-reversed. Dispatch happens once
/// for the whole stage set, as [`super::dit::BatchedStages`] documents.
pub(super) struct BatchedStagesDif<'a, T> {
    pub(super) re: &'a mut [T],
    pub(super) im: &'a mut [T],
    pub(super) tw: &'a [(T, T)],
    /// The four-step twiddle tables multiplied into the first stage's loads,
    /// or `None`; rows in the same natural order the data rows carry, the
    /// mirror of the bit-reversed planes the decimation-in-time set required.
    pub(super) fold: Option<&'a FourStepFold<T>>,
    /// Interleaved output written by the last pass in place of the planes,
    /// or `None` to leave the result in the planes. Rows of `batch`
    /// complexes as `2 * batch` reals; plane row `p` lands in output row
    /// `rev(p)`, the permutation the reinterleave pass used to absorb. Writing
    /// here deletes that pass: the last stage pair stores every element
    /// exactly once, and one register interleave plus two interleaved stores
    /// replace the two plane stores.
    pub(super) sink: Option<&'a mut [T]>,
    /// One tile block of interleaved rows for the sink, as reals.
    pub(super) staging: &'a mut [T],
    pub(super) batch: usize,
    pub(super) stride: usize,
    pub(super) len: usize,
}

impl<T> LaneKernel<T> for BatchedStagesDif<'_, T>
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
            fold,
            mut sink,
            staging,
            batch: b,
            stride: s,
            len,
        } = self;
        // Widest stage first, in sweeps of up to `SWEEP_STAGES` stages per
        // trip through the planes, two per pass while two remain and then
        // one: the mirror of the time-decimated set's grouping over the
        // same L1-resident tiles (see [`super::sweep`]), with the odd
        // remainder taken first so the sink rides a full tile. The four-step
        // twiddle rides the pass over stage `len` and the interleaved sink
        // the pass over stage 2. Stage `l` holds `W_l^j` at `l / 2 - 1`.
        let mut l_top = len;
        for (index, stages) in sweep_lengths_descending(len.trailing_zeros()).enumerate() {
            sect!(FREQUENCY_SWEEPS[index], {
                sweep_frequency(
                    re,
                    im,
                    tw,
                    fold,
                    sink.as_deref_mut(),
                    staging,
                    b,
                    s,
                    len,
                    l_top,
                    stages,
                    simd,
                );
            });
            l_top >>= stages;
        }
    }
}

/// Runs the same stage set decimated in frequency: rows arrive in natural
/// order and leave bit-reversed, which is the pairing that lets the sink
/// absorb the permutation the time-decimated set needed a whole pass to
/// repair (see [`BatchedStagesDif`]).
pub(super) fn run_batched_dif<T>(
    re: &mut [T],
    im: &mut [T],
    plan: &BatchedPlan<T>,
    fold: Option<&FourStepFold<T>>,
    sink: Option<&mut [T]>,
    staging: &mut [T],
    batch: usize,
    stride: usize,
) where
    T: LaneScalar + MixedRadixScalar + Lane,
{
    hermes_simd::vectorize(BatchedStagesDif {
        re,
        im,
        tw: &plan.tw,
        fold,
        sink,
        staging,
        batch,
        stride,
        len: plan.len,
    });
}

/// One frequency-decimated sweep over stages `l_top >> p` for `p < stages`.
///
/// `fold` rides the pass over stage `len` and `sink` the pass over stage 2,
/// each when that stage is in this sweep; the sink rows stage through
/// `staging` one block at a time.
#[expect(
    clippy::inline_always,
    reason = "must fold into the stage set's target-feature scope with the driver it calls"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the sweep is the loop nest of one stage set; its arguments are the set's fields plus the sweep's stage range"
)]
#[inline(always)]
pub(super) fn sweep_frequency<T, A>(
    re: &mut [T],
    im: &mut [T],
    tw: &[Pair<T>],
    fold: Option<&FourStepFold<T>>,
    sink: Option<&mut [T]>,
    staging: &mut [T],
    batch: usize,
    stride: usize,
    len: usize,
    l_top: usize,
    stages: u32,
    simd: Simd<T, A>,
) where
    T: LaneScalar + Lane,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let tile_rows = 1usize << stages;
    let l_bottom = l_top >> (stages - 1);
    let step_bottom = l_bottom >> 1;
    let span = l_top;
    let groups = len / span;
    let cols = block_columns::<T>(tile_rows, lanes, batch);
    let pitch = 2 * cols;
    let splat = |(wr, wi): Pair<T>| (simd.splat(wr), simd.splat(wi));
    // The sink rides the sweep over stage 2, whose tiles are consecutive
    // rows, as the source does; it is staged up to `STAGED_SINK_MAX_LEN`.
    let mut sink = sink.filter(|_| l_bottom == 2);
    // The sink writes direct only where that is safe and measured cheaper:
    // the caller's rows on a cache line, so no register store straddles
    // two lines (a memcpy handles a misaligned row better than split
    // stores), and the plane within `DIRECT_SINK_MAX_PLANE_BYTES`, where
    // the staged form's extra pass over the tile is the larger cost.
    // Otherwise the rows stage through one copy each up to
    // `STAGED_SINK_MAX_LEN`.
    let aligned = sink
        .as_deref()
        .is_some_and(|rows| (rows.as_ptr() as usize) % 64 == 0);
    let staged = len * batch <= STAGED_SINK_MAX_LEN
        && !(aligned && len * batch * size_of::<T>() <= DIRECT_SINK_MAX_PLANE_BYTES);
    let row_bits = len.trailing_zeros();
    if staged && sink.is_some() {
        assert!(
            staging.len() >= tile_rows * pitch,
            "invariant: the staging buffer holds one tile block"
        );
    }

    for j in 0..step_bottom {
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
                    let l = l_top >> o;
                    let quarter = l >> 2;
                    let wide = (l >> 1) - 1;
                    let narrow = quarter - 1;
                    let at = stages - o - 2;
                    let pass_fold = if l == len { fold } else { None };
                    // Stage 2 is in this pass only when it is the sweep's
                    // last, so the sink rides it exactly once.
                    let last = l == 4;
                    for q in 0..tile_rows >> 2 {
                        let i0 = spread(q, at, 2);
                        let e = j + (i0 & ((1 << at) - 1)) * step_bottom;
                        let tws = [tw[wide + e], tw[wide + e + quarter], tw[narrow + e]];
                        let twv = tws.map(splat);
                        let rows = Rows {
                            first: base + i0 * step_bottom,
                            step: step_bottom << at,
                        };
                        let pass_sink = match (last, sink.as_deref_mut()) {
                            (true, Some(_)) if staged => Some((
                                &mut *staging,
                                SinkRows::Staged {
                                    first: i0,
                                    pitch,
                                    first_column: block.start,
                                },
                            )),
                            (true, Some(sink)) => Some((sink, SinkRows::Direct { row_bits })),
                            _ => None,
                        };
                        butterfly_rows::<T, A, Dif4, 4, 3>(
                            re,
                            im,
                            rows,
                            stride,
                            batch,
                            block,
                            &tws,
                            &twv,
                            simd,
                            Seams::frequency(pass_fold, pass_sink),
                        );
                    }
                    o += 2;
                }
                if o < stages {
                    let l = l_top >> o;
                    let half = l >> 1;
                    let twx = half - 1;
                    let pass_fold = if l == len { fold } else { None };
                    let last = l == 2;
                    for q in 0..tile_rows >> 1 {
                        let i0 = spread(q, 0, 1);
                        let tws = [tw[twx + j]];
                        let twv = tws.map(splat);
                        let rows = Rows {
                            first: base + i0 * step_bottom,
                            step: step_bottom,
                        };
                        let pass_sink = match (last, sink.as_deref_mut()) {
                            (true, Some(_)) if staged => Some((
                                &mut *staging,
                                SinkRows::Staged {
                                    first: i0,
                                    pitch,
                                    first_column: block.start,
                                },
                            )),
                            (true, Some(sink)) => Some((sink, SinkRows::Direct { row_bits })),
                            _ => None,
                        };
                        butterfly_rows::<T, A, Dif2, 2, 1>(
                            re,
                            im,
                            rows,
                            stride,
                            batch,
                            block,
                            &tws,
                            &twv,
                            simd,
                            Seams::frequency(pass_fold, pass_sink),
                        );
                    }
                }
                if let Some(sink) = sink.as_deref_mut().filter(|_| staged) {
                    stage_out(
                        sink, staging, tile_rows, base, batch, row_bits, block, pitch,
                    );
                }
                start = block.end;
            }
        }
    }
}
