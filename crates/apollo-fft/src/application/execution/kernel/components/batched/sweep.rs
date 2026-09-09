//! Stage sweeps: several passes over one L1-resident tile per trip through
//! the planes.
//!
//! A pass streams every element of the planes from L2 and back once, and
//! the row-set driver runs at the L1 to L2 line-transfer rate whatever it
//! computes: about four line fills and four writebacks per radix-4 quad
//! (ADR 0056). The planes therefore cost by the number of passes, and a
//! radix-8 pass cannot cut that number on sixteen registers (ADR 0055).
//!
//! A sweep cuts it another way. The `2^S` rows a run of `S` consecutive
//! stages mixes form a tile that no other row touches during those stages,
//! so the sweep runs all of its passes over one tile before moving to the
//! next, and the tile's columns are blocked so each block stays in L1
//! between the passes. Every element then crosses the L1 boundary once per
//! sweep rather than once per pass, while each pass keeps the radix-4 (or
//! radix-2) butterfly, its register budget, its twiddle order and its
//! per-element operation order; results are bitwise those of the unswept
//! stage sets.
//!
//! Column blocking is what the rejected arm of ADR 0054 did not have in
//! this form: it blocked one cache line of columns across *every* stage, so
//! each row set touched eight scattered lines per block and the row loop
//! ran two iterations between row-set setups. A tile block here is
//! [`TILE_BYTES`] wide, the row loop runs sixteen or more vectors per row
//! set, and consecutive blocks continue the same row streams, which is what
//! the prefetchers follow.
//!
//! ## Sink staging
//!
//! The two seams read and write the caller's interleaved buffer in
//! bit-reversed row order, and the sixteen rows of one tile are then `n`
//! bytes apart, a multiple of 4 KiB from 4096 up: their lines fill the same
//! L1 sets and evict the tile between the sweep's passes whenever those
//! sets hold it, which made the sink sweep's cost a function of where the
//! caller's buffer sits against the planes (ADR 0057, 30k to 82k cycles at
//! 16384 `f64`). The sink therefore writes one tile block at a time into a
//! contiguous staging buffer of [`STAGING_LEN`] complexes at the end of the
//! scratch, and the block's rows copy out afterwards, one sequential row
//! move each, once the tile is dead. The source keeps its direct loads: its
//! four rows abreast are four streams in flight, and every staged form of
//! it measured slower (ADR 0058).
//!
//! ## Tile geometry
//!
//! For the time-decimated set the sweep over stages `l0, 2 l0, ..., l0
//! 2^(S-1)` has tiles of rows `base + j + i * step0` for `i < 2^S`, with
//! `step0 = l0 / 2`, `j < step0` and `base` a multiple of the top stage
//! length `l0 2^(S-1)`. Its pass over stages `l0 2^o` and `l0 2^(o+1)`
//! takes the rows whose tile index has bits `o` and `o + 1` varying, so its
//! row sets are `i0 + t 2^o` for `t < 4` with `i0` having those bits clear,
//! its row step is `step0 2^o`, and its twiddle exponent is `j` plus
//! `step0` times the bits of `i0` below `o`. The frequency-decimated set is
//! the mirror: stages descend from `l_top`, tiles are spaced by the bottom
//! stage's half length, and each pass takes the high bits first. The seams
//! ride the sweeps whose tiles are consecutive rows (`step0 = 1`), so a
//! row's tile-local index is its distance from the tile base.

use core::mem::size_of;

use hermes_simd::{LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

use super::radix::{
    butterfly_rows, Columns, Dif2, Dif4, Dit2, Dit4, Lane, Pair, Rows, Seams, SinkRows,
};
use super::{reverse_row, FourStepFold};

/// Stages a sweep fuses, so the tile is `2^SWEEP_STAGES` rows.
///
/// Four is two radix-4 passes per trip through the planes; six would be
/// three at a quarter of the column block. Measured on the pinned
/// performance core (ADR 0056) and re-measured before it moves.
pub(super) const SWEEP_STAGES: u32 = 4;

/// Bytes of both planes a tile's column block occupies, sized so the block
/// and the streams around it share a 48 KiB L1 without conflict: 16 KiB is
/// a third of it.
const TILE_BYTES: usize = 16 * 1024;

/// Complex elements the seam staging buffer holds: one tile block of
/// interleaved samples, which is [`TILE_BYTES`] whatever the scalar, so
/// `TILE_BYTES` over the narrowest complex this route dispatches (8 bytes);
/// wider scalars use a prefix.
pub(crate) const STAGING_LEN: usize = TILE_BYTES / 8;

/// Longest transform whose sink is staged.
///
/// Staging wins where the planes sit in L2, since the tile it protects is
/// what the sink's rows would evict; where the planes overflow L2 the sink
/// is bound by the writes themselves, and one row copy after another
/// loses the four-row overlap of the direct stores (262144 `f64`: the
/// sink sweep 1067k against 800k cycles, the census 5% slower; ADR 0058).
/// The bound is one host's 3 MiB L2 against 16 bytes per element of planes
/// plus the caller's buffer; re-measure before moving it.
pub(super) const STAGED_SINK_MAX_LEN: usize = 1 << 16;

/// Columns per tile block: [`TILE_BYTES`] over the tile's rows and both
/// planes, rounded down to whole vectors and up to at least one, and never
/// wider than the batch.
fn block_columns<T>(tile_rows: usize, lanes: usize, batch: usize) -> usize {
    let cols = TILE_BYTES / (tile_rows * 2 * size_of::<T>());
    (cols / lanes * lanes).max(lanes).min(batch)
}

/// The stage counts of consecutive sweeps over `stages` stages for the
/// time-decimated set: full sweeps first, then whatever remains, so the
/// source rides a full tile.
pub(super) fn sweep_lengths(stages: u32) -> impl Iterator<Item = u32> {
    let full = stages / SWEEP_STAGES;
    let rest = stages % SWEEP_STAGES;
    core::iter::repeat_n(SWEEP_STAGES, full as usize).chain((rest > 0).then_some(rest))
}

/// The stage counts for the frequency-decimated set: whatever remains first,
/// then full sweeps, so the sink rides a full tile. A two-row tile at the
/// end of nine stages has nothing for the staging to spread, and its copy
/// costs a third of the sweep (ADR 0058).
pub(super) fn sweep_lengths_descending(stages: u32) -> impl Iterator<Item = u32> {
    let full = stages / SWEEP_STAGES;
    let rest = stages % SWEEP_STAGES;
    (rest > 0)
        .then_some(rest)
        .into_iter()
        .chain(core::iter::repeat_n(SWEEP_STAGES, full as usize))
}

/// Inserts `width` zero bits into `q` at bit `at`: the tile index of a row
/// set's first row, given the set's index among the sets of one pass.
fn spread(q: usize, at: u32, width: u32) -> usize {
    let low = q & ((1usize << at) - 1);
    let high = q >> at;
    low | (high << (at + width))
}

/// Copies one block of staged rows out to the sink, one row at a time.
///
/// Tile row `r` is caller row `rev(base + r)`, whose `batch` complexes start
/// `row * batch * 2` reals in; the block spans batch columns `block`, and
/// staged row `r` starts `r * pitch` reals in. Each row is one sequential
/// move: the caller's rows share a page offset, so rows moved abreast alias
/// one another's stores through the page-offset check and measured far
/// slower than one row after another (ADR 0058).
#[expect(
    clippy::inline_always,
    reason = "must fold into the stage set's target-feature scope with the sweep that calls it"
)]
#[inline(always)]
fn stage_out<T: Copy>(
    sink: &mut [T],
    staging: &[T],
    rows: usize,
    base: usize,
    batch: usize,
    block: Columns,
    pitch: usize,
) {
    let width = 2 * (block.end - block.start);
    let row_bits = batch.trailing_zeros();
    for r in 0..rows {
        let at = reverse_row(base + r, row_bits) * batch * 2 + 2 * block.start;
        sink[at..at + width].copy_from_slice(&staging[r * pitch..r * pitch + width]);
    }
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
    let staged = len * batch <= STAGED_SINK_MAX_LEN;
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
                    stage_out(sink, staging, tile_rows, base, batch, block, pitch);
                }
                start = block.end;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{block_columns, spread, sweep_lengths, STAGING_LEN};

    #[test]
    fn sweeps_take_full_lengths_first_then_the_remainder() {
        assert_eq!(sweep_lengths(1).collect::<Vec<_>>(), [1]);
        assert_eq!(sweep_lengths(4).collect::<Vec<_>>(), [4]);
        assert_eq!(sweep_lengths(7).collect::<Vec<_>>(), [4, 3]);
        assert_eq!(sweep_lengths(8).collect::<Vec<_>>(), [4, 4]);
        assert_eq!(sweep_lengths(9).collect::<Vec<_>>(), [4, 4, 1]);
    }

    #[test]
    fn descending_sweeps_take_the_remainder_first() {
        use super::sweep_lengths_descending as descending;
        assert_eq!(descending(1).collect::<Vec<_>>(), [1]);
        assert_eq!(descending(4).collect::<Vec<_>>(), [4]);
        assert_eq!(descending(7).collect::<Vec<_>>(), [3, 4]);
        assert_eq!(descending(8).collect::<Vec<_>>(), [4, 4]);
        assert_eq!(descending(9).collect::<Vec<_>>(), [1, 4, 4]);
    }

    #[test]
    fn spread_inserts_zero_bits_at_the_pass_position() {
        assert_eq!(spread(0b101, 0, 2), 0b10100);
        assert_eq!(spread(0b110, 1, 2), 0b11000);
        assert_eq!(spread(0b101, 2, 2), 0b10001);
        assert_eq!(spread(0b11, 1, 1), 0b101);
    }

    #[test]
    fn block_columns_fill_the_tile_budget_in_whole_vectors() {
        assert_eq!(block_columns::<f64>(16, 4, 256), 64);
        assert_eq!(block_columns::<f32>(16, 8, 256), 128);
        assert_eq!(block_columns::<f64>(64, 4, 256), 16);
        assert_eq!(block_columns::<f64>(16, 4, 32), 32);
        assert_eq!(block_columns::<f32>(2, 8, 2), 2);
    }

    #[test]
    fn a_tile_block_of_either_scalar_fits_the_staging_buffer() {
        for stages in 1..=4u32 {
            let rows = 1usize << stages;
            assert!(rows * block_columns::<f64>(rows, 4, 1 << 20) <= STAGING_LEN);
            assert!(rows * block_columns::<f32>(rows, 8, 1 << 20) <= STAGING_LEN);
        }
    }
}
