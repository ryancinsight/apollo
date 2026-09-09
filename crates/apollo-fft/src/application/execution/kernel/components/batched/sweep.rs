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
//! stage's half length, and each pass takes the high bits first.

use core::mem::size_of;

use hermes_simd::{LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

use super::radix::{butterfly_rows, Columns, Dif2, Dif4, Dit2, Dit4, Lane, Pair, Rows, Seams};

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

/// Columns per tile block: [`TILE_BYTES`] over the tile's rows and both
/// planes, rounded down to whole vectors and up to at least one, and never
/// wider than the batch.
fn block_columns<T>(tile_rows: usize, lanes: usize, batch: usize) -> usize {
    let cols = TILE_BYTES / (tile_rows * 2 * size_of::<T>());
    (cols / lanes * lanes).max(lanes).min(batch)
}

/// The stage counts of consecutive sweeps over `stages` stages: full sweeps
/// first, then whatever remains.
pub(super) fn sweep_lengths(stages: u32) -> impl Iterator<Item = u32> {
    let full = stages / SWEEP_STAGES;
    let rest = stages % SWEEP_STAGES;
    core::iter::repeat_n(SWEEP_STAGES, full as usize).chain((rest > 0).then_some(rest))
}

/// Inserts `width` zero bits into `q` at bit `at`: the tile index of a row
/// set's first row, given the set's index among the sets of one pass.
fn spread(q: usize, at: u32, width: u32) -> usize {
    let low = q & ((1usize << at) - 1);
    let high = q >> at;
    low | (high << (at + width))
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
    let row_bits = len.trailing_zeros();
    let step0 = l0 >> 1;
    let tile_rows = 1usize << stages;
    let span = tile_rows * step0;
    let groups = len / span;
    let cols = block_columns::<T>(tile_rows, lanes, batch);
    let splat = |(wr, wi): Pair<T>| (simd.splat(wr), simd.splat(wi));

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
                    let pass_source = if l == 2 { source } else { None };
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
                            row_bits,
                            &tws,
                            &twv,
                            Seams::time(pass_source),
                        );
                    }
                    o += 2;
                }
                if o < stages {
                    let l = l0 << o;
                    let half = l >> 1;
                    let twx = half - 1;
                    let pass_source = if l == 2 { source } else { None };
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
                            row_bits,
                            &tws,
                            &twv,
                            Seams::time(pass_source),
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
/// each when that stage is in this sweep.
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
    fold: Option<(&[T], &[T])>,
    mut sink: Option<&mut [T]>,
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
    let row_bits = len.trailing_zeros();
    let tile_rows = 1usize << stages;
    let step_bottom = (l_top >> (stages - 1)) >> 1;
    let span = l_top;
    let groups = len / span;
    let cols = block_columns::<T>(tile_rows, lanes, batch);
    let splat = |(wr, wi): Pair<T>| (simd.splat(wr), simd.splat(wi));

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
                        butterfly_rows::<T, A, Dif4, 4, 3>(
                            re,
                            im,
                            rows,
                            stride,
                            batch,
                            block,
                            row_bits,
                            &tws,
                            &twv,
                            Seams::frequency(
                                pass_fold,
                                if last { sink.as_deref_mut() } else { None },
                            ),
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
                        butterfly_rows::<T, A, Dif2, 2, 1>(
                            re,
                            im,
                            rows,
                            stride,
                            batch,
                            block,
                            row_bits,
                            &tws,
                            &twv,
                            Seams::frequency(
                                pass_fold,
                                if last { sink.as_deref_mut() } else { None },
                            ),
                        );
                    }
                }
                start = block.end;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{block_columns, spread, sweep_lengths};

    #[test]
    fn sweeps_take_full_lengths_first_then_the_remainder() {
        assert_eq!(sweep_lengths(1).collect::<Vec<_>>(), [1]);
        assert_eq!(sweep_lengths(4).collect::<Vec<_>>(), [4]);
        assert_eq!(sweep_lengths(7).collect::<Vec<_>>(), [4, 3]);
        assert_eq!(sweep_lengths(8).collect::<Vec<_>>(), [4, 4]);
        assert_eq!(sweep_lengths(9).collect::<Vec<_>>(), [4, 4, 1]);
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
}
