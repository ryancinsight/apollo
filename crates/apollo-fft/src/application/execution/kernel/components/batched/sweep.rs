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

/// Columns per tile block: [`TILE_BYTES`] over the tile's rows and both
/// planes, rounded down to whole vectors and up to at least one, and never
/// wider than the batch.
pub(super) fn block_columns<T>(tile_rows: usize, lanes: usize, batch: usize) -> usize {
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
pub(super) fn spread(q: usize, at: u32, width: u32) -> usize {
    let low = q & ((1usize << at) - 1);
    let high = q >> at;
    low | (high << (at + width))
}
