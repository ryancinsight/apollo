//! The sink's policy and its staged row copy: where the frequency-decimated
//! set's last pass writes the caller's rows, and how a staged block reaches
//! them.

use super::register::reverse_row;
use super::seams::Columns;

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

/// Largest plane, in bytes, whose sink is written direct when the caller's
/// rows sit on a cache line.
///
/// Measured, not modelled: with the caller on a line the direct sink read
/// its sweep 33% below the staged one at 8 KiB planes (2048 `f32`: 1.6k to
/// 1.7k cycles against 2.4k to 2.7k, the transform 12% shorter) and level
/// at 16 KiB planes (2048 `f64`, 4096 `f32`), so the bound stops at two
/// pages (`output/apollo-planar-rectangular/sinkdir_*`,
/// backlog.md#apollo-planar-sink-aligned-direct).
pub(super) const DIRECT_SINK_MAX_PLANE_BYTES: usize = 2 * 4096;

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
pub(super) fn stage_out<T: Copy>(
    sink: &mut [T],
    staging: &[T],
    rows: usize,
    base: usize,
    batch: usize,
    row_bits: u32,
    block: Columns,
    pitch: usize,
) {
    let width = 2 * (block.end - block.start);
    for r in 0..rows {
        let at = reverse_row(base + r, row_bits) * batch * 2 + 2 * block.start;
        sink[at..at + width].copy_from_slice(&staging[r * pitch..r * pitch + width]);
    }
}
