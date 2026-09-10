//! Where a pass reads its rows and where it writes them: the row set, the
//! column block, and the seams a pass carries in its monomorphization.

use super::fold::FourStepFold;

/// The rows of one row set: `first + i * step` for `i < N`.
///
/// Every pass in both stage sets addresses equally spaced rows, and stating
/// the spacing rather than listing the rows lets the pass keep one base and
/// one step per plane instead of a table of offsets that would otherwise
/// occupy, and then overflow, the general registers.
#[derive(Clone, Copy)]
pub(super) struct Rows {
    /// Plane row of the first element of the set.
    pub(super) first: usize,
    /// Plane rows between consecutive elements of the set.
    pub(super) step: usize,
}

/// The columns of one row set a pass covers: batch indices `start..end`.
///
/// A sweep ([`super::sweep`]) runs its passes over one column block of a
/// tile at a time so the block stays in L1 between them; the block is the
/// whole batch only when the tile fits.
#[derive(Clone, Copy)]
pub(super) struct Columns {
    pub(super) start: usize,
    pub(super) end: usize,
}

/// Where the sink pass writes its rows.
///
/// Staged: one tile block in a contiguous buffer (see [`super::sweep`]),
/// tile row `r` starting `r * pitch` reals in, batch column `k` sitting
/// `2 * (k - first_column)` reals further; the row set's first row is tile
/// row `first` and its rows are [`Rows::step`] tile rows apart, the same
/// step as in the planes. Direct: the caller's rows, `row_bits` wide, plane
/// row `p` writing row `rev(p)`.
#[derive(Clone, Copy)]
pub(super) enum SinkRows {
    Staged {
        first: usize,
        pitch: usize,
        first_column: usize,
    },
    Direct {
        row_bits: u32,
    },
}

impl SinkRows {
    /// The sink of a pass without one: never addressed.
    pub(super) const NONE: Self = Self::Direct { row_bits: 0 };
}

/// Where a pass reads its staged planar rows: row `r` starts `r * pitch`
/// reals in and column `k` sits `k - first_column` further. The block the
/// second set's first sweep transposes out of the first set's planes
/// (see [`super::dif::BatchedStagesDif::transposed`]).
#[derive(Clone, Copy)]
pub(super) struct StagedRows {
    pub(super) pitch: usize,
    pub(super) first_column: usize,
}

impl StagedRows {
    /// The staged rows of a pass without them: never addressed.
    pub(super) const NONE: Self = Self {
        pitch: 0,
        first_column: 0,
    };
}

/// Where a pass reads its rows and where it writes them.
///
/// Exactly the combinations the two stage sets produce: the time-decimated
/// set reads the caller's interleaved rows on its first pass and the planes
/// otherwise; the frequency-decimated set folds the four-step twiddle on its
/// first pass (from a staged transpose of the first set's planes where the
/// driver routes it so), writes the staged sink block on its last, and both
/// when the two coincide. Each variant selects a monomorphized pass whose row loop
/// carries no seam it does not use, which is what keeps the pass's
/// addressing in registers.
pub(super) enum Seams<'a, 'b, T> {
    /// Planes in, planes out.
    Planes,
    /// Interleaved rows in, `rows` bits wide; plane row `p` reads row `rev(p)`.
    Source(&'a [T], u32),
    /// Planes in with the four-step twiddle tables multiplied into the loads.
    Fold(&'a FourStepFold<T>),
    /// Planes in, interleaved rows out, staged or direct.
    Sink(&'b mut [T], SinkRows),
    /// Folded loads and sink stores in one pass.
    FoldSink(&'a FourStepFold<T>, &'b mut [T], SinkRows),
    /// Staged planar rows in, with the four-step twiddle tables multiplied
    /// into the loads; planes out. The first pass over a transposed block.
    FoldStaged(&'a FourStepFold<T>, &'a [T], &'a [T], StagedRows),
}

impl<'a, 'b, T> Seams<'a, 'b, T> {
    /// The frequency-decimated set's seams for one pass.
    pub(super) fn frequency(
        fold: Option<&'a FourStepFold<T>>,
        sink: Option<(&'b mut [T], SinkRows)>,
    ) -> Self {
        match (fold, sink) {
            (Some(fold), Some((sink, staging))) => Self::FoldSink(fold, sink, staging),
            (Some(fold), None) => Self::Fold(fold),
            (None, Some((sink, staging))) => Self::Sink(sink, staging),
            (None, None) => Self::Planes,
        }
    }

    /// The time-decimated set's seams for one pass.
    pub(super) fn time(source: Option<(&'a [T], u32)>) -> Self {
        source.map_or(Self::Planes, |(source, row_bits)| {
            Self::Source(source, row_bits)
        })
    }
}
