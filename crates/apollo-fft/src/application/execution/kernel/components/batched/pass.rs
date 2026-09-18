//! The row-set driver shared by both stage sets: one pass over one row set,
//! monomorphized per seam combination so its row loop tests nothing and
//! computes only the offsets it uses.

use hermes_simd::{LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

use super::fold::FourStepFold;
use super::lane::Lane;
use super::radix::{cmul, Pair, Radix};
use super::register::{load, load_interleaved, reverse_row, store, store_interleaved};
use super::seams::{Columns, FoldDirection, Rows, Seams, SinkRows};

/// The operands every pass takes: the planes, the row set and column block,
/// and the radix's twiddles in scalar and register form.
struct PassOperands<'p, T, A, const NT: usize>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    re: &'p mut [T],
    im: &'p mut [T],
    rows: Rows,
    stride: usize,
    batch: usize,
    cols: Columns,
    tw: &'p [Pair<T>; NT],
    twv: &'p [Pair<Vector<T, A>>; NT],
    simd: Simd<T, A>,
}

/// The seam operands of one pass, empty where its monomorphization carries
/// no such seam.
struct SeamData<'a, 'b, T> {
    fold: Option<&'a FourStepFold<T>>,
    source: &'a [T],
    source_bits: u32,
    sink: &'b mut [T],
    sink_rows: SinkRows,
}

impl<'a, 'b, T> SeamData<'a, 'b, T> {
    fn planes() -> Self {
        Self {
            fold: None,
            source: &[],
            source_bits: 0,
            sink: &mut [],
            sink_rows: SinkRows::NONE,
        }
    }

    fn source(source: &'a [T], source_bits: u32) -> Self {
        Self {
            fold: None,
            source,
            source_bits,
            sink: &mut [],
            sink_rows: SinkRows::NONE,
        }
    }

    fn fold(fold: &'a FourStepFold<T>) -> Self {
        Self {
            fold: Some(fold),
            source: &[],
            source_bits: 0,
            sink: &mut [],
            sink_rows: SinkRows::NONE,
        }
    }

    fn sink(sink: &'b mut [T], sink_rows: SinkRows) -> Self {
        Self {
            fold: None,
            source: &[],
            source_bits: 0,
            sink,
            sink_rows,
        }
    }

    fn fold_sink(fold: &'a FourStepFold<T>, sink: &'b mut [T], sink_rows: SinkRows) -> Self {
        Self {
            fold: Some(fold),
            source: &[],
            source_bits: 0,
            sink,
            sink_rows,
        }
    }
}

/// Runs radix `R` over one row set across `cols`: the vector loop over
/// `LANE_COUNT` columns at a time and the scalar remainder.
///
/// `tw` holds the scalar twiddles for the remainder and `twv` their lane
/// splats for the vector loop. The caller has bounded every row of `rows` by
/// the plane extent, `cols` by the batch, the source rows by the caller's
/// buffer and the staged rows by the staging buffer, which is what the
/// unchecked loads and stores rely on.
#[expect(
    clippy::inline_always,
    reason = "the driver must fold into the dispatcher's target-feature scope with the kernel that calls it"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the driver is the inner function of both stage sets; its arguments are the pass's operands"
)]
#[inline(always)]
pub(super) fn butterfly_rows<T, A, R, const N: usize, const NT: usize>(
    re: &mut [T],
    im: &mut [T],
    rows: Rows,
    stride: usize,
    batch: usize,
    cols: Columns,
    tw: &[Pair<T>; NT],
    twv: &[Pair<Vector<T, A>>; NT],
    simd: Simd<T, A>,
    seams: Seams<'_, '_, T>,
) where
    T: LaneScalar + Lane,
    A: SimdArch + SimdKernel<T>,
    R: Radix<N, NT>,
{
    let ops = PassOperands {
        re,
        im,
        rows,
        stride,
        batch,
        cols,
        tw,
        twv,
        simd,
    };
    match seams {
        Seams::Planes => {
            pass::<T, A, R, N, NT, false, false, false, false, false>(ops, SeamData::planes())
        }
        Seams::Source(source, row_bits) => {
            pass::<T, A, R, N, NT, true, false, false, false, false>(
                ops,
                SeamData::source(source, row_bits),
            )
        }
        // The table's form is fixed per length (`COMPACT_FOLD_MIN_LEN`) and
        // the direction per transform, so each combination is its own pass
        // rather than a test in the row loop.
        Seams::Fold(fold, FoldDirection::Forward) if fold.lanes == batch => {
            pass::<T, A, R, N, NT, false, true, false, false, false>(ops, SeamData::fold(fold))
        }
        Seams::Fold(fold, FoldDirection::Conjugate) if fold.lanes == batch => {
            pass::<T, A, R, N, NT, false, true, false, false, true>(ops, SeamData::fold(fold))
        }
        Seams::Fold(fold, FoldDirection::Forward) => {
            pass::<T, A, R, N, NT, false, true, true, false, false>(ops, SeamData::fold(fold))
        }
        Seams::Fold(fold, FoldDirection::Conjugate) => {
            pass::<T, A, R, N, NT, false, true, true, false, true>(ops, SeamData::fold(fold))
        }
        Seams::Sink(sink, staging) => pass::<T, A, R, N, NT, false, false, false, true, false>(
            ops,
            SeamData::sink(sink, staging),
        ),
        Seams::FoldSink(fold, FoldDirection::Forward, sink, staging) if fold.lanes == batch => {
            pass::<T, A, R, N, NT, false, true, false, true, false>(
                ops,
                SeamData::fold_sink(fold, sink, staging),
            )
        }
        Seams::FoldSink(fold, FoldDirection::Conjugate, sink, staging) if fold.lanes == batch => {
            pass::<T, A, R, N, NT, false, true, false, true, true>(
                ops,
                SeamData::fold_sink(fold, sink, staging),
            )
        }
        Seams::FoldSink(fold, FoldDirection::Forward, sink, staging) => {
            pass::<T, A, R, N, NT, false, true, true, true, false>(
                ops,
                SeamData::fold_sink(fold, sink, staging),
            )
        }
        Seams::FoldSink(fold, FoldDirection::Conjugate, sink, staging) => {
            pass::<T, A, R, N, NT, false, true, true, true, true>(
                ops,
                SeamData::fold_sink(fold, sink, staging),
            )
        }
    }
}

/// One monomorphized pass: the seams it carries are compile-time, so its
/// row loop tests nothing and computes only the offsets it uses. `CONJ`
/// (meaningful only with `FOLD`) selects the fold table's direction: the
/// dispatcher in [`butterfly_rows`] resolves it once per pass from the
/// seam's [`FoldDirection`], never per element.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope, as the driver"
)]
#[inline(always)]
fn pass<
    T,
    A,
    R,
    const N: usize,
    const NT: usize,
    const SOURCE: bool,
    const FOLD: bool,
    const COMPACT: bool,
    const SINK: bool,
    const CONJ: bool,
>(
    ops: PassOperands<'_, T, A, NT>,
    seam: SeamData<'_, '_, T>,
) where
    T: LaneScalar + Lane,
    A: SimdArch + SimdKernel<T>,
    R: Radix<N, NT>,
{
    let PassOperands {
        re,
        im,
        rows,
        stride,
        batch,
        cols,
        tw,
        twv,
        simd,
    } = ops;
    let SeamData {
        fold,
        source,
        source_bits,
        sink,
        sink_rows,
    } = seam;
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let plane_first = rows.first * stride;
    let plane_step = rows.step * stride;
    // The fold's fine table is one row of `lanes` per data row and its
    // coarse table one scalar per data row and lane group; both index by
    // the plane row, which is the data row for the set that folds.
    let fold_lanes = fold.map_or(1, |fold| fold.lanes);
    let fold_groups = batch / fold_lanes;
    // Both fold tables index by the plane row, affinely like the planes.
    let (fine_first, fine_step) = (rows.first * fold_lanes, rows.step * fold_lanes);
    let (coarse_first, coarse_step) = (rows.first * fold_groups, rows.step * fold_groups);
    // The seam rows keep tables: the bit-reversed row map is not affine, and
    // the sink's is chosen at run time. Both are dead, and so unmaterialized,
    // in the passes without that seam. A sink row's column `k` sits
    // `2 * k - column_base` reals in.
    let source_rows: [usize; N] =
        core::array::from_fn(|i| reverse_row(rows.first + i * rows.step, source_bits) * batch * 2);
    let (sink_base, column_base): ([usize; N], usize) = match sink_rows {
        SinkRows::Staged {
            first,
            pitch,
            first_column,
        } => (
            core::array::from_fn(|i| (first + i * rows.step) * pitch),
            2 * first_column,
        ),
        SinkRows::Direct { row_bits } => (
            core::array::from_fn(|i| reverse_row(rows.first + i * rows.step, row_bits) * batch * 2),
            0,
        ),
    };

    let mut k = cols.start;
    while k + lanes <= cols.end {
        let mut x: [Pair<Vector<T, A>>; N] = core::array::from_fn(|i| {
            if SOURCE {
                load_interleaved::<T, A>(source, source_rows[i] + 2 * k)
            } else {
                let at = plane_first + i * plane_step + k;
                (load::<T, A>(re, at), load::<T, A>(im, at))
            }
        });
        if FOLD {
            let fold = fold.expect("invariant: a folding pass carries its tables");
            for (i, value) in x.iter_mut().enumerate() {
                let fine = fine_first + i * fine_step;
                let coarse = coarse_first + i * coarse_step;
                // The fine row is the whole twiddle row below the compact
                // bound; above it one register of it per lane group, times
                // the group's coarse twiddle broadcast.
                if COMPACT {
                    let fine = (
                        load::<T, A>(&fold.fine_re, fine),
                        load::<T, A>(&fold.fine_im, fine),
                    );
                    let group = coarse + k / fold_lanes;
                    let coarse = (
                        simd.splat(fold.coarse_re[group]),
                        simd.splat(fold.coarse_im[group]),
                    );
                    let w = cmul(fine, coarse);
                    // `CONJ` negates the product's imaginary part: an exact
                    // sign-bit flip, bitwise identical to conjugating the
                    // two factors before the multiply (the product's real
                    // part is a difference of same-sign products, unchanged
                    // under negating both; its imaginary part is a sum of
                    // same-sign products, negated under negating both).
                    let w = if CONJ { (w.0, w.1.neg()) } else { w };
                    *value = cmul(*value, w);
                } else {
                    let at = fine + k;
                    let w = (
                        load::<T, A>(&fold.fine_re, at),
                        load::<T, A>(&fold.fine_im, at),
                    );
                    let w = if CONJ { (w.0, w.1.neg()) } else { w };
                    *value = cmul(*value, w);
                }
            }
        }
        let y = R::apply(x, twv);
        if SINK {
            for (i, value) in y.into_iter().enumerate() {
                let at = sink_base[i] + (2 * k - column_base);
                store_interleaved::<T, A>(value.0, value.1, sink, at);
            }
        } else {
            for (i, value) in y.into_iter().enumerate() {
                let at = plane_first + i * plane_step + k;
                store::<T, A>(value.0, re, at);
                store::<T, A>(value.1, im, at);
            }
        }
        k += lanes;
    }

    // Scalar remainder when the block is not a lane multiple, which is only
    // a batch narrower than one register; the plane column order is then
    // the identity, so column `k` is interleaved sample `k`.
    for k in k..cols.end {
        let mut x: [Pair<T>; N] = core::array::from_fn(|i| {
            if SOURCE {
                let at = source_rows[i] + 2 * k;
                (source[at], source[at + 1])
            } else {
                let at = plane_first + i * plane_step + k;
                (re[at], im[at])
            }
        });
        if FOLD {
            let fold = fold.expect("invariant: a folding pass carries its tables");
            for (i, value) in x.iter_mut().enumerate() {
                let fine = fine_first + i * fine_step;
                let coarse = coarse_first + i * coarse_step;
                let fine = fine + k % fold_lanes;
                let group = coarse + k / fold_lanes;
                let w = cmul(
                    (fold.fine_re[fine], fold.fine_im[fine]),
                    (fold.coarse_re[group], fold.coarse_im[group]),
                );
                // See the vector loop above: negating the product's
                // imaginary part is bitwise the conjugated-factors product.
                let w = if CONJ { (w.0, w.1.neg()) } else { w };
                *value = cmul(*value, w);
            }
        }
        let y = R::apply(x, tw);
        if SINK {
            for (i, value) in y.into_iter().enumerate() {
                let at = sink_base[i] + (2 * k - column_base);
                sink[at] = value.0;
                sink[at + 1] = value.1;
            }
        } else {
            for (i, value) in y.into_iter().enumerate() {
                let at = plane_first + i * plane_step + k;
                re[at] = value.0;
                im[at] = value.1;
            }
        }
    }
}
