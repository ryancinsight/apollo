//! The row-set driver shared by both stage sets: one pass over one row set,
//! monomorphized per seam combination so its row loop tests nothing and
//! computes only the offsets it uses.

use hermes_simd::{LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

use super::fold::FourStepFold;
use super::lane::Lane;
use super::radix::{cmul, Pair, Radix};
use super::register::{load, load_interleaved, reverse_row, store, store_interleaved};
use super::seams::{Columns, Rows, Seams, SinkRows, StagedRows};

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
    match seams {
        Seams::Planes => pass::<T, A, R, N, NT, false, false, false, false, false>(
            re,
            im,
            rows,
            stride,
            batch,
            cols,
            tw,
            twv,
            simd,
            None,
            &[],
            0,
            &mut [],
            SinkRows::NONE,
            &[],
            &[],
            StagedRows::NONE,
        ),
        Seams::Source(source, row_bits) => {
            pass::<T, A, R, N, NT, true, false, false, false, false>(
                re,
                im,
                rows,
                stride,
                batch,
                cols,
                tw,
                twv,
                simd,
                None,
                source,
                row_bits,
                &mut [],
                SinkRows::NONE,
                &[],
                &[],
                StagedRows::NONE,
            )
        }
        Seams::Fold(fold) => {
            // The table's form is fixed per length (`COMPACT_FOLD_MIN_LEN`),
            // so each form is its own pass rather than a test in the row loop.
            if fold.lanes == batch {
                pass::<T, A, R, N, NT, false, true, false, false, false>(
                    re,
                    im,
                    rows,
                    stride,
                    batch,
                    cols,
                    tw,
                    twv,
                    simd,
                    Some(fold),
                    &[],
                    0,
                    &mut [],
                    SinkRows::NONE,
                    &[],
                    &[],
                    StagedRows::NONE,
                )
            } else {
                pass::<T, A, R, N, NT, false, true, true, false, false>(
                    re,
                    im,
                    rows,
                    stride,
                    batch,
                    cols,
                    tw,
                    twv,
                    simd,
                    Some(fold),
                    &[],
                    0,
                    &mut [],
                    SinkRows::NONE,
                    &[],
                    &[],
                    StagedRows::NONE,
                )
            }
        }
        Seams::Sink(sink, staging) => pass::<T, A, R, N, NT, false, false, false, true, false>(
            re,
            im,
            rows,
            stride,
            batch,
            cols,
            tw,
            twv,
            simd,
            None,
            &[],
            0,
            sink,
            staging,
            &[],
            &[],
            StagedRows::NONE,
        ),
        Seams::FoldSink(fold, sink, staging) => {
            // The table's form is fixed per length (`COMPACT_FOLD_MIN_LEN`),
            // so each form is its own pass rather than a test in the row loop.
            if fold.lanes == batch {
                pass::<T, A, R, N, NT, false, true, false, true, false>(
                    re,
                    im,
                    rows,
                    stride,
                    batch,
                    cols,
                    tw,
                    twv,
                    simd,
                    Some(fold),
                    &[],
                    0,
                    sink,
                    staging,
                    &[],
                    &[],
                    StagedRows::NONE,
                )
            } else {
                pass::<T, A, R, N, NT, false, true, true, true, false>(
                    re,
                    im,
                    rows,
                    stride,
                    batch,
                    cols,
                    tw,
                    twv,
                    simd,
                    Some(fold),
                    &[],
                    0,
                    sink,
                    staging,
                    &[],
                    &[],
                    StagedRows::NONE,
                )
            }
        }
        Seams::FoldStaged(fold, staged_re, staged_im, staged_rows) => {
            // The table's form is fixed per length (`COMPACT_FOLD_MIN_LEN`),
            // so each form is its own pass rather than a test in the row loop.
            if fold.lanes == batch {
                pass::<T, A, R, N, NT, false, true, false, false, true>(
                    re,
                    im,
                    rows,
                    stride,
                    batch,
                    cols,
                    tw,
                    twv,
                    simd,
                    Some(fold),
                    &[],
                    0,
                    &mut [],
                    SinkRows::NONE,
                    staged_re,
                    staged_im,
                    staged_rows,
                )
            } else {
                pass::<T, A, R, N, NT, false, true, true, false, true>(
                    re,
                    im,
                    rows,
                    stride,
                    batch,
                    cols,
                    tw,
                    twv,
                    simd,
                    Some(fold),
                    &[],
                    0,
                    &mut [],
                    SinkRows::NONE,
                    staged_re,
                    staged_im,
                    staged_rows,
                )
            }
        }
    }
}

/// One monomorphized pass: the seams it carries are compile-time, so its
/// row loop tests nothing and computes only the offsets it uses.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope, as the driver"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the pass is the inner function of one driver; its arguments are the driver's, unbundled so each monomorphization keeps only the seams it carries in registers"
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
    const STAGED: bool,
>(
    re: &mut [T],
    im: &mut [T],
    rows: Rows,
    stride: usize,
    batch: usize,
    cols: Columns,
    tw: &[Pair<T>; NT],
    twv: &[Pair<Vector<T, A>>; NT],
    simd: Simd<T, A>,
    fold: Option<&FourStepFold<T>>,
    source: &[T],
    source_bits: u32,
    sink: &mut [T],
    sink_rows: SinkRows,
    staged_re: &[T],
    staged_im: &[T],
    staged_rows: StagedRows,
) where
    T: LaneScalar + Lane,
    A: SimdArch + SimdKernel<T>,
    R: Radix<N, NT>,
{
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

    // The staged rows hold the whole plane at their own pitch, so they
    // index affinely like the planes.
    let staged_base: [usize; N] =
        core::array::from_fn(|i| (rows.first + i * rows.step) * staged_rows.pitch);

    let mut k = cols.start;
    while k + lanes <= cols.end {
        let mut x: [Pair<Vector<T, A>>; N] = core::array::from_fn(|i| {
            if SOURCE {
                load_interleaved::<T, A>(source, source_rows[i] + 2 * k)
            } else if STAGED {
                let at = staged_base[i] + (k - staged_rows.first_column);
                (load::<T, A>(staged_re, at), load::<T, A>(staged_im, at))
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
                    *value = cmul(*value, cmul(fine, coarse));
                } else {
                    let at = fine + k;
                    let w = (
                        load::<T, A>(&fold.fine_re, at),
                        load::<T, A>(&fold.fine_im, at),
                    );
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
            } else if STAGED {
                let at = staged_base[i] + (k - staged_rows.first_column);
                (staged_re[at], staged_im[at])
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
