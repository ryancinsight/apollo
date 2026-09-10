//! The four-step driver over the padded planar layout.

use super::boundary::TransposedPlanes;
use super::dif::run_batched_dif;
use super::dit::run_batched;
use super::lane_order::LaneOrder;
use super::plane::{
    planar_applies, plane_geometry, scratch_len, split_plane, transpose_planes,
    transpose_planes_into, PlaneView, ROW_PAD,
};
use super::{boundary, sweep, BatchedPlanCache};
use eunomia::Complex;

/// How the rectangle's planes reach the second stage set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum TransposeRoute {
    /// A transpose pass into the second plane pair before the set runs.
    Pass,
    /// The set's first sweep transposes each column block into the staging
    /// buffer itself, and there is no pass (ADR 0060, revision of
    /// 2026-09-10).
    Staged,
}

impl TransposeRoute {
    /// The route for `rows × cols` first planes at the dispatched tile width.
    pub(super) fn select<T>(rows: usize, cols: usize, lanes: usize) -> Self {
        if boundary::staged_transpose_applies::<T>(rows, cols, lanes) {
            Self::Staged
        } else {
            Self::Pass
        }
    }
}

/// Four-step FFT over the padded planar layout, the transpose route chosen
/// by [`TransposeRoute::select`].
///
/// # Panics
///
/// As [`four_step_batched_by`].
pub(crate) fn four_step_batched<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    four_step_batched_by::<T, INVERSE>(data, scratch, None);
}

/// Four-step FFT over the padded planar layout by `route`, or for `None`
/// by the route [`TransposeRoute::select`] chooses; squares transpose in
/// place whatever the route.
///
/// Three steps and no more: the first stage set reads the caller's rows in
/// bit-reversed row order straight out of `data`, one transpose moves the
/// planes to the second axis (in place for a square; into the second plane
/// pair for the rectangle an odd power runs as, or inside the second set's
/// first sweep where the transposed block fits the staging buffer, ADR
/// 0060), and the second
/// stage set folds the four-step twiddle into its first loads and writes
/// `data` back from its last. The per-element operation order is the same
/// whatever the shape.
///
/// # Panics
///
/// Panics if `data.len()` is not a length [`planar_applies`] admits, or if
/// `scratch` is shorter than [`scratch_len`].
pub(super) fn four_step_batched_by<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    route: Option<TransposeRoute>,
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let n = data.len();
    assert!(planar_applies(n), "requires a planar power of two");
    let (n1, n2) = plane_geometry(n);
    let (stride_a, stride_b) = (n2 + ROW_PAD, n1 + ROW_PAD);
    let plane_a = n1 * stride_a;
    let plane_b = if n1 == n2 { 0 } else { n2 * stride_b };
    assert!(
        scratch.len() >= scratch_len(n),
        "scratch must hold the padded planes and the staging block"
    );
    // The planes start on a cache line; the slack in `scratch_len` absorbs
    // the shift (`PLANE_ALIGN_SLACK`).
    let lead = (64 - (scratch.as_ptr() as usize) % 64) % 64 / core::mem::size_of::<Complex<T>>();
    let scratch = &mut scratch[lead..];
    let (a, rest) = scratch.split_at_mut(plane_a);
    let (b, rest) = rest.split_at_mut(plane_b);
    let staging: &mut [T] = eunomia::layout::cast_slice_mut(&mut rest[..sweep::STAGING_LEN]);
    let data: &mut [T] = eunomia::layout::cast_slice_mut(data);
    // Both sets' batches are at least a register wide from 512 up, so one
    // order serves both; below that the square's order is the identity.
    let order = LaneOrder::for_batch::<T>(n1.min(n2));
    let (a_re, a_im) = split_plane(a, plane_a);

    // 1. `n2` transforms of length `n1` along the first axis; the caller's
    //    rows are batch-major for this direction, so the first pass reads
    //    them in place and no transpose is needed.
    let plan = T::cached_plan::<INVERSE>(n1);
    sect!("stages1", {
        run_batched(a_re, a_im, plan.as_ref(), Some(&*data), n2, stride_a)
    });

    // 2. Transpose so the second axis becomes batch-major. Pure exchange:
    //    the four-step twiddle rides stage set 2's first loads below. The
    //    same selector as the stage sets, so the tile width and the plane
    //    column order are the one backend's. The rectangle whose transposed
    //    column block fits the staging buffer has no pass here: the second
    //    set's first sweep transposes the planes block by block itself.
    let route = route.unwrap_or_else(|| TransposeRoute::select::<T>(n1, n2, order.lanes()));
    let (re, im, stride, transposed) = if n1 == n2 {
        sect!("transpose", {
            let handled = hermes_simd::vectorize(boundary::TransposePlanes {
                re: &mut *a_re,
                im: &mut *a_im,
                m: n1,
                stride: stride_a,
            });
            if !handled {
                transpose_planes(a_re, a_im, n1, stride_a, order);
            }
        });
        (a_re, a_im, stride_a, None)
    } else if route == TransposeRoute::Staged {
        let (b_re, b_im) = split_plane(b, plane_b);
        let source = TransposedPlanes {
            re: &*a_re,
            im: &*a_im,
            stride: stride_a,
        };
        (b_re, b_im, stride_b, Some(source))
    } else {
        let (b_re, b_im) = split_plane(b, plane_b);
        sect!("transpose", {
            let handled = hermes_simd::vectorize(boundary::TransposePlanesInto {
                src_re: &*a_re,
                src_im: &*a_im,
                rows: n1,
                cols: n2,
                src_stride: stride_a,
                dst_re: &mut *b_re,
                dst_im: &mut *b_im,
                dst_stride: stride_b,
            });
            if !handled {
                transpose_planes_into(
                    PlaneView {
                        re: a_re,
                        im: a_im,
                        rows: n1,
                        cols: n2,
                        stride: stride_a,
                    },
                    b_re,
                    b_im,
                    stride_b,
                    order,
                );
            }
        });
        (b_re, b_im, stride_b, None)
    };

    // 3. `n1` transforms of length `n2` along the second axis, with the
    //    four-step twiddle `W_N^(n2 k1)` folded into the first stage's loads
    //    from a table with the planes' own shape. This set is decimated in
    //    frequency: the transpose leaves natural row order, which is what
    //    DIF consumes, and it leaves its output bit-reversed, which the sink
    //    absorbs by writing plane row `p` to row `rev(p)` — the mirror of
    //    the source's free permutation. Output `k1 + n1 k2` lands at row
    //    `k2`, column `k1`.
    let plan = if n1 == n2 {
        plan
    } else {
        T::cached_plan::<INVERSE>(n2)
    };
    let fold = T::cached_four_step_fold::<INVERSE>(n, n2, n1);
    sect!("stages2", {
        run_batched_dif(
            re,
            im,
            plan.as_ref(),
            Some(fold.as_ref()),
            Some(data),
            staging,
            transposed,
            n1,
            stride,
        )
    });
}
