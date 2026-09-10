//! Vectorized transpose boundaries for the batched driver: in place for a
//! square, into the second plane pair for a rectangle.
//!
//! Both kernels dispatch through `vectorize`, as the stage sets do, so the
//! tile width and the plane column order ([`super::lane_order`]) are the one
//! dispatched backend's, known at compile time inside each kernel: the
//! tile rows are relabeled through the order on one side of the
//! in-register transpose and through its inverse on the other, so data
//! rows stay natural and no pass carries a cross-lane permute. A width
//! without a native tile transpose, or a shape it does not divide, falls
//! back to the scalar loops in the parent module, which remain the
//! reference implementation.

use super::lane_order::{sublane_inverse, sublane_order};
use super::BatchedPlanCache;
use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel, SimdPermute, SimdStorage, Vector};

/// Loads `LANE_COUNT` lanes at element offset `at` of `data`: the tile
/// form, whose rows start wherever a padded stride puts them.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope"
)]
#[inline(always)]
fn lanes_at<T, A>(data: &[T], at: usize) -> Vector<T, A>
where
    T: BatchedPlanCache,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: the kernel asserted the plane holds every tile its loops
    // address before entering them, and `A` is proven by the dispatch token.
    unsafe { Vector::<T, A>::load_unaligned(data.as_ptr().add(at)) }
}

/// Stores `v` at element offset `at` of `data`; the counterpart of [`lanes_at`].
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope"
)]
#[inline(always)]
fn put_lanes_at<T, A>(v: Vector<T, A>, data: &mut [T], at: usize)
where
    T: BatchedPlanCache,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: as `lanes_at` above.
    unsafe { v.store_unaligned(data.as_mut_ptr().add(at)) }
}

/// In-place square transpose of both padded planes through native in-register
/// tiles (`Vector::transpose_square`): each off-diagonal tile pair loads two
/// tiles, transposes both in registers, and stores them exchanged; diagonal
/// tiles transpose in place. The plane column order rides the tile as a
/// register relabeling on both sides of the in-register transpose, so data
/// rows stay natural and columns stay in plane order with every load and
/// store at its natural row.
pub(crate) struct TransposePlanes<'a, T> {
    /// Padded real plane.
    pub(crate) re: &'a mut [T],
    /// Padded imaginary plane.
    pub(crate) im: &'a mut [T],
    /// Square dimension in lanes.
    pub(crate) m: usize,
    /// Padded plane row stride.
    pub(crate) stride: usize,
}

impl<T: BatchedPlanCache> LaneKernel<T> for TransposePlanes<'_, T> {
    /// Whether the dispatched width handled the pass.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature                   frame (hermes LaneKernel contract)"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _capability: Simd<T, A>) -> bool {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        if !matches!(lanes, 2 | 4 | 8 | 16) || self.m % lanes != 0 {
            return false;
        }
        let (m, stride) = (self.m, self.stride);
        assert!(
            self.re.len() >= m * stride && self.im.len() >= m * stride && m * stride % lanes == 0,
            "invariant: both planes hold m padded rows"
        );
        for plane in [self.re, self.im] {
            match lanes {
                2 => transpose_plane::<T, A, 2>(plane, m, stride),
                4 => transpose_plane::<T, A, 4>(plane, m, stride),
                8 => transpose_plane::<T, A, 8>(plane, m, stride),
                16 => transpose_plane::<T, A, 16>(plane, m, stride),
                _ => unreachable!("lane width was validated above"),
            }
        }
        true
    }
}

#[expect(
    clippy::inline_always,
    reason = "the tile width must remain constant inside the target-feature frame"
)]
#[inline(always)]
fn transpose_plane<T, A, const LANES: usize>(plane: &mut [T], m: usize, stride: usize)
where
    T: BatchedPlanCache,
    A: SimdArch + SimdKernel<T>,
{
    // Plane cell `(r, c)` holds logical column `order(c)` of row `r`, so the
    // transposed block must satisfy `out[k][l] = in[order(l)][inverse(k)]`.
    // Feeding the tile transpose the loaded rows in `order` and reading its
    // result rows back in the inverse is exactly that (the order is no
    // involution at the AVX-512 widths), and both are register relabelings
    // at compile time: every load and store keeps its natural row, which is
    // what lets the eight-row `f32` tile keep its addressing.
    let order: [usize; LANES] =
        const { sublane_order::<LANES>(<A as SimdPermute<T>>::SUBLANE_LANES) };
    let inverse: [usize; LANES] =
        const { sublane_inverse::<LANES>(<A as SimdPermute<T>>::SUBLANE_LANES) };
    let transpose = |tile: [Vector<T, A>; LANES]| -> [Vector<T, A>; LANES] {
        let mut ordered: [Vector<T, A>; LANES] = core::array::from_fn(|k| tile[order[k]]);
        Vector::transpose_square(&mut ordered);
        core::array::from_fn(|k| ordered[inverse[k]])
    };
    // Tiles are addressed by element offset: the padded stride need not
    // be a lane multiple, which the eight-wide pad is not at sixteen lanes.
    for bi in (0..m).step_by(LANES) {
        let base = |r: usize, c: usize| r * stride + c;
        let tile = transpose(core::array::from_fn(|r| {
            lanes_at::<T, A>(plane, base(bi + r, bi))
        }));
        for (r, row) in tile.into_iter().enumerate() {
            put_lanes_at(row, plane, base(bi + r, bi));
        }

        for bj in (bi + LANES..m).step_by(LANES) {
            let upper = transpose(core::array::from_fn(|r| {
                lanes_at::<T, A>(plane, base(bi + r, bj))
            }));
            let lower = transpose(core::array::from_fn(|r| {
                lanes_at::<T, A>(plane, base(bj + r, bi))
            }));
            for (r, row) in lower.into_iter().enumerate() {
                put_lanes_at(row, plane, base(bi + r, bj));
            }
            for (r, row) in upper.into_iter().enumerate() {
                put_lanes_at(row, plane, base(bj + r, bi));
            }
        }
    }
}

/// Out-of-place transpose of `rows × cols` padded planes into `cols × rows`
/// padded planes through the same relabeled tile transpose as
/// [`TransposePlanes`]: every source tile lands transposed at its mirrored
/// position in the destination, and no tile is revisited.
pub(crate) struct TransposePlanesInto<'a, T> {
    /// Source real plane.
    pub(crate) src_re: &'a [T],
    /// Source imaginary plane.
    pub(crate) src_im: &'a [T],
    /// Source rows.
    pub(crate) rows: usize,
    /// Source columns, live.
    pub(crate) cols: usize,
    /// Source padded row stride.
    pub(crate) src_stride: usize,
    /// Destination real plane, `cols` rows of `rows` columns.
    pub(crate) dst_re: &'a mut [T],
    /// Destination imaginary plane.
    pub(crate) dst_im: &'a mut [T],
    /// Destination padded row stride.
    pub(crate) dst_stride: usize,
}

impl<T: BatchedPlanCache> LaneKernel<T> for TransposePlanesInto<'_, T> {
    /// Whether the dispatched width handled the pass.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _capability: Simd<T, A>) -> bool {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        if !matches!(lanes, 2 | 4 | 8 | 16) || self.rows % lanes != 0 || self.cols % lanes != 0 {
            return false;
        }
        let Self {
            src_re,
            src_im,
            rows,
            cols,
            src_stride,
            dst_re,
            dst_im,
            dst_stride,
        } = self;
        assert!(
            src_re.len() >= rows * src_stride
                && src_im.len() >= rows * src_stride
                && dst_re.len() >= cols * dst_stride
                && dst_im.len() >= cols * dst_stride,
            "invariant: the source holds rows padded rows and the destination cols"
        );
        for (src, dst) in [(src_re, dst_re), (src_im, dst_im)] {
            match lanes {
                2 => transpose_plane_into::<T, A, 2>(src, rows, cols, src_stride, dst, dst_stride),
                4 => transpose_plane_into::<T, A, 4>(src, rows, cols, src_stride, dst, dst_stride),
                8 => transpose_plane_into::<T, A, 8>(src, rows, cols, src_stride, dst, dst_stride),
                16 => {
                    transpose_plane_into::<T, A, 16>(src, rows, cols, src_stride, dst, dst_stride)
                }
                _ => unreachable!("lane width was validated above"),
            }
        }
        true
    }
}

/// One source tile of `LANES` rows at `(bi, bj)` transposed to `(bj, bi)`
/// of `dst`, relabeled as [`transpose_plane_into`] describes. A named
/// function, not a closure: the block fill of the staged transpose runs
/// every tile through this path, and a closure here compiled out of the
/// target-feature frame, each tile then a call with its shuffles emulated
/// (seven times the in-frame cost).
#[expect(
    clippy::inline_always,
    reason = "the tile width must remain constant inside the target-feature frame"
)]
#[inline(always)]
fn transpose_tile<T, A, const LANES: usize>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    dst_stride: usize,
    bi: usize,
    bj: usize,
) where
    T: BatchedPlanCache,
    A: SimdArch + SimdKernel<T>,
{
    let order: [usize; LANES] =
        const { sublane_order::<LANES>(<A as SimdPermute<T>>::SUBLANE_LANES) };
    let inverse: [usize; LANES] =
        const { sublane_inverse::<LANES>(<A as SimdPermute<T>>::SUBLANE_LANES) };
    let mut ordered: [Vector<T, A>; LANES] =
        core::array::from_fn(|k| lanes_at::<T, A>(src, (bi + order[k]) * src_stride + bj));
    Vector::transpose_square(&mut ordered);
    for k in 0..LANES {
        put_lanes_at(ordered[inverse[k]], dst, (bj + k) * dst_stride + bi);
    }
}

#[expect(
    clippy::inline_always,
    reason = "the tile width must remain constant inside the target-feature frame"
)]
#[inline(always)]
fn transpose_plane_into<T, A, const LANES: usize>(
    src: &[T],
    rows: usize,
    cols: usize,
    src_stride: usize,
    dst: &mut [T],
    dst_stride: usize,
) where
    T: BatchedPlanCache,
    A: SimdArch + SimdKernel<T>,
{
    // The same relabeling as `transpose_plane`: `out[k][l] =
    // in[order(l)][inverse(k)]` within each tile, the tile itself moving
    // from `(bi, bj)` to `(bj, bi)`.
    let order: [usize; LANES] =
        const { sublane_order::<LANES>(<A as SimdPermute<T>>::SUBLANE_LANES) };
    let inverse: [usize; LANES] =
        const { sublane_inverse::<LANES>(<A as SimdPermute<T>>::SUBLANE_LANES) };
    // Two source tiles per step, so two independent shuffle chains are in
    // flight as in the in-place kernel's tile pair; destination rows are
    // written sequentially, each store extending a row the previous tile
    // just wrote. A row count of one odd tile block keeps the single form.
    for bj in (0..cols).step_by(LANES) {
        let mut bi = 0;
        while bi + 2 * LANES <= rows {
            let mut first: [Vector<T, A>; LANES] =
                core::array::from_fn(|k| lanes_at::<T, A>(src, (bi + order[k]) * src_stride + bj));
            let mut second: [Vector<T, A>; LANES] = core::array::from_fn(|k| {
                lanes_at::<T, A>(src, (bi + LANES + order[k]) * src_stride + bj)
            });
            Vector::transpose_square(&mut first);
            Vector::transpose_square(&mut second);
            for k in 0..LANES {
                let row = (bj + k) * dst_stride;
                put_lanes_at(first[inverse[k]], dst, row + bi);
                put_lanes_at(second[inverse[k]], dst, row + bi + LANES);
            }
            bi += 2 * LANES;
        }
        if bi < rows {
            transpose_tile::<T, A, LANES>(src, src_stride, dst, dst_stride, bi, bj);
        }
    }
}
