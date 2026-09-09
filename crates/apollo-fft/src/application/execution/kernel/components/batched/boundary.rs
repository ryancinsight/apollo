//! Vectorized transpose and half-combine boundaries for the batched driver.
//!
//! Both kernels dispatch through `vectorize`, as the stage sets do, so the
//! tile width and the plane column order ([`super::lane_order`]) are the one
//! dispatched backend's, known at compile time inside each kernel: the
//! transpose permutes its tile rows through the order to leave data rows
//! natural, and the combine reads its planes and its interleaved twiddles in
//! that order through the sub-lane unpacks, so neither pass carries a
//! cross-lane permute. A width without a native tile
//! transpose, or a shape it does not divide, falls back to the scalar loops
//! in the parent module, which remain the reference implementation.

use super::lane_order::sublane_order;
use super::BatchedPlanCache;
use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel, SimdPermute, SimdStorage, Vector};

/// Loads chunk `index` (a `LANE_COUNT`-lane group) from `data`.
///
/// The checked `SimdView` accessor asserts `offset + LANE_COUNT <= len()`
/// on every touch, and these planes arrive as runtime-sized slices, so no
/// such check can fold: the transpose and reinterleave passes carried a
/// compare and a branch to a panic block around each vector moved
/// (gap_audit.md#base128-bounds). Every caller below derives its chunk index
/// from `m` and `stride` with the plane's own extent, which the wrapping
/// kernel asserts once on entry.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope"
)]
#[inline(always)]
fn chunk<T, A>(data: &[T], index: usize) -> Vector<T, A>
where
    T: BatchedPlanCache,
    A: SimdArch + SimdKernel<T>,
{
    let at = index * <A as SimdStorage<T>>::LANE_COUNT;
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: the kernel asserted the plane holds every chunk its loops
    // address before entering them, and `A` is proven by the dispatch token.
    unsafe { Vector::<T, A>::load_unaligned(data.as_ptr().add(at)) }
}

/// Stores `v` into chunk `index` of `data`; the counterpart of [`chunk`].
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope"
)]
#[inline(always)]
fn put_chunk<T, A>(v: Vector<T, A>, data: &mut [T], index: usize)
where
    T: BatchedPlanCache,
    A: SimdArch + SimdKernel<T>,
{
    let at = index * <A as SimdStorage<T>>::LANE_COUNT;
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: as `chunk` above.
    unsafe { v.store_unaligned(data.as_mut_ptr().add(at)) }
}

/// Combines two planar half-transforms and writes interleaved output.
///
/// Each vector covers `LANE_COUNT` consecutive complex outputs. The even and
/// odd planes supply separate real/imaginary registers, while the cached
/// twiddle and the two output halves use the public interleaved complex layout.
/// The planes and the deinterleaved twiddle share the plane column order,
/// and the row permutation rides the output address.
pub(crate) struct CombinePlanarHalves<'a, T> {
    /// Even-half real plane.
    pub(crate) even_re: &'a [T],
    /// Even-half imaginary plane.
    pub(crate) even_im: &'a [T],
    /// Odd-half real plane.
    pub(crate) odd_re: &'a [T],
    /// Odd-half imaginary plane.
    pub(crate) odd_im: &'a [T],
    /// Interleaved complex twiddles, represented as scalar lanes.
    pub(crate) twiddles: &'a [T],
    /// Interleaved low output half, represented as scalar lanes.
    pub(crate) low: &'a mut [T],
    /// Interleaved high output half, represented as scalar lanes.
    pub(crate) high: &'a mut [T],
    /// Live row length in complexes.
    pub(crate) m: usize,
    /// Padded plane row stride.
    pub(crate) stride: usize,
}

impl<T: BatchedPlanCache> LaneKernel<T> for CombinePlanarHalves<'_, T> {
    /// Whether the dispatched width handled the pass.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _capability: Simd<T, A>) -> bool {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        if lanes < 2 || self.m % lanes != 0 || self.stride % lanes != 0 {
            return false;
        }

        let half = self.m * self.m;
        let plane = self.m * self.stride;
        assert!(
            self.even_re.len() >= plane
                && self.even_im.len() >= plane
                && self.odd_re.len() >= plane
                && self.odd_im.len() >= plane
                && self.twiddles.len() >= 2 * half
                && self.low.len() >= 2 * half
                && self.high.len() >= 2 * half,
            "invariant: combine inputs hold two padded planes, half twiddles, and both output halves"
        );

        let bits = self.m.trailing_zeros();
        for row in 0..self.m {
            let base = row * self.stride;
            let dst = (row.reverse_bits() >> (usize::BITS - bits)) * self.m;
            for column in (0..self.m).step_by(lanes) {
                let plane_chunk = (base + column) / lanes;
                let even_re = chunk::<T, A>(self.even_re, plane_chunk);
                let even_im = chunk::<T, A>(self.even_im, plane_chunk);
                let odd_re = chunk::<T, A>(self.odd_re, plane_chunk);
                let odd_im = chunk::<T, A>(self.odd_im, plane_chunk);

                let output_chunk = 2 * (dst + column) / lanes;
                let twiddle_lo = chunk::<T, A>(self.twiddles, output_chunk);
                let twiddle_hi = chunk::<T, A>(self.twiddles, output_chunk + 1);
                let (twiddle_re, twiddle_im) = twiddle_lo.deinterleave_sublanes(twiddle_hi);
                let rotated_re = twiddle_re.mul_add(odd_re, -(twiddle_im * odd_im));
                let rotated_im = twiddle_re.mul_add(odd_im, twiddle_im * odd_re);

                let (low_lo, low_hi) =
                    (even_re + rotated_re).interleave_sublanes(even_im + rotated_im);
                let (high_lo, high_hi) =
                    (even_re - rotated_re).interleave_sublanes(even_im - rotated_im);
                put_chunk(low_lo, self.low, output_chunk);
                put_chunk(low_hi, self.low, output_chunk + 1);
                put_chunk(high_lo, self.high, output_chunk);
                put_chunk(high_hi, self.high, output_chunk + 1);
            }
        }
        true
    }
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
        if !matches!(lanes, 2 | 4 | 8 | 16) || self.m % lanes != 0 || self.stride % lanes != 0 {
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
    // transposed block must satisfy `out[k][l] = in[order(l)][order(k)]`.
    // Feeding the tile transpose the loaded rows in `order` and reading its
    // result rows back in `order` is exactly that, and both are register
    // relabelings at compile time: every load and store keeps its natural
    // row, which is what lets the eight-row `f32` tile keep its addressing.
    let order: [usize; LANES] =
        const { sublane_order::<LANES>(<A as SimdPermute<T>>::SUBLANE_LANES) };
    let transpose = |tile: [Vector<T, A>; LANES]| -> [Vector<T, A>; LANES] {
        let mut ordered: [Vector<T, A>; LANES] = core::array::from_fn(|k| tile[order[k]]);
        Vector::transpose_square(&mut ordered);
        core::array::from_fn(|k| ordered[order[k]])
    };
    for bi in (0..m).step_by(LANES) {
        let base = |r: usize, c: usize| (r * stride + c) / LANES;
        let tile = transpose(core::array::from_fn(|r| {
            chunk::<T, A>(plane, base(bi + r, bi))
        }));
        for (r, row) in tile.into_iter().enumerate() {
            put_chunk(row, plane, base(bi + r, bi));
        }

        for bj in (bi + LANES..m).step_by(LANES) {
            let upper = transpose(core::array::from_fn(|r| {
                chunk::<T, A>(plane, base(bi + r, bj))
            }));
            let lower = transpose(core::array::from_fn(|r| {
                chunk::<T, A>(plane, base(bj + r, bi))
            }));
            for (r, row) in lower.into_iter().enumerate() {
                put_chunk(row, plane, base(bi + r, bj));
            }
            for (r, row) in upper.into_iter().enumerate() {
                put_chunk(row, plane, base(bj + r, bi));
            }
        }
    }
}
