//! Vectorized interleaved↔planar boundaries for the batched driver.
//!
//! The per-pass attribution (`RESIDENT_SECTIONS=1`) put 45% of the batched
//! route's budget in data movement. With the native AVX2 interleave network
//! (hermes `HS-AVX2-INTERLEAVE-OVERRIDES`) and capability-hoisted view-chunk
//! access, the reinterleave pass measured 1252 -> 900 TSC at N = 1024 pinned
//! and is kept. The mirrored deinterleave kernel measured *slower* than its
//! scalar loop (1431 -> 1520) — LLVM already auto-vectorizes that loop well,
//! and the earlier "boundary vectorization loses" verdict still holds on the
//! load side — so only the store-side kernel exists here.
//!
//! The driver requests the four-lane width explicitly. A host without that
//! width, or a shape not divisible by it, falls back to the scalar loop, which
//! remains the reference implementation.

use super::BatchedPlanCache;
use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

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

/// Padded planar planes back into interleaved rows, natural order.
pub(crate) struct InterleaveRows<'a, T> {
    /// Padded real plane.
    pub(crate) re: &'a [T],
    /// Padded imaginary plane.
    pub(crate) im: &'a [T],
    /// Interleaved output, `2 * n` lanes.
    pub(crate) data: &'a mut [T],
    /// Row length in complexes.
    pub(crate) m: usize,
    /// Padded plane row stride.
    pub(crate) stride: usize,
}

impl<T: BatchedPlanCache> LaneKernel<T> for InterleaveRows<'_, T> {
    /// Whether the dispatched width handled the pass.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature \
                  frame (hermes LaneKernel contract)"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _capability: Simd<T, A>) -> bool {
        if <A as SimdStorage<T>>::LANE_COUNT != 4 || self.m % 4 != 0 || self.stride % 4 != 0 {
            return false;
        }
        // One bound for the whole pass, so the per-chunk compares vanish.
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        assert!(
            self.re.len() >= self.m * self.stride
                && self.im.len() >= self.m * self.stride
                && self.data.len() >= 2 * self.m * self.m
                && self.m * self.stride % lanes == 0,
            "invariant: both planes hold m padded rows and the output 2m^2 lanes"
        );
        for row in 0..self.m {
            let src_c = row * self.stride / 4;
            let dst_c = row * self.m / 2;
            for r in 0..self.m / 4 {
                let e = chunk::<T, A>(self.re, src_c + r);
                let o = chunk::<T, A>(self.im, src_c + r);
                let (lo, hi) = e.interleave(o);
                put_chunk(lo, self.data, dst_c + 2 * r);
                put_chunk(hi, self.data, dst_c + 2 * r + 1);
            }
        }
        true
    }
}

/// In-place square transpose of both padded planes through in-register
/// `4 x 4` tiles (`Vector::transpose_square`): each off-diagonal tile pair
/// loads eight vectors, transposes both tiles in registers, and stores them
/// exchanged; diagonal tiles transpose in place.
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
        if <A as SimdStorage<T>>::LANE_COUNT != 4 || self.m % 4 != 0 || self.stride % 4 != 0 {
            return false;
        }
        let (m, stride) = (self.m, self.stride);
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        assert!(
            self.re.len() >= m * stride && self.im.len() >= m * stride && m * stride % lanes == 0,
            "invariant: both planes hold m padded rows"
        );
        for plane in [self.re, self.im] {
            for bi in (0..m).step_by(4) {
                // Diagonal tile: transpose in place.
                let base = |r: usize, c: usize| (r * stride + c) / 4;
                let mut tile = [
                    chunk::<T, A>(plane, base(bi, bi)),
                    chunk::<T, A>(plane, base(bi + 1, bi)),
                    chunk::<T, A>(plane, base(bi + 2, bi)),
                    chunk::<T, A>(plane, base(bi + 3, bi)),
                ];
                Vector::transpose_square(&mut tile);
                for (r, row) in tile.into_iter().enumerate() {
                    put_chunk(row, plane, base(bi + r, bi));
                }
                // Off-diagonal pairs: transpose both, store exchanged.
                for bj in (bi + 4..m).step_by(4) {
                    let mut upper = [
                        chunk::<T, A>(plane, base(bi, bj)),
                        chunk::<T, A>(plane, base(bi + 1, bj)),
                        chunk::<T, A>(plane, base(bi + 2, bj)),
                        chunk::<T, A>(plane, base(bi + 3, bj)),
                    ];
                    let mut lower = [
                        chunk::<T, A>(plane, base(bj, bi)),
                        chunk::<T, A>(plane, base(bj + 1, bi)),
                        chunk::<T, A>(plane, base(bj + 2, bi)),
                        chunk::<T, A>(plane, base(bj + 3, bi)),
                    ];
                    Vector::transpose_square(&mut upper);
                    Vector::transpose_square(&mut lower);
                    for (r, row) in lower.into_iter().enumerate() {
                        put_chunk(row, plane, base(bi + r, bj));
                    }
                    for (r, row) in upper.into_iter().enumerate() {
                        put_chunk(row, plane, base(bj + r, bi));
                    }
                }
            }
        }
        true
    }
}
