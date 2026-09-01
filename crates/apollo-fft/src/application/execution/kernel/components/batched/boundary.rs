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
//! The driver requests the scalar-selected preferred width explicitly: eight
//! lanes for f32, four for f64, then four as the portable SIMD fallback. A host
//! without either width, or a shape not divisible by it, falls back to the
//! scalar loop, which remains the reference implementation. All three boundary
//! kernels here — transpose, half combine, and reinterleave — take that width
//! from `BatchedPlanCache::BOUNDARY_LANES`; one const-generic body serves both
//! widths and dispatch stays outside the tile loops.

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
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        if !matches!(lanes, 4 | 8) || self.m % lanes != 0 || self.stride % lanes != 0 {
            return false;
        }
        // One bound for the whole pass, so the per-chunk compares vanish.
        assert!(
            self.re.len() >= self.m * self.stride
                && self.im.len() >= self.m * self.stride
                && self.data.len() >= 2 * self.m * self.m
                && self.m * self.stride % lanes == 0,
            "invariant: both planes hold m padded rows and the output 2m^2 lanes"
        );
        // The stage set that produced these planes is decimated in
        // frequency, so its output rows are bit-reversed. Absorbing that
        // here is the whole of the permutation the route used to spend a
        // separate pass on, and it rides the *write* side: plane row `p`
        // holds output row `rev(p)`, bit reversal being an involution, so
        // the plane reads stay sequential and only the stores scatter
        // (gap_audit.md#sink-permutation).
        //
        // Both indices below count whole `lanes`-wide chunks. A source chunk
        // is `lanes` reals of one plane; a destination chunk is `lanes`
        // interleaved reals, so one row spans `2 * m / lanes` of them and each
        // read pair yields two of them. `stride % lanes == 0` and
        // `m % lanes == 0` are guarded above, so both divisions are exact.
        let bits = self.m.trailing_zeros();
        for row in 0..self.m {
            let src_c = row * self.stride / lanes;
            let dst_c = (row.reverse_bits() >> (usize::BITS - bits)) * 2 * self.m / lanes;
            for r in 0..self.m / lanes {
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

/// Combines two planar half-transforms and writes interleaved output.
///
/// Each vector covers `LANE_COUNT` consecutive complex outputs. The even and
/// odd planes supply separate real/imaginary registers, while the cached
/// twiddle and the two output halves use the public interleaved complex layout.
/// The row permutation rides the output address exactly as in
/// [`InterleaveRows`].
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
        if !matches!(lanes, 4 | 8) || self.m % lanes != 0 || self.stride % lanes != 0 {
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
                let (twiddle_re, twiddle_im) = twiddle_lo.deinterleave(twiddle_hi);
                let rotated_re = twiddle_re.mul_add(odd_re, -(twiddle_im * odd_im));
                let rotated_im = twiddle_re.mul_add(odd_im, twiddle_im * odd_re);

                let (low_lo, low_hi) = (even_re + rotated_re).interleave(even_im + rotated_im);
                let (high_lo, high_hi) = (even_re - rotated_re).interleave(even_im - rotated_im);
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
/// tiles transpose in place.
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
        if !matches!(lanes, 4 | 8) || self.m % lanes != 0 || self.stride % lanes != 0 {
            return false;
        }
        let (m, stride) = (self.m, self.stride);
        assert!(
            self.re.len() >= m * stride && self.im.len() >= m * stride && m * stride % lanes == 0,
            "invariant: both planes hold m padded rows"
        );
        match lanes {
            4 => {
                for plane in [self.re, self.im] {
                    transpose_plane::<T, A, 4>(plane, m, stride);
                }
            }
            8 => {
                for plane in [self.re, self.im] {
                    transpose_plane::<T, A, 8>(plane, m, stride);
                }
            }
            _ => unreachable!("lane width was validated above"),
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
    for bi in (0..m).step_by(LANES) {
        let base = |r: usize, c: usize| (r * stride + c) / LANES;
        let mut tile: [Vector<T, A>; LANES] =
            core::array::from_fn(|r| chunk::<T, A>(plane, base(bi + r, bi)));
        Vector::transpose_square(&mut tile);
        for (r, row) in tile.into_iter().enumerate() {
            put_chunk(row, plane, base(bi + r, bi));
        }

        for bj in (bi + LANES..m).step_by(LANES) {
            let mut upper: [Vector<T, A>; LANES] =
                core::array::from_fn(|r| chunk::<T, A>(plane, base(bi + r, bj)));
            let mut lower: [Vector<T, A>; LANES] =
                core::array::from_fn(|r| chunk::<T, A>(plane, base(bj + r, bi)));
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
