//! Vectorized gather and combine for the small-size split.
//!
//! The split's piece attribution put its non-base cost at 193 ns of 558 at
//! n = 256 and 641 of 1371 at n = 512 — the gather and the combine chain
//! together outweighing the gap to the reference, while the scalar-combine
//! verdict on record predated the instance-major kernel entirely
//! (gap_audit.md#split-boundary).
//!
//! The combine runs planar: chunks deinterleave into real and imaginary
//! vectors, the twiddle rotation is two multiplies and two FMAs with no
//! per-sample shuffle, and the butterfly's sum and difference interleave
//! back on the way out. The twiddles deinterleave on the fly from the same
//! interleaved cache the scalar loop read — measured against a dup-split
//! variant this costs one extra shuffle pair per four samples and avoids a
//! second cached table representation.
//!
//! The gather is the phase-one concatenation network: subsequence chunks are
//! whole-register blends of parent chunk pairs, at pair distance 1 for two
//! blocks and distance 2 for four.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

/// One combine butterfly over four complex samples held planar.
///
/// `X[j] = E[j] + W^j O[j]` and `X[j + len] = E[j] - W^j O[j]`, returned as
/// interleaved chunk pairs ready to store.
#[expect(
    clippy::inline_always,
    reason = "must fold into the dispatcher's target-feature scope; an \
              out-of-line call here compiles at baseline"
)]
#[inline(always)]
#[expect(clippy::type_complexity, reason = "four result registers, two per output row")]
fn combine_quad<T, A>(
    e0: Vector<T, A>,
    e1: Vector<T, A>,
    o0: Vector<T, A>,
    o1: Vector<T, A>,
    w0: Vector<T, A>,
    w1: Vector<T, A>,
) -> ((Vector<T, A>, Vector<T, A>), (Vector<T, A>, Vector<T, A>))
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let (er, ei) = e0.deinterleave(e1);
    let (or_, oi) = o0.deinterleave(o1);
    let (wr, wi) = w0.deinterleave(w1);
    let rr = wr.mul_add(or_, -(wi * oi));
    let ri = wr.mul_add(oi, wi * or_);
    let low = (er + rr).interleave(ei + ri);
    let high = (er - rr).interleave(ei - ri);
    (low, high)
}

/// In-place combine of adjacent transform pairs: `low`/`high` are read as
/// `E`/`O` and rewritten as the butterfly's sum and difference rows.
pub(super) struct CombineInPlace<'a, T> {
    pub(super) low: &'a mut [T],
    pub(super) high: &'a mut [T],
    /// Interleaved `W^j` lanes, one complex per output pair.
    pub(super) tw: &'a [T],
}

impl<T: LaneScalar + MixedRadixScalar> LaneKernel<T> for CombineInPlace<'_, T> {
    /// Whether the dispatched width handled the pass.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature \
                  frame (hermes LaneKernel contract)"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        if lanes != 4 || self.low.len() % 8 != 0 {
            return false;
        }
        assert!(
            self.high.len() == self.low.len() && self.tw.len() >= self.low.len(),
            "invariant: matched halves and one twiddle per pair"
        );
        let chunks = self.low.len() / 4;
        let tw = simd.view(self.tw);
        let mut c = 0;
        while c < chunks {
            let low_v = simd.view(&*self.low);
            let high_v = simd.view(&*self.high);
            let e0 = Vector::from_view_chunk(&low_v, c);
            let e1 = Vector::from_view_chunk(&low_v, c + 1);
            let o0 = Vector::from_view_chunk(&high_v, c);
            let o1 = Vector::from_view_chunk(&high_v, c + 1);
            let w0 = Vector::from_view_chunk(&tw, c);
            let w1 = Vector::from_view_chunk(&tw, c + 1);
            let ((l0, l1), (h0, h1)) = combine_quad(e0, e1, o0, o1, w0, w1);
            let mut low_m = simd.view_mut(&mut *self.low);
            l0.store_to_view_chunk(&mut low_m, c);
            l1.store_to_view_chunk(&mut low_m, c + 1);
            let mut high_m = simd.view_mut(&mut *self.high);
            h0.store_to_view_chunk(&mut high_m, c);
            h1.store_to_view_chunk(&mut high_m, c + 1);
            c += 2;
        }
        true
    }
}

/// The final combine, reading `even`/`odd` from scratch and writing the
/// assembled transform to `low`/`high` in the caller's buffer.
pub(super) struct CombineInto<'a, T> {
    pub(super) even: &'a [T],
    pub(super) odd: &'a [T],
    pub(super) low: &'a mut [T],
    pub(super) high: &'a mut [T],
    /// Interleaved `W^j` lanes, one complex per output pair.
    pub(super) tw: &'a [T],
}

impl<T: LaneScalar + MixedRadixScalar> LaneKernel<T> for CombineInto<'_, T> {
    /// Whether the dispatched width handled the pass.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature \
                  frame (hermes LaneKernel contract)"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        if lanes != 4 || self.even.len() % 8 != 0 {
            return false;
        }
        assert!(
            self.odd.len() == self.even.len()
                && self.low.len() == self.even.len()
                && self.high.len() == self.even.len()
                && self.tw.len() >= self.even.len(),
            "invariant: matched quarters and one twiddle per pair"
        );
        let chunks = self.even.len() / 4;
        let even = simd.view(self.even);
        let odd = simd.view(self.odd);
        let tw = simd.view(self.tw);
        let mut low = simd.view_mut(&mut *self.low);
        let mut high = simd.view_mut(&mut *self.high);
        let mut c = 0;
        while c < chunks {
            let e0 = Vector::from_view_chunk(&even, c);
            let e1 = Vector::from_view_chunk(&even, c + 1);
            let o0 = Vector::from_view_chunk(&odd, c);
            let o1 = Vector::from_view_chunk(&odd, c + 1);
            let w0 = Vector::from_view_chunk(&tw, c);
            let w1 = Vector::from_view_chunk(&tw, c + 1);
            let ((l0, l1), (h0, h1)) = combine_quad(e0, e1, o0, o1, w0, w1);
            l0.store_to_view_chunk(&mut low, c);
            l1.store_to_view_chunk(&mut low, c + 1);
            h0.store_to_view_chunk(&mut high, c);
            h1.store_to_view_chunk(&mut high, c + 1);
            c += 2;
        }
        true
    }
}

/// Gathers the split's stride-`BLOCKS` subsequences into contiguous blocks.
///
/// A parent chunk holds two adjacent samples, so a subsequence chunk is a
/// whole-register concatenation of two parent chunks — the phase-one blend
/// network. `BLOCKS = 2` pairs neighbours; `BLOCKS = 4` pairs at distance
/// two and lands the four subsequences in the bit-reversed block order the
/// combine expects.
pub(super) struct GatherBlocks<'a, T, const BLOCKS: usize> {
    pub(super) src: &'a [T],
    pub(super) dst: &'a mut [T],
}

impl<T: LaneScalar + MixedRadixScalar, const BLOCKS: usize> LaneKernel<T>
    for GatherBlocks<'_, T, BLOCKS>
{
    /// Whether the dispatched width handled the pass.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature \
                  frame (hermes LaneKernel contract)"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        if lanes != 4 {
            return false;
        }
        assert!(
            (BLOCKS == 2 || BLOCKS == 4) && self.src.len() == BLOCKS * 256,
            "invariant: two or four 128-sample blocks"
        );
        assert_eq!(self.dst.len(), self.src.len(), "matched gather buffers");
        let src = simd.view(self.src);
        let mut dst = simd.view_mut(&mut *self.dst);
        // The high-complex blend mask, as in the base kernel's phase one.
        let zero = T::from_precise(0.0);
        let neg = T::from_precise(-1.0);
        let mask = [zero, zero, neg, neg];
        let mask = simd.view(&mask);
        let hi_mask = Vector::<T, A>::from_view_chunk(&mask, 0);
        // Per group: parent chunks at pair distance `BLOCKS / 2` concatenate
        // into one chunk of each of two subsequences.
        let groups = BLOCKS * 32;
        let dist = BLOCKS / 2;
        for g in 0..groups {
            // Consecutive pair slots within a stride-`dist` group layout.
            let base = (g / dist) * 2 * dist + (g % dist);
            let lo = Vector::<T, A>::from_view_chunk(&src, base);
            let hi = Vector::<T, A>::from_view_chunk(&src, base + dist);
            let even = hi_mask.blend(hi.swap_pairs(), lo);
            let odd = hi_mask.blend(hi, lo.swap_pairs());
            // Subsequence pair (2 * (g % dist), 2 * (g % dist) + 1), sample
            // chunk `g / dist`; blocks land in bit-reversed order, which for
            // two and four blocks is the pair order this network emits when
            // block `2s` and `2s + 1` sit at rows `s` and `s + BLOCKS / 2`.
            let s = g % dist;
            let k = g / dist;
            even.store_to_view_chunk(&mut dst, s * 64 + k);
            odd.store_to_view_chunk(&mut dst, (s + dist) * 64 + k);
        }
        true
    }
}
