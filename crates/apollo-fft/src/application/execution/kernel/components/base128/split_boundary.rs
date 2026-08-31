//! Vectorized gather for the small-size split.
//!
//! The split's piece attribution put its non-base cost at 193 ns of 558 at
//! n = 256 and 641 of 1371 at n = 512. Two vectorization candidates came
//! out of that, and only one survived measurement
//! (gap_audit.md#split-boundary):
//!
//! - **The gather wins.** The scalar strided read costs 55/117 ns; the
//!   whole-register concatenation network below compiles to a six
//!   instruction loop — two loads, two `vperm2f128`, two stores — and
//!   measures 38 ns at two blocks.
//! - **The combine loses.** A planar-deinterleave combine kernel measured
//!   176 ns against the scalar loop's 96.5 in isolation: the scalar loop
//!   auto-vectorizes to about 3.4 cycles per butterfly already, and the
//!   planar form's deinterleave/reinterleave shuffles cost more than they
//!   save. That is the second independent confirmation of the original
//!   scalar-combine verdict, so the combine stays scalar and the fused
//!   radix-4 form in [`super`] attacks its pass count instead.
//!
//! Bounds are hoisted: one assert per slice at kernel entry, then raw chunk
//! access — these buffers arrive as runtime-length slices, and the checked
//! view accessor re-derives its bound per touch when the length is not a
//! compile-time constant (gap_audit.md#base128-bounds).

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};
/// One-vector load at chunk `c`, which the caller has proved in bounds.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope"
)]
#[inline(always)]
fn chunk<T, A>(data: &[T], c: usize) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!((c + 1) * <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: every caller asserts its slice length once at entry, and each
    // chunk index below is derived from that length; the kernel's dispatch
    // token proves the host executes `A`.
    unsafe { Vector::<T, A>::load_unaligned(data.as_ptr().add(c * 4)) }
}

/// One-vector store at chunk `c`, which the caller has proved in bounds.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope"
)]
#[inline(always)]
fn put_chunk<T, A>(v: Vector<T, A>, data: &mut [T], c: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!((c + 1) * <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: as `chunk` above.
    unsafe { v.store_unaligned(data.as_mut_ptr().add(c * 4)) }
}

/// Gathers the split's stride-`BLOCKS` subsequences into contiguous blocks.
///
/// A parent chunk holds two adjacent samples, so a subsequence chunk is a
/// whole-register concatenation of two parent chunks — the phase-one blend
/// network. `BLOCKS = 2` pairs neighbours; `BLOCKS = 4` pairs at distance
/// two, and the network lands the four subsequences in exactly the
/// bit-reversed block order the combine chain expects.
pub(crate) struct GatherBlocks<'a, T, const BLOCKS: usize> {
    pub(crate) src: &'a [T],
    pub(crate) dst: &'a mut [T],
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
        if <A as SimdStorage<T>>::LANE_COUNT != 4 {
            return false;
        }
        // One bound for the whole pass, so the per-chunk compares vanish.
        assert!(
            (BLOCKS == 2 || BLOCKS == 4)
                && self.src.len() == BLOCKS * 256
                && self.dst.len() == self.src.len(),
            "invariant: two or four 128-sample blocks"
        );
        // The high-complex blend mask, as in the base kernels' phase one.
        let zero = T::from_precise(0.0);
        let neg = T::from_precise(-1.0);
        let mask = [zero, zero, neg, neg];
        let mask = simd.view(&mask);
        let hi_mask = Vector::<T, A>::from_view_chunk(&mask, 0);
        // Per group: parent chunks at pair distance `BLOCKS / 2` concatenate
        // into one chunk of each of two subsequences.
        let dist = BLOCKS / 2;
        for g in 0..BLOCKS * 32 {
            let s = g % dist;
            let k = g / dist;
            let base = k * 2 * dist + s;
            let lo = chunk::<T, A>(self.src, base);
            let hi = chunk::<T, A>(self.src, base + dist);
            let even = hi_mask.blend(hi.swap_pairs(), lo);
            let odd = hi_mask.blend(hi, lo.swap_pairs());
            // Subsequences `2s` and `2s + 1` of the stride-`BLOCKS`
            // decimation; block row `s` and `s + dist` is the bit-reversed
            // block order (identity for two blocks, [0, 2, 1, 3] for four).
            put_chunk(even, self.dst, s * 64 + k);
            put_chunk(odd, self.dst, (s + dist) * 64 + k);
        }
        true
    }
}
