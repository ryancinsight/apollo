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
//! The blend network is expressed through hermes' pair-granularity
//! deinterleave, so the four-lane f64 route and the eight-lane f32 route run
//! the same construction at their native widths -- a four-byte scalar
//! previously ran the four-lane form in the scalar-emulated frame. One
//! `deinterleave_pairs` of two consecutive chunks splits their complex
//! samples into even and odd halves; four blocks take one fused four-way
//! pair deinterleave per quad of consecutive chunks.
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
    unsafe {
        Vector::<T, A>::load_unaligned(data.as_ptr().add(c * <A as SimdStorage<T>>::LANE_COUNT))
    }
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
    unsafe { v.store_unaligned(data.as_mut_ptr().add(c * <A as SimdStorage<T>>::LANE_COUNT)) }
}

/// Gathers the split's stride-`BLOCKS` subsequences into contiguous blocks.
///
/// A parent chunk holds two adjacent samples, so a subsequence chunk is a
/// whole-register concatenation of two parent chunks — the phase-one blend
/// network. `BLOCKS = 2` pairs neighbours; `BLOCKS = 4` pairs at distance
/// two, and the network lands the four subsequences in exactly the
/// bit-reversed block order the combine chain expects.
pub(crate) struct GatherBlocks<'a, T, const BLOCKS: usize, const BLOCK_LANES: usize> {
    pub(crate) src: &'a [T],
    pub(crate) dst: &'a mut [T],
}

impl<T: LaneScalar + MixedRadixScalar, const BLOCKS: usize, const BLOCK_LANES: usize> LaneKernel<T>
    for GatherBlocks<'_, T, BLOCKS, BLOCK_LANES>
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
        let _ = simd;
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        if lanes != 4 && lanes != 8 {
            return false;
        }
        // One bound for the whole pass, so the per-chunk compares vanish.
        assert!(
            (BLOCKS == 2 || BLOCKS == 4 || BLOCKS == 8)
                && self.src.len() == BLOCKS * BLOCK_LANES
                && self.dst.len() == self.src.len(),
            "invariant: two, four or eight blocks of one base length"
        );
        // Chunks per 128-sample block at the dispatched width.
        let cpb = BLOCK_LANES / lanes;
        if BLOCKS == 2 {
            // One pair-deinterleave of two consecutive chunks splits their
            // complex samples into the even and odd subsequences.
            for g in 0..cpb {
                let (even, odd) =
                    chunk::<T, A>(self.src, 2 * g).deinterleave_pairs(chunk(self.src, 2 * g + 1));
                put_chunk(even, self.dst, g);
                put_chunk(odd, self.dst, cpb + g);
            }
        } else if BLOCKS == 4 {
            // Four blocks: one fused four-way pair deinterleave per quad of
            // consecutive chunks yields one chunk of each stride-4
            // subsequence at any width, and the outputs store in the
            // bit-reversed block order [0, 2, 1, 3] the combine chain
            // expects.
            for k in 0..cpb {
                let (b0, b1, b2, b3) = chunk::<T, A>(self.src, 4 * k).deinterleave_pairs4(
                    chunk(self.src, 4 * k + 1),
                    chunk(self.src, 4 * k + 2),
                    chunk(self.src, 4 * k + 3),
                );
                put_chunk(b0, self.dst, k);
                put_chunk(b2, self.dst, cpb + k);
                put_chunk(b1, self.dst, 2 * cpb + k);
                put_chunk(b3, self.dst, 3 * cpb + k);
            }
        } else {
            // Eight blocks: hermes' eight-way pair split yields one chunk of
            // every stride-8 subsequence per octet of consecutive chunks, and
            // the outputs store in the bit-reversed block order
            // [0, 4, 2, 6, 1, 5, 3, 7] — the three-bit reversal the combine
            // chain expects, as [0, 2, 1, 3] is the two-bit one above.
            for k in 0..cpb {
                let blocks = chunk::<T, A>(self.src, 8 * k).deinterleave_pairs8(
                    chunk(self.src, 8 * k + 1),
                    chunk(self.src, 8 * k + 2),
                    chunk(self.src, 8 * k + 3),
                    chunk(self.src, 8 * k + 4),
                    chunk(self.src, 8 * k + 5),
                    chunk(self.src, 8 * k + 6),
                    chunk(self.src, 8 * k + 7),
                );
                for (position, subsequence) in [0usize, 4, 2, 6, 1, 5, 3, 7].into_iter().enumerate()
                {
                    put_chunk(blocks[subsequence], self.dst, position * cpb + k);
                }
            }
        }
        true
    }
}

/// One radix-2 combine level over pairs of `len`-sample blocks, in place:
/// `low[j] + W^j high[j]` and `low[j] - W^j high[j]` for every pair.
///
/// The interleaved complex multiply is the dup-split form both base kernels
/// use: the twiddle register duplicated into its real and imaginary lanes,
/// one adjacent swap, one multiply and one alternating fused multiply-add.
/// The scalar loop in the parent module is the reference form and the pass
/// for a width that does not divide the block.
pub(crate) struct CombineLevel<'a, T> {
    /// The whole array as interleaved reals: pairs of two `len`-sample blocks.
    pub(crate) data: &'a mut [T],
    /// `W^j` for `j < len`, interleaved reals.
    pub(crate) twiddles: &'a [T],
    /// Block length in complex samples.
    pub(crate) len: usize,
}

impl<T: LaneScalar + MixedRadixScalar> LaneKernel<T> for CombineLevel<'_, T> {
    /// Whether the dispatched width handled the pass.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature \
                  frame (hermes LaneKernel contract)"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        let _ = simd;
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        let reals = 2 * self.len;
        if lanes < 2 || reals % lanes != 0 {
            return false;
        }
        assert!(
            self.data.len() % (2 * reals) == 0 && self.twiddles.len() >= reals,
            "invariant: whole block pairs and one twiddle per block sample"
        );
        let chunks = reals / lanes;
        for pair in 0..self.data.len() / (2 * reals) {
            let low = pair * 2 * chunks;
            let high = low + chunks;
            for c in 0..chunks {
                let w = chunk::<T, A>(self.twiddles, c);
                let (wr, wi) = (w.dup_even(), w.dup_odd());
                let h = chunk::<T, A>(self.data, high + c);
                let rotated = h.fmaddsub(wr, h.swap_adjacent() * wi);
                let l = chunk::<T, A>(self.data, low + c);
                put_chunk(l + rotated, self.data, low + c);
                put_chunk(l - rotated, self.data, high + c);
            }
        }
        true
    }
}
