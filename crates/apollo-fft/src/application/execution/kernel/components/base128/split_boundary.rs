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

/// Gathers the split's `BLOCKS` stride-`BLOCKS` subsequences into
/// contiguous blocks.
///
/// Four: one fused four-way pair deinterleave per quad of consecutive
/// parent chunks yields one chunk of each subsequence at any width, and
/// the network lands the four subsequences in exactly the bit-reversed
/// block order the radix-4 sink expects. Eight: the four-way deinterleave
/// of each half of eight consecutive chunks yields that half's stride-4
/// subsequences, and one pair deinterleave across the halves splits each
/// into its two stride-8 ones, landing in the natural order the radix-8
/// sink expects.
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
            (BLOCKS == 4 || BLOCKS == 8)
                && self.src.len() == BLOCKS * BLOCK_LANES
                && self.dst.len() == self.src.len(),
            "invariant: four or eight blocks of one base length"
        );
        // Chunks per block at the dispatched width.
        let cpb = BLOCK_LANES / lanes;
        if BLOCKS == 8 {
            for k in 0..cpb {
                let (s0, s1, s2, s3) = chunk::<T, A>(self.src, 8 * k).deinterleave_pairs4(
                    chunk(self.src, 8 * k + 1),
                    chunk(self.src, 8 * k + 2),
                    chunk(self.src, 8 * k + 3),
                );
                let (t0, t1, t2, t3) = chunk::<T, A>(self.src, 8 * k + 4).deinterleave_pairs4(
                    chunk(self.src, 8 * k + 5),
                    chunk(self.src, 8 * k + 6),
                    chunk(self.src, 8 * k + 7),
                );
                let (b0, b4) = s0.deinterleave_pairs(t0);
                let (b1, b5) = s1.deinterleave_pairs(t1);
                let (b2, b6) = s2.deinterleave_pairs(t2);
                let (b3, b7) = s3.deinterleave_pairs(t3);
                put_chunk(b0, self.dst, k);
                put_chunk(b1, self.dst, cpb + k);
                put_chunk(b2, self.dst, 2 * cpb + k);
                put_chunk(b3, self.dst, 3 * cpb + k);
                put_chunk(b4, self.dst, 4 * cpb + k);
                put_chunk(b5, self.dst, 5 * cpb + k);
                put_chunk(b6, self.dst, 6 * cpb + k);
                put_chunk(b7, self.dst, 7 * cpb + k);
            }
            return true;
        }
        // Four blocks store in the bit-reversed block order [0, 2, 1, 3].
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
        true
    }
}
