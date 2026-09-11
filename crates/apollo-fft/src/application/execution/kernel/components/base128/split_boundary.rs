//! The split's boundary passes over the parent: the gather of four blocks,
//! and the radix-8 pass and interleave that bracket eight.
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

use crate::application::execution::kernel::components::register_butterfly::{
    radix8, DupSplitEighths,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{
    ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector,
};
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

/// Gathers the split's four stride-4 subsequences into contiguous blocks.
///
/// One fused four-way pair deinterleave per quad of consecutive parent
/// chunks yields one chunk of each subsequence at any width, and the
/// network lands the four subsequences in exactly the bit-reversed block
/// order the radix-4 sink expects.
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
            BLOCKS == 4
                && self.src.len() == BLOCKS * BLOCK_LANES
                && self.dst.len() == self.src.len(),
            "invariant: four blocks of one base length"
        );
        // Chunks per block at the dispatched width; the outputs store in
        // the bit-reversed block order [0, 2, 1, 3].
        let cpb = BLOCK_LANES / lanes;
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

/// The radix-8 pass over a parent of eight contiguous eighths, in place:
/// RustFFT's column pass ahead of its inner butterflies. For input
/// `x[c + BASE r]`, `r < 8`, chunk `c` loads the eight eighths' registers
/// (eight streams `BASE` samples apart), runs the register radix-8 across
/// them, and stores `Y_0` and `W_{8 BASE}^{q c} Y_q` back where they came
/// from — the twiddle after the butterfly, from one contiguous chunk-major
/// stream ([`super::instance_major::SplitSinks::rows`]) — so eighth `q`
/// then transforms as a contiguous `BASE`-point block whose spectrum is
/// `X[8 k + q]`.
pub(crate) struct ColumnRadix8<'a, T, const BLOCK_LANES: usize, const INVERSE: bool> {
    /// The parent, `8 BLOCK_LANES` lanes.
    pub(crate) data: &'a mut [T],
    /// `7 BLOCK_LANES` lanes, chunk-major.
    pub(crate) twiddles: &'a [T],
    /// `W_8^1` as `(re, im)` for the eighths.
    pub(crate) eighth_one: [T; 2],
    /// `W_8^3` as `(re, im)` for the eighths.
    pub(crate) eighth_three: [T; 2],
}

impl<T: LaneScalar + MixedRadixScalar, const BLOCK_LANES: usize, const INVERSE: bool> LaneKernel<T>
    for ColumnRadix8<'_, T, BLOCK_LANES, INVERSE>
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
        if lanes != 4 && lanes != 8 {
            return false;
        }
        // One bound for the whole pass, so the per-chunk compares vanish.
        assert!(
            self.data.len() == 8 * BLOCK_LANES && self.twiddles.len() == 7 * BLOCK_LANES,
            "invariant: eight eighths and seven twiddle rows of one base length"
        );
        let cpb = BLOCK_LANES / lanes;
        let eighths = DupSplitEighths {
            one: (
                simd.splat(self.eighth_one[0]),
                simd.splat(self.eighth_one[1]),
            ),
            three: (
                simd.splat(self.eighth_three[0]),
                simd.splat(self.eighth_three[1]),
            ),
        };
        let data = self.data;
        for c in 0..cpb {
            let y = radix8::<T, A, INVERSE, _>(
                [
                    ComplexReg::from_interleaved(chunk(data, c)),
                    ComplexReg::from_interleaved(chunk(data, cpb + c)),
                    ComplexReg::from_interleaved(chunk(data, 2 * cpb + c)),
                    ComplexReg::from_interleaved(chunk(data, 3 * cpb + c)),
                    ComplexReg::from_interleaved(chunk(data, 4 * cpb + c)),
                    ComplexReg::from_interleaved(chunk(data, 5 * cpb + c)),
                    ComplexReg::from_interleaved(chunk(data, 6 * cpb + c)),
                    ComplexReg::from_interleaved(chunk(data, 7 * cpb + c)),
                ],
                &eighths,
            );
            // The seven twiddle registers of chunk `c`, in the stream's order.
            let t1 = ComplexReg::<T, A>::from_interleaved(chunk(self.twiddles, 7 * c));
            let t2 = ComplexReg::<T, A>::from_interleaved(chunk(self.twiddles, 7 * c + 1));
            let t3 = ComplexReg::<T, A>::from_interleaved(chunk(self.twiddles, 7 * c + 2));
            let t4 = ComplexReg::<T, A>::from_interleaved(chunk(self.twiddles, 7 * c + 3));
            let t5 = ComplexReg::<T, A>::from_interleaved(chunk(self.twiddles, 7 * c + 4));
            let t6 = ComplexReg::<T, A>::from_interleaved(chunk(self.twiddles, 7 * c + 5));
            let t7 = ComplexReg::<T, A>::from_interleaved(chunk(self.twiddles, 7 * c + 6));
            put_chunk(y[0].into_interleaved(), data, c);
            put_chunk((y[1] * t1).into_interleaved(), data, cpb + c);
            put_chunk((y[2] * t2).into_interleaved(), data, 2 * cpb + c);
            put_chunk((y[3] * t3).into_interleaved(), data, 3 * cpb + c);
            put_chunk((y[4] * t4).into_interleaved(), data, 4 * cpb + c);
            put_chunk((y[5] * t5).into_interleaved(), data, 5 * cpb + c);
            put_chunk((y[6] * t6).into_interleaved(), data, 6 * cpb + c);
            put_chunk((y[7] * t7).into_interleaved(), data, 7 * cpb + c);
        }
        true
    }
}

/// Interleaves eight transformed eighths into natural order: block `q` of
/// `src` holds `X[8 k + q]` at `k`, and `dst` receives `X` in order —
/// RustFFT's transpose after its inner butterflies, run as pair
/// interleaves in registers. Chunk `k` of every block loads (eight streams
/// `BASE` samples apart) and the eight registers of consecutive output
/// store contiguously: at four complexes a register two levels of pair
/// interleave transpose each four-block tile, at two one level pairs the
/// even and odd blocks.
pub(crate) struct InterleaveBlocks<'a, T, const BLOCK_LANES: usize> {
    /// Eight transformed blocks, `8 BLOCK_LANES` lanes.
    pub(crate) src: &'a [T],
    /// The parent, `8 BLOCK_LANES` lanes.
    pub(crate) dst: &'a mut [T],
}

impl<T: LaneScalar + MixedRadixScalar, const BLOCK_LANES: usize> LaneKernel<T>
    for InterleaveBlocks<'_, T, BLOCK_LANES>
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
            self.src.len() == 8 * BLOCK_LANES && self.dst.len() == self.src.len(),
            "invariant: eight blocks of one base length"
        );
        let cpb = BLOCK_LANES / lanes;
        for k in 0..cpb {
            let r0 = chunk::<T, A>(self.src, k);
            let r1 = chunk::<T, A>(self.src, cpb + k);
            let r2 = chunk::<T, A>(self.src, 2 * cpb + k);
            let r3 = chunk::<T, A>(self.src, 3 * cpb + k);
            let r4 = chunk::<T, A>(self.src, 4 * cpb + k);
            let r5 = chunk::<T, A>(self.src, 5 * cpb + k);
            let r6 = chunk::<T, A>(self.src, 6 * cpb + k);
            let r7 = chunk::<T, A>(self.src, 7 * cpb + k);
            if lanes == 8 {
                // Output register `2 k` holds blocks 0 to 3 at `k`, `2 k + 1`
                // blocks 4 to 7: the transpose of each four-block tile.
                let (a0, a1) = r0.interleave_pairs(r2);
                let (b0, b1) = r1.interleave_pairs(r3);
                let (o0, o2) = a0.interleave_pairs(b0);
                let (o4, o6) = a1.interleave_pairs(b1);
                let (c0, c1) = r4.interleave_pairs(r6);
                let (d0, d1) = r5.interleave_pairs(r7);
                let (o1, o3) = c0.interleave_pairs(d0);
                let (o5, o7) = c1.interleave_pairs(d1);
                put_chunk(o0, self.dst, 8 * k);
                put_chunk(o1, self.dst, 8 * k + 1);
                put_chunk(o2, self.dst, 8 * k + 2);
                put_chunk(o3, self.dst, 8 * k + 3);
                put_chunk(o4, self.dst, 8 * k + 4);
                put_chunk(o5, self.dst, 8 * k + 5);
                put_chunk(o6, self.dst, 8 * k + 6);
                put_chunk(o7, self.dst, 8 * k + 7);
            } else {
                // Output register `4 k + p` holds blocks `2 p` and `2 p + 1`
                // at `k`; chunk `k` of a block holds `k` and `k + 1`.
                let (o0, o4) = r0.interleave_pairs(r1);
                let (o1, o5) = r2.interleave_pairs(r3);
                let (o2, o6) = r4.interleave_pairs(r5);
                let (o3, o7) = r6.interleave_pairs(r7);
                put_chunk(o0, self.dst, 8 * k);
                put_chunk(o1, self.dst, 8 * k + 1);
                put_chunk(o2, self.dst, 8 * k + 2);
                put_chunk(o3, self.dst, 8 * k + 3);
                put_chunk(o4, self.dst, 8 * k + 4);
                put_chunk(o5, self.dst, 8 * k + 5);
                put_chunk(o6, self.dst, 8 * k + 6);
                put_chunk(o7, self.dst, 8 * k + 7);
            }
        }
        true
    }
}
