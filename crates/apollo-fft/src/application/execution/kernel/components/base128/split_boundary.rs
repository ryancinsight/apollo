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

use super::instance_major::BlockSource;
use crate::application::execution::kernel::components::register_butterfly::{
    radix3, radix4, radix8, DupSplitEighths, Thirds,
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

/// Reads a contiguous column register at the dispatched complex width.
#[expect(
    clippy::inline_always,
    reason = "preserve the dispatcher's target-feature scope"
)]
#[inline(always)]
fn column_input<T, A, S>(simd: Simd<T, A>, source: &S, out: &[T], c: usize) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    S: BlockSource<T>,
{
    if <A as SimdStorage<T>>::LANE_COUNT == 8 {
        source.chunk::<A, 4>(simd, out, c)
    } else {
        source.chunk::<A, 2>(simd, out, c)
    }
}
/// `v` times twiddle `j` of chunk `c` in `twiddles`, `per_chunk` twiddles a
/// chunk: split (`SPLIT`), its duplicated real register and then its
/// `(-im, im)` register, one swap of `v` and no shuffle of the twiddle;
/// otherwise one interleaved register.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope"
)]
#[inline(always)]
fn twiddled<T, A, const SPLIT: bool>(
    v: ComplexReg<T, A>,
    twiddles: &[T],
    per_chunk: usize,
    c: usize,
    j: usize,
) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    if SPLIT {
        let at = 2 * (per_chunk * c + j);
        let v = v.into_interleaved();
        v.mul_add(
            chunk(twiddles, at),
            v.swap_adjacent() * chunk(twiddles, at + 1),
        )
    } else {
        let w = ComplexReg::<T, A>::from_interleaved(chunk(twiddles, per_chunk * c + j));
        (v * w).into_interleaved()
    }
}

/// The radix-`RADIX` pass over `RADIX` contiguous parent slices, in place or
/// into scratch: RustFFT's column pass ahead of its inner butterflies. For
/// input `x[c + BASE r]`, `r < RADIX`, chunk `c` loads the slices' registers
/// (`RADIX` streams `BASE` samples apart), runs the register radix across
/// them, and stores `Y_0` and `W_{RADIX BASE}^{q c} Y_q` into the
/// corresponding output slice — the twiddle after the butterfly, from one
/// contiguous chunk-major stream ([`super::instance_major::SplitSinks::rows`])
/// — so slice `q` then transforms as a contiguous `BASE`-point block whose
/// spectrum is `X[RADIX k + q]`. Radices 3, 4 and 8; `SPLIT` selects the
/// twiddle layout (`instance_major::TwiddleLayout`).
pub(crate) struct ColumnPass<'a, T, S, const RADIX: usize, const INVERSE: bool, const SPLIT: bool> {
    /// The parent, `RADIX` blocks of `block_lanes` lanes.
    pub(crate) source: S,
    /// `RADIX` contiguous output blocks.
    pub(crate) dst: &'a mut [T],
    /// `(RADIX - 1) block_lanes` lanes a twiddle register, chunk-major.
    pub(crate) twiddles: &'a [T],
    /// One block in scalar lanes: a runtime value, so one pass serves every
    /// level of a chain (the bound is asserted once at entry either way).
    pub(crate) block_lanes: usize,
    /// `W_8^1` as `(re, im)` for the eighths; read at radix 8 only.
    pub(crate) eighth_one: [T; 2],
    /// `W_8^3` as `(re, im)` for the eighths; read at radix 8 only.
    pub(crate) eighth_three: [T; 2],
}

impl<
        T: LaneScalar + MixedRadixScalar,
        S: BlockSource<T>,
        const RADIX: usize,
        const INVERSE: bool,
        const SPLIT: bool,
    > LaneKernel<T> for ColumnPass<'_, T, S, RADIX, INVERSE, SPLIT>
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
            (RADIX == 3 || RADIX == 4 || RADIX == 8)
                && S::BLOCKS == 1
                && self.block_lanes % lanes == 0
                && self.source.parent_lanes(self.dst) == RADIX * self.block_lanes
                && self.dst.len() == RADIX * self.block_lanes
                && self.twiddles.len()
                    == if SPLIT { 2 } else { 1 } * (RADIX - 1) * self.block_lanes,
            "invariant: the radix's slices and twiddle rows of one block length"
        );
        let cpb = self.block_lanes / lanes;
        let source = self.source;
        let out = self.dst;
        if RADIX == 3 {
            let thirds = Thirds::new(simd);
            for c in 0..cpb {
                let y = radix3::<T, A, INVERSE>(
                    [
                        ComplexReg::from_interleaved(column_input(simd, &source, out, c)),
                        ComplexReg::from_interleaved(column_input(simd, &source, out, cpb + c)),
                        ComplexReg::from_interleaved(column_input(simd, &source, out, 2 * cpb + c)),
                    ],
                    &thirds,
                );
                put_chunk(y[0].into_interleaved(), out, c);
                put_chunk(
                    twiddled::<T, A, SPLIT>(y[1], self.twiddles, RADIX - 1, c, 0),
                    out,
                    cpb + c,
                );
                put_chunk(
                    twiddled::<T, A, SPLIT>(y[2], self.twiddles, RADIX - 1, c, 1),
                    out,
                    2 * cpb + c,
                );
            }
            return true;
        }
        if RADIX == 4 {
            for c in 0..cpb {
                let y = radix4::<T, A, INVERSE>([
                    ComplexReg::from_interleaved(column_input(simd, &source, out, c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, cpb + c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, 2 * cpb + c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, 3 * cpb + c)),
                ]);
                put_chunk(y[0].into_interleaved(), out, c);
                put_chunk(
                    twiddled::<T, A, SPLIT>(y[1], self.twiddles, RADIX - 1, c, 0),
                    out,
                    cpb + c,
                );
                put_chunk(
                    twiddled::<T, A, SPLIT>(y[2], self.twiddles, RADIX - 1, c, 1),
                    out,
                    2 * cpb + c,
                );
                put_chunk(
                    twiddled::<T, A, SPLIT>(y[3], self.twiddles, RADIX - 1, c, 2),
                    out,
                    3 * cpb + c,
                );
            }
            return true;
        }
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
        for c in 0..cpb {
            let y = radix8::<T, A, INVERSE, _>(
                [
                    ComplexReg::from_interleaved(column_input(simd, &source, out, c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, cpb + c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, 2 * cpb + c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, 3 * cpb + c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, 4 * cpb + c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, 5 * cpb + c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, 6 * cpb + c)),
                    ComplexReg::from_interleaved(column_input(simd, &source, out, 7 * cpb + c)),
                ],
                &eighths,
            );
            // The seven twiddles of chunk `c`, two registers each.
            put_chunk(y[0].into_interleaved(), out, c);
            put_chunk(
                twiddled::<T, A, SPLIT>(y[1], self.twiddles, RADIX - 1, c, 0),
                out,
                cpb + c,
            );
            put_chunk(
                twiddled::<T, A, SPLIT>(y[2], self.twiddles, RADIX - 1, c, 1),
                out,
                2 * cpb + c,
            );
            put_chunk(
                twiddled::<T, A, SPLIT>(y[3], self.twiddles, RADIX - 1, c, 2),
                out,
                3 * cpb + c,
            );
            put_chunk(
                twiddled::<T, A, SPLIT>(y[4], self.twiddles, RADIX - 1, c, 3),
                out,
                4 * cpb + c,
            );
            put_chunk(
                twiddled::<T, A, SPLIT>(y[5], self.twiddles, RADIX - 1, c, 4),
                out,
                5 * cpb + c,
            );
            put_chunk(
                twiddled::<T, A, SPLIT>(y[6], self.twiddles, RADIX - 1, c, 5),
                out,
                6 * cpb + c,
            );
            put_chunk(
                twiddled::<T, A, SPLIT>(y[7], self.twiddles, RADIX - 1, c, 6),
                out,
                7 * cpb + c,
            );
        }
        true
    }
}

/// Interleaves `BLOCKS` transformed slices into natural order: block `q` of
/// `src` holds `X[BLOCKS k + q]` at `k`, and `dst` receives `X` in order —
/// RustFFT's transpose after its inner butterflies, run as pair
/// interleaves in registers. Chunk `k` of every block loads (`BLOCKS`
/// streams `BASE` samples apart) and the `BLOCKS` registers of consecutive
/// output store contiguously: eight blocks at four complexes a register are
/// two fused four-way tile transposes and at two a pairing of even and odd
/// blocks, four blocks one tile transpose and one pairing; three blocks
/// are one three-way pair interleave at either width.
pub(crate) struct InterleaveBlocks<'a, T, const BLOCKS: usize> {
    /// `BLOCKS` transformed blocks, `BLOCKS block_lanes` lanes.
    pub(crate) src: &'a [T],
    /// The parent, `BLOCKS block_lanes` lanes.
    pub(crate) dst: &'a mut [T],
    /// One block in scalar lanes (as [`ColumnPass::block_lanes`]).
    pub(crate) block_lanes: usize,
}

impl<T: LaneScalar + MixedRadixScalar, const BLOCKS: usize> LaneKernel<T>
    for InterleaveBlocks<'_, T, BLOCKS>
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
            (BLOCKS == 3 || BLOCKS == 4 || BLOCKS == 8)
                && self.block_lanes % lanes == 0
                && self.src.len() == BLOCKS * self.block_lanes
                && self.dst.len() == self.src.len(),
            "invariant: the blocks of one block length"
        );
        let cpb = self.block_lanes / lanes;
        if BLOCKS == 3 {
            for k in 0..cpb {
                let r0 = chunk::<T, A>(self.src, k);
                let r1 = chunk::<T, A>(self.src, cpb + k);
                let r2 = chunk::<T, A>(self.src, 2 * cpb + k);
                let (o0, o1, o2) = r0.interleave_pairs3(r1, r2);
                put_chunk(o0, self.dst, 3 * k);
                put_chunk(o1, self.dst, 3 * k + 1);
                put_chunk(o2, self.dst, 3 * k + 2);
            }
            return true;
        }
        if BLOCKS == 4 {
            for k in 0..cpb {
                let r0 = chunk::<T, A>(self.src, k);
                let r1 = chunk::<T, A>(self.src, cpb + k);
                let r2 = chunk::<T, A>(self.src, 2 * cpb + k);
                let r3 = chunk::<T, A>(self.src, 3 * cpb + k);
                let (o0, o1, o2, o3) = if lanes == 8 {
                    // The four registers are one 4-by-4 complex tile.
                    r0.deinterleave_pairs4(r1, r2, r3)
                } else {
                    // Output register `4 k + 2 s + p` holds blocks `2 p` and
                    // `2 p + 1` at sample `2 k + s`.
                    let (o0, o2) = r0.interleave_pairs(r1);
                    let (o1, o3) = r2.interleave_pairs(r3);
                    (o0, o1, o2, o3)
                };
                put_chunk(o0, self.dst, 4 * k);
                put_chunk(o1, self.dst, 4 * k + 1);
                put_chunk(o2, self.dst, 4 * k + 2);
                put_chunk(o3, self.dst, 4 * k + 3);
            }
            return true;
        }
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
                // Each four-register tile is a 4-by-4 complex transpose.
                // The fused primitive avoids repeating cross-half shuffles.
                let (o0, o2, o4, o6) = r0.deinterleave_pairs4(r1, r2, r3);
                let (o1, o3, o5, o7) = r4.deinterleave_pairs4(r5, r6, r7);
                put_chunk(o0, self.dst, 8 * k);
                put_chunk(o1, self.dst, 8 * k + 1);
                put_chunk(o2, self.dst, 8 * k + 2);
                put_chunk(o3, self.dst, 8 * k + 3);
                put_chunk(o4, self.dst, 8 * k + 4);
                put_chunk(o5, self.dst, 8 * k + 5);
                put_chunk(o6, self.dst, 8 * k + 6);
                put_chunk(o7, self.dst, 8 * k + 7);
            } else {
                // Output register `8 k + 4 s + p` holds blocks `2 p` and `2 p + 1`
                // at sample `2 k + s`, where `s` is zero or one.
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
