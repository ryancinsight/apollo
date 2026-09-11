//! The base kernel's output strategies: where each column-pass register
//! lands.
//!
//! A sink holds only shared inputs — the peer spectrum, the twiddles, the
//! even halves — and writes into the kernel's output surface, which it
//! indexes as the block itself (direct), the block's pair (combine), or its
//! four (final combine). Holding no `&mut` of its own is what lets a block
//! read its samples out of the very surface its sink later writes: the
//! reads finish in the row phase, before the first store, so the split's
//! combining blocks load the parent directly and no gather pass exists.

use hermes_simd::{ComplexReg, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

/// Stores `v` at chunk `chunk` of `out`.
///
/// The kernel asserts `out.len() == OUT_BLOCKS * LANES` once at entry and
/// every chunk index a sink forms is `q * LANES / LANE_COUNT + j` with
/// `q < OUT_BLOCKS` and `j < LANES / LANE_COUNT`, so the per-store bound is
/// a debug assertion rather than a checked branch in the column loop.
#[expect(
    clippy::inline_always,
    reason = "the base kernel invokes this once per SIMD chunk"
)]
#[inline(always)]
fn put<T, A>(simd: &Simd<T, A>, v: Vector<T, A>, out: &mut [T], chunk: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let offset = chunk * lanes;
    debug_assert!(
        offset + lanes <= out.len(),
        "invariant: sink chunk in bounds"
    );
    let _ = simd;
    // SAFETY: `simd` proves the host supports `A`; the kernel's entry
    // assertion on `out.len()` and the chunk arithmetic above keep
    // `offset + LANE_COUNT <= out.len()`, so the store writes a full
    // vector in bounds.
    unsafe { v.store_unaligned(out.as_mut_ptr().add(offset)) }
}

/// Chunk `chunk` of a fixed-size input.
#[expect(
    clippy::inline_always,
    reason = "the base kernel invokes this once per SIMD chunk"
)]
#[inline(always)]
fn input<T, A, const LANES: usize>(
    simd: &Simd<T, A>,
    data: &[T; LANES],
    chunk: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    ComplexReg::from_interleaved(Vector::from_view_chunk(&simd.view(data), chunk))
}

/// Output strategy of the column pass.
pub(crate) trait StoreSink<T: LaneScalar> {
    /// Blocks of the base length the output surface spans.
    const OUT_BLOCKS: usize;

    /// Stores column register `reg`, which holds output chunk `chunk` of
    /// this block's spectrum, into `out`.
    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
        out: &mut [T],
    );
}

/// The spectrum lands in place: `out` is the block.
pub(crate) struct DirectSink;

impl<T: LaneScalar> StoreSink<T> for DirectSink {
    const OUT_BLOCKS: usize = 1;

    #[expect(
        clippy::inline_always,
        reason = "the base kernel invokes this concrete sink once per SIMD chunk"
    )]
    #[inline(always)]
    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
        out: &mut [T],
    ) {
        put(simd, reg.into_interleaved(), out, chunk);
    }
}

/// The odd block of a pair combines with the even block's spectrum on the
/// way out: `out` is the pair, low half `peer + W reg`, high half
/// `peer - W reg`.
pub(crate) struct CombineSink<'a, T, const LANES: usize> {
    /// The even block's spectrum.
    pub(crate) peer: &'a [T; LANES],
    /// `W_{2 BASE}^j` per chunk.
    pub(crate) tw: &'a [T; LANES],
}

impl<T: LaneScalar, const LANES: usize> StoreSink<T> for CombineSink<'_, T, LANES> {
    const OUT_BLOCKS: usize = 2;

    #[expect(
        clippy::inline_always,
        reason = "the base kernel invokes this concrete sink once per SIMD chunk"
    )]
    #[inline(always)]
    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
        out: &mut [T],
    ) {
        let block = LANES / <A as SimdStorage<T>>::LANE_COUNT;
        let even = input(simd, self.peer, chunk);
        let twiddle = input(simd, self.tw, chunk);
        let (low, high) = even.butterfly(reg * twiddle);
        put(simd, low.into_interleaved(), out, chunk);
        put(simd, high.into_interleaved(), out, block + chunk);
    }
}

/// Chunk `chunk` of `out` as a complex register.
#[expect(
    clippy::inline_always,
    reason = "the base kernel invokes this once per SIMD chunk"
)]
#[inline(always)]
fn take<T, A>(simd: &Simd<T, A>, out: &[T], chunk: usize) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let offset = chunk * lanes;
    debug_assert!(
        offset + lanes <= out.len(),
        "invariant: sink chunk in bounds"
    );
    let _ = simd;
    // SAFETY: as `put` — the kernel's entry assertion on `out.len()` and
    // the chunk arithmetic keep the load in bounds; `simd` proves the host.
    ComplexReg::from_interleaved(unsafe { Vector::load_unaligned(out.as_ptr().add(offset)) })
}

/// The last block of four: its pair butterfly against `peer`, then the
/// outer level against the even pair's halves, the four outputs stored to
/// the four quarters of `out` at `chunk`.
#[expect(
    clippy::inline_always,
    reason = "the base kernel invokes this once per SIMD chunk"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "one butterfly's operands; bundling them would be the sink itself"
)]
#[inline(always)]
fn final_combine<T, A, const LANES: usize>(
    simd: &Simd<T, A>,
    reg: ComplexReg<T, A>,
    chunk: usize,
    out: &mut [T],
    peer: &[T; LANES],
    inner_tw: &[T; LANES],
    even_low: ComplexReg<T, A>,
    even_high: ComplexReg<T, A>,
    outer_low_tw: &[T; LANES],
    outer_high_tw: &[T; LANES],
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let block = LANES / <A as SimdStorage<T>>::LANE_COUNT;
    let peer = input(simd, peer, chunk);
    let inner_tw = input(simd, inner_tw, chunk);
    let (odd_low, odd_high) = peer.butterfly(reg * inner_tw);
    let outer_low_tw = input(simd, outer_low_tw, chunk);
    let outer_high_tw = input(simd, outer_high_tw, chunk);
    let (out0, out2) = even_low.butterfly(odd_low * outer_low_tw);
    let (out1, out3) = even_high.butterfly(odd_high * outer_high_tw);
    put(simd, out0.into_interleaved(), out, chunk);
    put(simd, out1.into_interleaved(), out, block + chunk);
    put(simd, out2.into_interleaved(), out, 2 * block + chunk);
    put(simd, out3.into_interleaved(), out, 3 * block + chunk);
}

/// The last block of four applies its pair butterfly against `peer` and
/// the outer level against the even pair's halves, held apart from `out`,
/// as its registers leave the kernel: `out` is the four-block spectrum,
/// filled in one pass. The form for a block that reads `out` itself.
pub(crate) struct FinalCombineSink<'a, T, const LANES: usize> {
    /// The third block's spectrum.
    pub(crate) peer: &'a [T; LANES],
    /// `W_{2 BASE}^j` per chunk.
    pub(crate) inner_tw: &'a [T; LANES],
    /// The even pair's low half.
    pub(crate) even_low: &'a [T; LANES],
    /// The even pair's high half.
    pub(crate) even_high: &'a [T; LANES],
    /// `W_{4 BASE}^j` per chunk, `j < BASE`.
    pub(crate) outer_low_tw: &'a [T; LANES],
    /// `W_{4 BASE}^{j + BASE}` per chunk.
    pub(crate) outer_high_tw: &'a [T; LANES],
}

impl<T: LaneScalar, const LANES: usize> StoreSink<T> for FinalCombineSink<'_, T, LANES> {
    const OUT_BLOCKS: usize = 4;

    #[expect(
        clippy::inline_always,
        reason = "the base kernel invokes this concrete sink once per SIMD chunk"
    )]
    #[inline(always)]
    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
        out: &mut [T],
    ) {
        let even_low = input(simd, self.even_low, chunk);
        let even_high = input(simd, self.even_high, chunk);
        final_combine(
            simd,
            reg,
            chunk,
            out,
            self.peer,
            self.inner_tw,
            even_low,
            even_high,
            self.outer_low_tw,
            self.outer_high_tw,
        );
    }
}

/// [`FinalCombineSink`] with the even pair's halves already in the first
/// two quarters of `out`, read at `chunk` before the four outputs overwrite
/// them: the form for a block whose samples were gathered, so `out` holds
/// nothing it still needs.
pub(crate) struct FinalCombineInPlaceSink<'a, T, const LANES: usize> {
    /// The third block's spectrum.
    pub(crate) peer: &'a [T; LANES],
    /// `W_{2 BASE}^j` per chunk.
    pub(crate) inner_tw: &'a [T; LANES],
    /// `W_{4 BASE}^j` per chunk, `j < BASE`.
    pub(crate) outer_low_tw: &'a [T; LANES],
    /// `W_{4 BASE}^{j + BASE}` per chunk.
    pub(crate) outer_high_tw: &'a [T; LANES],
}

impl<T: LaneScalar, const LANES: usize> StoreSink<T> for FinalCombineInPlaceSink<'_, T, LANES> {
    const OUT_BLOCKS: usize = 4;

    #[expect(
        clippy::inline_always,
        reason = "the base kernel invokes this concrete sink once per SIMD chunk"
    )]
    #[inline(always)]
    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
        out: &mut [T],
    ) {
        let block = LANES / <A as SimdStorage<T>>::LANE_COUNT;
        let even_low = take(simd, out, chunk);
        let even_high = take(simd, out, block + chunk);
        final_combine(
            simd,
            reg,
            chunk,
            out,
            self.peer,
            self.inner_tw,
            even_low,
            even_high,
            self.outer_low_tw,
            self.outer_high_tw,
        );
    }
}
