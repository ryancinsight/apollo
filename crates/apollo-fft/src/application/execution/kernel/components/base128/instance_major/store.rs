//! Fixed-shape output strategies for the instance-major column pass.

use hermes_simd::{ComplexReg, LaneScalar, Simd, SimdArch, SimdKernel};

/// The split's combine butterfly, applied by phase 3 as it stores.
///
/// When present, the block being transformed is the odd half of a split
/// pair whose even half `peer` is already transformed: the register bound
/// for output index `j` instead produces `peer[j] + W^j reg` into `low[j]`
/// and `peer[j] - W^j reg` into `high[j]`. The separate combine pass — and
/// the store of this block's spectrum that it would have reloaded — cease
/// to exist (gap_audit.md#combine-sink).
pub(crate) struct CombineSink<'a, T> {
    /// The even block's transformed spectrum, one chunk per output index.
    pub(crate) peer: &'a [T; 256],
    /// Interleaved `W^j` twiddles from the split's cached table.
    pub(crate) tw: &'a [T; 256],
    /// The parent transform's low output half.
    pub(crate) low: &'a mut [T; 256],
    /// The parent transform's high output half.
    pub(crate) high: &'a mut [T; 256],
}

/// The four-block split's final butterfly, applied by block three as it stores.
///
/// `even_low` and `even_high` hold block one's already-combined pair. The
/// current register combines with `peer` through `inner_tw`, then each half
/// combines with its corresponding even value through the two halves of the
/// outer twiddle table. The first two output quarters replace the even
/// intermediates in place; the last two land in `high_low` and `high_high`.
pub(crate) struct FinalCombineSink<'a, T> {
    /// The transformed spectrum of block two.
    pub(crate) peer: &'a [T; 256],
    /// Interleaved inner twiddles for the block-two/block-three pair.
    pub(crate) inner_tw: &'a [T; 256],
    /// Block one's low pair result, replaced by output quarter zero.
    pub(crate) even_low: &'a mut [T; 256],
    /// Block one's high pair result, replaced by output quarter one.
    pub(crate) even_high: &'a mut [T; 256],
    /// Outer twiddles for output quarters zero and two.
    pub(crate) outer_low_tw: &'a [T; 256],
    /// Outer twiddles for output quarters one and three.
    pub(crate) outer_high_tw: &'a [T; 256],
    /// Output quarter two.
    pub(crate) high_low: &'a mut [T; 256],
    /// Output quarter three.
    pub(crate) high_high: &'a mut [T; 256],
}

pub(super) trait StoreSink<T: LaneScalar> {
    const DIRECT: bool;

    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
    );
}

pub(super) struct DirectSink;

impl<T: LaneScalar> StoreSink<T> for DirectSink {
    const DIRECT: bool = true;

    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        _simd: &Simd<T, A>,
        _reg: ComplexReg<T, A>,
        _chunk: usize,
    ) {
        unreachable!("invariant: direct stores use the base output view")
    }
}

impl<T: LaneScalar> StoreSink<T> for CombineSink<'_, T> {
    const DIRECT: bool = false;

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
    ) {
        let peer = simd.view(self.peer);
        let tw = simd.view(self.tw);
        let even = ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(
            &peer, chunk,
        ));
        let twiddle =
            ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(&tw, chunk));
        let (low, high) = even.butterfly(reg * twiddle);
        let mut low_view = simd.view_mut(&mut *self.low);
        low.into_interleaved()
            .store_to_view_chunk(&mut low_view, chunk);
        let mut high_view = simd.view_mut(&mut *self.high);
        high.into_interleaved()
            .store_to_view_chunk(&mut high_view, chunk);
    }
}

impl<T: LaneScalar> StoreSink<T> for FinalCombineSink<'_, T> {
    const DIRECT: bool = false;

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
    ) {
        let peer = simd.view(self.peer);
        let inner_tw = simd.view(self.inner_tw);
        let peer = ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(
            &peer, chunk,
        ));
        let inner_tw = ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(
            &inner_tw, chunk,
        ));
        let (odd_low, odd_high) = peer.butterfly(reg * inner_tw);

        let even_low = simd.view(&*self.even_low);
        let even_low = ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(
            &even_low, chunk,
        ));
        let even_high = simd.view(&*self.even_high);
        let even_high = ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(
            &even_high, chunk,
        ));
        let outer_low_tw = simd.view(self.outer_low_tw);
        let outer_low_tw = ComplexReg::<T, A>::from_interleaved(
            hermes_simd::Vector::from_view_chunk(&outer_low_tw, chunk),
        );
        let outer_high_tw = simd.view(self.outer_high_tw);
        let outer_high_tw = ComplexReg::<T, A>::from_interleaved(
            hermes_simd::Vector::from_view_chunk(&outer_high_tw, chunk),
        );
        let (out0, out2) = even_low.butterfly(odd_low * outer_low_tw);
        let (out1, out3) = even_high.butterfly(odd_high * outer_high_tw);

        let mut out0_view = simd.view_mut(&mut *self.even_low);
        out0.into_interleaved()
            .store_to_view_chunk(&mut out0_view, chunk);
        let mut out1_view = simd.view_mut(&mut *self.even_high);
        out1.into_interleaved()
            .store_to_view_chunk(&mut out1_view, chunk);
        let mut out2_view = simd.view_mut(&mut *self.high_low);
        out2.into_interleaved()
            .store_to_view_chunk(&mut out2_view, chunk);
        let mut out3_view = simd.view_mut(&mut *self.high_high);
        out3.into_interleaved()
            .store_to_view_chunk(&mut out3_view, chunk);
    }
}
