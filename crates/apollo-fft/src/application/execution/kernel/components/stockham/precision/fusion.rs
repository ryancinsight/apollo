use super::traits::private;

pub(crate) struct StockhamFused1;
pub(crate) struct StockhamFused2;
pub(crate) struct StockhamFused3;
pub(crate) struct StockhamFused4;

impl private::Sealed for StockhamFused1 {}
impl private::Sealed for StockhamFused2 {}
impl private::Sealed for StockhamFused3 {}
impl private::Sealed for StockhamFused4 {}

/// Compile-time Stockham fusion-width policy.
///
/// The word `radix` has two distinct meanings in FFT code:
///
/// - factor radix: a DFT factor such as 3, 5, or 7 in a mixed-radix plan;
/// - fused radix-2 width: the number of adjacent radix-2 Stockham stages folded
///   into one autosort codelet.
///
/// This trait encodes the second meaning. `FUSED_RADIX2_WIDTH = 2^STAGE_COUNT`
/// is a codelet width, not a transform factorization choice. The runtime value
/// passed to `apply` is the current Stockham stage stride.
pub(crate) trait StockhamFusion: private::Sealed {
    const STAGE_COUNT: u32;
    const FUSED_RADIX2_WIDTH: usize;
    const TWIDDLE_STRIDE_FACTOR: usize;
}

impl StockhamFusion for StockhamFused1 {
    const STAGE_COUNT: u32 = 1;
    const FUSED_RADIX2_WIDTH: usize = 2;
    const TWIDDLE_STRIDE_FACTOR: usize = 1;
}

impl StockhamFusion for StockhamFused2 {
    const STAGE_COUNT: u32 = 2;
    const FUSED_RADIX2_WIDTH: usize = 4;
    const TWIDDLE_STRIDE_FACTOR: usize = 3;
}

impl StockhamFusion for StockhamFused3 {
    const STAGE_COUNT: u32 = 3;
    const FUSED_RADIX2_WIDTH: usize = 8;
    const TWIDDLE_STRIDE_FACTOR: usize = 7;
}

impl StockhamFusion for StockhamFused4 {
    const STAGE_COUNT: u32 = 4;
    const FUSED_RADIX2_WIDTH: usize = 16;
    const TWIDDLE_STRIDE_FACTOR: usize = 15;
}
#[inline]
pub(crate) fn fusion_fits<C: StockhamFusion>(stride: usize, n: usize) -> bool {
    stride <= n / C::FUSED_RADIX2_WIDTH
}

#[inline]
pub(crate) fn fusion_twiddle_len<C: StockhamFusion>(stride: usize) -> usize {
    stride * C::TWIDDLE_STRIDE_FACTOR
}

#[inline]
pub(crate) fn stockham_twiddle_table_len(n: usize) -> usize {
    debug_assert!(n >= 2);
    n - 1
}
