/// The AVX register operations and fused stage kernels one Stockham precision
/// runs on.
///
/// # Safety
/// Every method is `unsafe` under one contract: the caller has established AVX
/// and FMA (the 512-bit backend, AVX-512F) on the running host, through the
/// compile-time features or `is_x86_feature_detected!`. The slice methods take
/// the stride-aligned source, destination and twiddle slices the Stockham stage
/// loop computed for the fused stage they implement.
use super::super::precision::traits::StockhamPrecision;
use crate::application::execution::kernel::radix_stage::NormalizeSlice;
pub(crate) trait StockhamAvxBackend: Copy + Sized + 'static {
    type Real: Copy + Sized + 'static;
    type Complex: Copy + Sized + 'static;
    type Vector: Copy + Sized + 'static;

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
    unsafe fn permute_complex_swap(a: Self::Vector) -> Self::Vector;

    #[inline]
    unsafe fn cmul(wr: Self::Vector, wi: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: the trait's contract, AVX and FMA on the caller, is the whole of
        // each callee's.
        let swapped = unsafe { Self::permute_complex_swap(b) };
        // SAFETY: as above.
        unsafe { Self::fmaddsub(wr, b, Self::mul(wi, swapped)) }
    }

    unsafe fn stockham_quad_groups_eight_low_live(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
        fourth_twiddles: &[Self::Complex],
    );
}

/// Fused pair kernel available only on the 512-bit backends. Split from
/// [`StockhamAvxBackend`] so the capability is a type fact: the 256-bit
/// `f32`/`f64` backends cannot name this method, and no `unreachable!`
/// default decides at runtime what the type system already knows.
pub(crate) trait StockhamPairGroups: StockhamAvxBackend {
    unsafe fn stage_pair_groups_two(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
    );
}

/// Wide-lane dispatch policy: the double-width lanes, the sibling receiving
/// every shape they decline, and the measured group thresholds selecting
/// between them. Implemented once per 512-bit backend with concrete lane
/// widths — a generic over lane width as a const argument is rejected on
/// the pinned stable toolchain, so the width lives inside these small
/// forwarding methods instead of a const parameter.
///
/// The two implementors share one dispatch shape; only the tuning differs.
/// The AVX/FMA and scalar markers are intentionally excluded: their dispatch
/// trees differ in branch structure between precisions, so merging them
/// would parameterize branch existence rather than thresholds.
pub(crate) trait StockhamWideBackend: StockhamPairGroups
where
    Self::Complex: NormalizeSlice<Scale = Self::Real>,
{
    /// AVX/FMA sibling receiving every shape the wide lanes decline.
    type Sibling: StockhamPrecision<Real = Self::Real, Complex = Self::Complex>;
    /// `stage` delegates below this group count: 4 precise, 8 reduced.
    const STAGE_NARROW: usize;
    /// `stage_pair` radix-one delegates below this length: 8 precise, 16 reduced.
    const PAIR_RADIX_ONE_MIN: usize;
    /// `stage_pair` runs the 512-bit kernel at exactly this group count:
    /// 4 precise, 8 reduced (wider multiples stay on the sibling).
    const PAIR_WIDE_GROUPS: usize;
    /// `stage_pair` delegates below this group count: 8 precise, 16 reduced.
    const PAIR_NARROW: usize;
    /// `stage_triple` runs the wide lanes at or above this group count:
    /// 16 precise, 32 reduced.
    const TRIPLE_WIDE: usize;

    fn stage_one(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        twiddles: &[Self::Complex],
    ) -> bool;
    fn stage(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        twiddles: &[Self::Complex],
    ) -> bool;
    fn stage_pair_one(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        second_twiddles: &[Self::Complex],
    ) -> bool;
    fn stage_pair(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
    ) -> bool;
    fn stage_triple_one(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
    ) -> bool;
    fn stage_triple(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
    ) -> bool;
}
