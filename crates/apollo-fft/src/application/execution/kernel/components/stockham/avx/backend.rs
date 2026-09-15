/// The AVX register operations and fused stage kernels one Stockham precision
/// runs on.
///
/// # Safety
/// Every method is `unsafe` under one contract: the caller has established AVX
/// and FMA (the 512-bit backend, AVX-512F) on the running host, through the
/// compile-time features or `is_x86_feature_detected!`. The slice methods take
/// the stride-aligned source, destination and twiddle slices the Stockham stage
/// loop computed for the fused stage they implement.
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

    unsafe fn stage_pair_groups_two(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
    ) {
        let _ = (src, dst, radix, first_twiddles, second_twiddles);
        unreachable!("Not implemented for this precision");
    }

    unsafe fn stockham_quad_groups_eight(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
        fourth_twiddles: &[Self::Complex],
    ) {
        let _ = (
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
            fourth_twiddles,
        );
        unreachable!("Not implemented for this precision");
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
