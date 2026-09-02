pub(crate) trait StockhamAvxBackend: Copy + Sized + 'static {
    type Real: Copy + Sized + 'static;
    type Complex: Copy + Sized + 'static;
    type Vector: Copy + Sized + 'static;

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
    unsafe fn permute_complex_swap(a: Self::Vector) -> Self::Vector;

    #[inline]
    unsafe fn cmul(wr: Self::Vector, wi: Self::Vector, b: Self::Vector) -> Self::Vector {
        let swapped = unsafe { Self::permute_complex_swap(b) };
        unsafe { Self::fmaddsub(wr, b, Self::mul(wi, swapped)) }
    }

    unsafe fn stage_groups_one(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        twiddles: &[Self::Complex],
    );

    unsafe fn stage_pair_groups_two(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
    );

    unsafe fn stage_pair_quarter_groups_two(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
    ) {
        let _ = (src, dst, radix, first_twiddles, second_twiddles);
        unreachable!("Not implemented for this precision");
    }

    unsafe fn stage_triple_quarter_groups_one(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
    );

    unsafe fn stage_triple_quarter_groups_two(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
    ) {
        let _ = (
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
        );
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
