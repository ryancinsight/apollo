pub(crate) mod private {
    pub trait Sealed {}
}

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
pub(crate) struct F64Stockham;
#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
pub(crate) struct F32Stockham;

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl private::Sealed for F64Stockham {}
#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl private::Sealed for F32Stockham {}

pub(crate) trait StockhamPrecision: private::Sealed {
    type Real: Copy;
    type Complex: Copy;

    const MAX_FUSED_STAGES: u32;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let _ = (stride, n, input_is_data);
        true
    }

    #[inline]
    fn stage_quad_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let _ = (stride, n, input_is_data);
        false
    }

    fn stage(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        twiddles: &[Self::Complex],
    );

    fn stage_pair(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
    );
    fn stage_triple(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
    );
    fn stage_quad(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
        fourth_twiddles: &[Self::Complex],
    );

    fn scale(data: &mut [Self::Complex], scale: Self::Real);
}

#[cfg(target_arch = "x86_64")]
pub(crate) trait StockhamRadix16AvxLeaf: StockhamPrecision {
    unsafe fn stage_quad_groups_eight_avx_fma(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
        fourth_twiddles: &[Self::Complex],
    );
}

pub(crate) struct StockhamTwiddleCursor<'a, T> {
    ptr: *const T,
    len: usize,
    consumed: usize,
    _lifetime: std::marker::PhantomData<&'a [T]>,
}

impl<'a, T> StockhamTwiddleCursor<'a, T> {
    #[inline]
    pub(crate) fn new(twiddles: &'a [T]) -> Self {
        Self {
            ptr: twiddles.as_ptr(),
            len: twiddles.len(),
            consumed: 0,
            _lifetime: std::marker::PhantomData,
        }
    }

    #[inline]
    pub(crate) unsafe fn take(&mut self, len: usize) -> &'a [T] {
        debug_assert!(self.consumed + len <= self.len);
        let start = self.consumed;
        self.consumed += len;
        unsafe { std::slice::from_raw_parts(self.ptr.add(start), len) }
    }

    #[inline]
    pub(crate) fn consumed(&self) -> usize {
        self.consumed
    }
}

#[inline]
pub(crate) unsafe fn stockham_twiddle_subslice<T>(
    twiddles: &[T],
    start: usize,
    len: usize,
) -> &[T] {
    debug_assert!(start + len <= twiddles.len());
    unsafe { std::slice::from_raw_parts(twiddles.as_ptr().add(start), len) }
}
