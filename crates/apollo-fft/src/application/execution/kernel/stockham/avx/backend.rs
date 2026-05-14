pub(crate) trait StockhamAvxBackend: Copy + Sized + 'static {
    type Real: Copy + Sized + 'static;
    type Complex: Copy + Sized + 'static;
    type Vector: Copy + Sized + 'static;

    const COMPLEX_PER_VECTOR: usize;

    unsafe fn unpack_complex(c: Self::Complex) -> (Self::Real, Self::Real);
    unsafe fn complex_mul(a: Self::Complex, b: Self::Complex) -> Self::Complex;
    unsafe fn complex_add(a: Self::Complex, b: Self::Complex) -> Self::Complex;
    unsafe fn complex_sub(a: Self::Complex, b: Self::Complex) -> Self::Complex;

    unsafe fn set1_real(val: Self::Real) -> Self::Vector;
    unsafe fn set1_imag(val: Self::Real) -> Self::Vector;

    unsafe fn loadu_complex(ptr: *const Self::Complex) -> Self::Vector;
    unsafe fn storeu_complex(ptr: *mut Self::Complex, val: Self::Vector);

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
    unsafe fn permute_complex_swap(a: Self::Vector) -> Self::Vector;

    #[inline(always)]
    unsafe fn cmul(wr: Self::Vector, wi: Self::Vector, b: Self::Vector) -> Self::Vector {
        let swapped = unsafe { Self::permute_complex_swap(b) };
        unsafe { Self::fmaddsub(wr, b, Self::mul(wi, swapped)) }
    }
}
