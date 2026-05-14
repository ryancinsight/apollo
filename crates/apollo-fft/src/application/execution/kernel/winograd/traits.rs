mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Scalar operations required by generic Winograd DFT helpers.
pub trait WinogradScalar:
    private::Sealed + num_traits::Float + num_traits::NumAssign + Send + Sync
{
    /// Convert an analytically defined f64 constant to this scalar precision.
    fn cast_f64(v: f64) -> Self;
    /// Return sqrt(2)/2 in this scalar precision.
    fn sq2o2() -> Self;
}
impl WinogradScalar for f64 {
    #[inline]
    fn cast_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn sq2o2() -> Self {
        std::f64::consts::SQRT_2 / 2.0
    }
}
impl WinogradScalar for f32 {
    #[inline]
    fn cast_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn sq2o2() -> Self {
        (std::f64::consts::SQRT_2 / 2.0) as f32
    }
}

#[inline]
pub(crate) fn dft2_impl<F: WinogradScalar>(
    a: &mut num_complex::Complex<F>,
    b: &mut num_complex::Complex<F>,
) {
    let tmp = *a;
    *a = tmp + *b;
    *b = tmp - *b;
}

/// Apply `W_N^{k·j}` twiddle multiplication in-place.
/// Used by the radix outer loop to apply inter-group twiddles.
#[inline]
pub(crate) fn apply_twiddle_impl<F: WinogradScalar>(
    v: num_complex::Complex<F>,
    tw: num_complex::Complex<F>,
) -> num_complex::Complex<F> {
    num_complex::Complex::new(v.re * tw.re - v.im * tw.im, v.re * tw.im + v.im * tw.re)
}

// ─────────────────────────────────────────────────────────────────────────────
