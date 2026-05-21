mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Scalar operations required by generic Winograd DFT helpers.
pub trait WinogradScalar:
    private::Sealed
    + super::radix::odd_prime_pair::PrimePairTables
    + num_traits::Float
    + num_traits::NumAssign
    + Send
    + Sync
    + 'static
{
    /// Convert an analytically defined constant to this scalar precision.
    fn from_precise(v: f64) -> Self;
    /// Return sqrt(2)/2 in this scalar precision.
    fn sq2o2() -> Self;
}
impl WinogradScalar for f64 {
    #[inline(always)]
    fn from_precise(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn sq2o2() -> Self {
        std::f64::consts::SQRT_2 / 2.0
    }
}
impl WinogradScalar for f32 {
    #[inline(always)]
    fn from_precise(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn sq2o2() -> Self {
        (std::f64::consts::SQRT_2 / 2.0) as f32
    }
}

#[inline(always)]
pub(crate) fn dft2_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>; 2]) {
    let a = data[0];
    let b = data[1];
    data[0] = a + b;
    data[1] = a - b;
}

/// Apply `W_N^{k·j}` twiddle multiplication in-place.
/// Used by the radix outer loop to apply inter-group twiddles.
#[inline(always)]
pub(crate) fn apply_twiddle_impl<F: WinogradScalar>(
    v: num_complex::Complex<F>,
    tw: num_complex::Complex<F>,
) -> num_complex::Complex<F> {
    num_complex::Complex::new(v.re * tw.re - v.im * tw.im, v.re * tw.im + v.im * tw.re)
}

// ─────────────────────────────────────────────────────────────────────────────
