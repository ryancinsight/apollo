pub(crate) mod private {
    pub trait Sealed {
        const COMPACT_PAIR_MAX_HALF_LENGTH: usize;
    }

    impl Sealed for f32 {
        // Local release measurements show the compact kernel wins through
        // H=15; H=20 regresses, so larger transforms retain const
        // specialization.
        const COMPACT_PAIR_MAX_HALF_LENGTH: usize = 15;
    }

    impl Sealed for f64 {
        const COMPACT_PAIR_MAX_HALF_LENGTH: usize = 0;
    }
}

/// Scalar operations required by generic Winograd DFT helpers.
pub trait WinogradScalar:
    private::Sealed
    + super::radix::odd_prime_pair::PrimePairTables
    + eunomia::RealField
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
    + core::ops::DivAssign
    + Send
    + Sync
    + 'static
{
    /// Convert an analytically defined constant to this scalar precision.
    fn from_precise(v: f64) -> Self;
    /// Return sqrt(2)/2 in this scalar precision.
    fn sq2o2() -> Self;
    /// Runs a closure with a thread-local complex scratch buffer.
    fn with_winograd_scratch<R>(n: usize, f: impl FnOnce(&mut [eunomia::Complex<Self>]) -> R) -> R;
}

thread_local! {
    static TL_WINOGRAD_SCRATCH_64: mnemosyne::scratch::ScratchPool<eunomia::Complex64> = mnemosyne::scratch::ScratchPool::new();
    static TL_WINOGRAD_SCRATCH_32: mnemosyne::scratch::ScratchPool<eunomia::Complex32> = mnemosyne::scratch::ScratchPool::new();
}

impl WinogradScalar for f64 {
    #[inline]
    fn from_precise(v: f64) -> Self {
        v
    }
    #[inline]
    fn sq2o2() -> Self {
        std::f64::consts::SQRT_2 / 2.0
    }
    #[inline]
    fn with_winograd_scratch<R>(n: usize, f: impl FnOnce(&mut [eunomia::Complex<Self>]) -> R) -> R {
        TL_WINOGRAD_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }
}
impl WinogradScalar for f32 {
    #[inline]
    fn from_precise(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn sq2o2() -> Self {
        (std::f64::consts::SQRT_2 / 2.0) as f32
    }
    #[inline]
    fn with_winograd_scratch<R>(n: usize, f: impl FnOnce(&mut [eunomia::Complex<Self>]) -> R) -> R {
        TL_WINOGRAD_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }
}

// Canonical implementation lives in butterflies::dft (shared across GT/Rader/etc).
#[inline]
pub(crate) fn dft2_impl<F: WinogradScalar>(data: &mut [eunomia::Complex<F>; 2]) {
    crate::application::execution::kernel::components::butterflies::dft2_impl::<F>(data);
}

/// Apply `W_N^{k·j}` twiddle multiplication in-place.
/// Used by the radix outer loop to apply inter-group twiddles.
#[inline]
pub(crate) fn apply_twiddle_impl<F: WinogradScalar>(
    v: eunomia::Complex<F>,
    tw: eunomia::Complex<F>,
) -> eunomia::Complex<F> {
    eunomia::Complex::new(v.re * tw.re - v.im * tw.im, v.re * tw.im + v.im * tw.re)
}

// ─────────────────────────────────────────────────────────────────────────────
