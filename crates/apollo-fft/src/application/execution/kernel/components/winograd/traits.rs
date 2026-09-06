mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
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

    /// Whether the fused generated-codelet body should delegate its column
    /// phase to the `#[inline(never)]` split variant for this scalar. The
    /// fused body inlines every leaf copy into one monomorphization; the
    /// documented f32 LLVM codegen asymmetry (see `static_rader.rs`) shows
    /// that fusion exploding into spills exactly where f64 schedules it, so
    /// the split caps the register-pressure envelope. Defaults to `false`:
    /// each routed length must be measured before it is enabled.
    fn prefers_split_codelet(n: usize) -> bool {
        let _ = n;
        false
    }
}

thread_local! {
    #[expect(
        clippy::missing_const_for_thread_local,
        reason = "lazy initialization preserves the Rader benchmark path"
    )]
    static TL_WINOGRAD_SCRATCH_64: mnemosyne::scratch::ScratchPool<eunomia::Complex64> =
        mnemosyne::scratch::ScratchPool::new();
    #[expect(
        clippy::missing_const_for_thread_local,
        reason = "lazy initialization preserves the mixed-precision Rader path"
    )]
    static TL_WINOGRAD_SCRATCH_32: mnemosyne::scratch::ScratchPool<eunomia::Complex32> =
        mnemosyne::scratch::ScratchPool::new();
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
    fn prefers_split_codelet(n: usize) -> bool {
        // Measured, not assumed: `composite_split_ab_by_core_type` timed the
        // fused body against the split variant in one pinned run (same-run
        // A/B, disjoint medians). The split lost for f32 at n=50 (~3%) and
        // tied at n=144; the fused f32/f64 ratio there is 0.68, so the fused
        // body is not the f32 defect. Keep the fused body; the split
        // generators remain for the f64@144 perf-core lead recorded on the
        // board.
        let _ = n;
        false
    }
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
