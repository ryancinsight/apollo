//! Plan-owned FFT workspace allocation helpers.

use num_complex::{Complex32, Complex64};

mod sealed {
    pub trait Sealed {}

    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

// ── Thread-local plan scratch buffers ───────────────────────────────────────
//
// The 2D and 3D plans previously stored `Mutex<Vec<F::Complex>>` fields that
// serialized every axis-pass call on a mutex even for single-threaded use.
// Replaced by per-precision `thread_local!` buffers that grow on demand and
// are reused across calls, matching the kernel scratch pattern in
// `mixed_radix/caches/scratch.rs`.

thread_local! {
    static TL_2D_SCRATCH_64: mnemosyne::scratch::ScratchPool<Complex64> = mnemosyne::scratch::ScratchPool::new();
    static TL_2D_SCRATCH_32: mnemosyne::scratch::ScratchPool<Complex32> = mnemosyne::scratch::ScratchPool::new();
    static TL_3D_SCRATCH_Y_64: mnemosyne::scratch::ScratchPool<Complex64> = mnemosyne::scratch::ScratchPool::new();
    static TL_3D_SCRATCH_Y_32: mnemosyne::scratch::ScratchPool<Complex32> = mnemosyne::scratch::ScratchPool::new();
    static TL_3D_SCRATCH_X_64: mnemosyne::scratch::ScratchPool<Complex64> = mnemosyne::scratch::ScratchPool::new();
    static TL_3D_SCRATCH_X_32: mnemosyne::scratch::ScratchPool<Complex32> = mnemosyne::scratch::ScratchPool::new();
}

/// Sealed trait providing thread-local plan scratch buffer access per complex type.
pub trait PlanScratch: sealed::Sealed + 'static {
    /// Run a closure with a thread-local 2D column-scratch buffer sized to `n`.
    fn with_2d_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Self]) -> R) -> R;
    /// Run a closure with a thread-local 3D Y-axis scratch buffer sized to `n`.
    fn with_3d_y_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Self]) -> R) -> R;
    /// Run a closure with a thread-local 3D X-axis scratch buffer sized to `n`.
    fn with_3d_x_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Self]) -> R) -> R;
}

impl PlanScratch for Complex64 {
    #[inline]
    fn with_2d_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        TL_2D_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_3d_y_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        TL_3D_SCRATCH_Y_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_3d_x_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        TL_3D_SCRATCH_X_64.with(|pool| pool.with_scratch(n, f))
    }
}

impl PlanScratch for Complex32 {
    #[inline]
    fn with_2d_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        TL_2D_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_3d_y_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        TL_3D_SCRATCH_Y_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_3d_x_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        TL_3D_SCRATCH_X_32.with(|pool| pool.with_scratch(n, f))
    }
}

/// Run `f` with a thread-local 2D column-scratch buffer sized to `n`.
#[inline]
pub(crate) fn with_2d_scratch<C: PlanScratch, R>(n: usize, f: impl FnOnce(&mut [C]) -> R) -> R {
    C::with_2d_scratch_impl(n, f)
}

/// Run `f` with a thread-local 3D Y-axis scratch buffer sized to `n`.
#[inline]
pub(crate) fn with_3d_y_scratch<C: PlanScratch, R>(n: usize, f: impl FnOnce(&mut [C]) -> R) -> R {
    C::with_3d_y_scratch_impl(n, f)
}

/// Run `f` with a thread-local 3D X-axis scratch buffer sized to `n`.
#[inline]
pub(crate) fn with_3d_x_scratch<C: PlanScratch, R>(n: usize, f: impl FnOnce(&mut [C]) -> R) -> R {
    C::with_3d_x_scratch_impl(n, f)
}
