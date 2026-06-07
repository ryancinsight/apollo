//! Plan-owned FFT workspace allocation helpers.

#![allow(clippy::uninit_vec)]

use num_complex::{Complex32, Complex64};
use std::cell::RefCell;

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
    static TL_2D_SCRATCH_64: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
    static TL_2D_SCRATCH_32: RefCell<Vec<Complex32>> = const { RefCell::new(Vec::new()) };
    static TL_3D_SCRATCH_Y_64: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
    static TL_3D_SCRATCH_Y_32: RefCell<Vec<Complex32>> = const { RefCell::new(Vec::new()) };
    static TL_3D_SCRATCH_X_64: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
    static TL_3D_SCRATCH_X_32: RefCell<Vec<Complex32>> = const { RefCell::new(Vec::new()) };
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
        TL_2D_SCRATCH_64.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = n.saturating_sub(scratch.len());
            if need > 0 {
                scratch.reserve(need);
                // SAFETY: the caller's gather loop overwrites every element
                // before any read; zero-fill is unnecessary (matching the
                // Matching the kernel scratch contract: the gather loop
                // overwrites every element before any read.
                unsafe {
                    scratch.set_len(n);
                }
            }
            f(&mut scratch[..n])
        })
    }

    #[inline]
    fn with_3d_y_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        TL_3D_SCRATCH_Y_64.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = n.saturating_sub(scratch.len());
            if need > 0 {
                scratch.reserve(need);
                unsafe {
                    scratch.set_len(n);
                }
            }
            f(&mut scratch[..n])
        })
    }

    #[inline]
    fn with_3d_x_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        TL_3D_SCRATCH_X_64.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = n.saturating_sub(scratch.len());
            if need > 0 {
                scratch.reserve(need);
                unsafe {
                    scratch.set_len(n);
                }
            }
            f(&mut scratch[..n])
        })
    }
}

impl PlanScratch for Complex32 {
    #[inline]
    fn with_2d_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        TL_2D_SCRATCH_32.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = n.saturating_sub(scratch.len());
            if need > 0 {
                scratch.reserve(need);
                // SAFETY: the caller's gather loop overwrites every element
                // before any read; zero-fill is unnecessary.
                unsafe {
                    scratch.set_len(n);
                }
            }
            f(&mut scratch[..n])
        })
    }

    #[inline]
    fn with_3d_y_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        TL_3D_SCRATCH_Y_32.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = n.saturating_sub(scratch.len());
            if need > 0 {
                scratch.reserve(need);
                unsafe {
                    scratch.set_len(n);
                }
            }
            f(&mut scratch[..n])
        })
    }

    #[inline]
    fn with_3d_x_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        TL_3D_SCRATCH_X_32.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = n.saturating_sub(scratch.len());
            if need > 0 {
                scratch.reserve(need);
                unsafe {
                    scratch.set_len(n);
                }
            }
            f(&mut scratch[..n])
        })
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
