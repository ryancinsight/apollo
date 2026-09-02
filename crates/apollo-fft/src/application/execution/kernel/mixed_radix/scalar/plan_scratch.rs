//! Thread-local plan scratch buffers per complex scalar type.
//!
//! 2D and 3D plans use per-precision thread-local buffers instead of
//! plan-owned mutex-protected buffers. The trait is sealed to the two complex
//! scalar layouts supported by [`MixedRadixScalar`].
//!
//! [`MixedRadixScalar`]: super::MixedRadixScalar

use eunomia::{Complex32, Complex64};

const SCRATCH_2D_SLOT: usize = 0;
const SCRATCH_3D_Y_SLOT: usize = 1;
const SCRATCH_3D_X_SLOT: usize = 2;
const PLAN_SCRATCH_ROLE_COUNT: usize = 3;

mod sealed {
    pub trait Sealed {}

    impl Sealed for eunomia::Complex32 {}
    impl Sealed for eunomia::Complex64 {}
}

thread_local! {
    static TL_PLAN_SCRATCH_BANK_64: mnemosyne::scratch::ScratchBank<Complex64, PLAN_SCRATCH_ROLE_COUNT> =
        const { mnemosyne::scratch::ScratchBank::new() };
    static TL_PLAN_SCRATCH_BANK_32: mnemosyne::scratch::ScratchBank<Complex32, PLAN_SCRATCH_ROLE_COUNT> =
        const { mnemosyne::scratch::ScratchBank::new() };
}

/// Sealed trait providing thread-local plan scratch buffer access per complex type.
pub trait PlanScratch: sealed::Sealed + 'static {
    /// Run a closure with a thread-local 2D column-scratch buffer sized to `n`.
    fn with_2d_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Self]) -> R) -> R
    where
        Self: Sized;

    /// Run a closure with a thread-local 3D Y-axis scratch buffer sized to `n`.
    fn with_3d_y_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Self]) -> R) -> R
    where
        Self: Sized;

    /// Run a closure with a thread-local 3D X-axis scratch buffer sized to `n`.
    fn with_3d_x_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Self]) -> R) -> R
    where
        Self: Sized;
}

impl PlanScratch for Complex64 {
    #[inline]
    fn with_2d_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        TL_PLAN_SCRATCH_BANK_64.with(|bank| bank.with_scratch::<SCRATCH_2D_SLOT, _>(n, f))
    }

    #[inline]
    fn with_3d_y_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        TL_PLAN_SCRATCH_BANK_64.with(|bank| bank.with_scratch::<SCRATCH_3D_Y_SLOT, _>(n, f))
    }

    #[inline]
    fn with_3d_x_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        TL_PLAN_SCRATCH_BANK_64.with(|bank| bank.with_scratch::<SCRATCH_3D_X_SLOT, _>(n, f))
    }
}

impl PlanScratch for Complex32 {
    #[inline]
    fn with_2d_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        TL_PLAN_SCRATCH_BANK_32.with(|bank| bank.with_scratch::<SCRATCH_2D_SLOT, _>(n, f))
    }

    #[inline]
    fn with_3d_y_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        TL_PLAN_SCRATCH_BANK_32.with(|bank| bank.with_scratch::<SCRATCH_3D_Y_SLOT, _>(n, f))
    }

    #[inline]
    fn with_3d_x_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        TL_PLAN_SCRATCH_BANK_32.with(|bank| bank.with_scratch::<SCRATCH_3D_X_SLOT, _>(n, f))
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

/// Run `f` with rank-disjoint thread-local logical-view staging sized to `n`.
///
/// A rank-one transform borrows the 3-D Y role, a rank-two transform the 3-D
/// X role, and a rank-three transform the 2-D role. No role is reached by the
/// nested passes of the rank that borrows it, so staging remains live without
/// adding public trait surface or a fourth full-volume scratch allocation.
#[inline]
pub(crate) fn with_view_staging<C: PlanScratch, const N: usize, R>(
    n: usize,
    f: impl FnOnce(&mut [C]) -> R,
) -> R {
    match N {
        1 => C::with_3d_y_scratch_impl(n, f),
        2 => C::with_3d_x_scratch_impl(n, f),
        3 => C::with_2d_scratch_impl(n, f),
        _ => unreachable!(
            "invariant: logical-view staging is only used by rank-one, -two and -three plans"
        ),
    }
}
