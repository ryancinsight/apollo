//! Thread-local plan scratch buffers per complex scalar type.
//!
//! 2D and 3D plans use per-precision thread-local buffers instead of
//! plan-owned mutex-protected buffers. The trait is sealed to the complex
//! layouts supported by [`MixedRadixScalar`]; each binds its thread-local bank
//! and one blanket implementation serves them all.
//!
//! Every role gives its storage back when a Moirai worker parks (the idle
//! hook in `scratch_hook`), and workers park between parallel regions. A
//! buffer a parallel task borrows is therefore allocated and zeroed again on
//! each worker for each region: keep such borrows lane-sized. A slab-sized
//! per-task block at 64³ made the half forward's z sweep 144-174 µs against
//! 37 µs for a lane-sized one.
//!
//! [`MixedRadixScalar`]: super::MixedRadixScalar

use eunomia::{Complex32, Complex64};
use mnemosyne::scratch::ScratchBank;
use std::thread::LocalKey;

const SCRATCH_2D_SLOT: usize = 0;
const SCRATCH_3D_Y_SLOT: usize = 1;
const SCRATCH_3D_X_SLOT: usize = 2;
const PLAN_SCRATCH_ROLE_COUNT: usize = 3;

thread_local! {
    static TL_PLAN_SCRATCH_BANK_64: ScratchBank<Complex64, PLAN_SCRATCH_ROLE_COUNT> =
        const { ScratchBank::new() };
    static TL_PLAN_SCRATCH_BANK_32: ScratchBank<Complex32, PLAN_SCRATCH_ROLE_COUNT> =
        const { ScratchBank::new() };
}

mod sealed {
    use super::PLAN_SCRATCH_ROLE_COUNT;
    use mnemosyne::scratch::{ScratchBank, ScratchElement};
    use std::thread::LocalKey;

    /// Binds a complex element to its thread-local plan scratch bank.
    ///
    /// Rust has no generic statics, so the bank each element owns is the one
    /// per-element fact; everything done with it is written once, in the
    /// blanket [`PlanScratch`](super::PlanScratch) implementation.
    pub trait Sealed: ScratchElement + Sized + 'static {
        const BANK: &'static LocalKey<ScratchBank<Self, PLAN_SCRATCH_ROLE_COUNT>>;
    }
}

impl sealed::Sealed for Complex64 {
    const BANK: &'static LocalKey<ScratchBank<Self, PLAN_SCRATCH_ROLE_COUNT>> =
        &TL_PLAN_SCRATCH_BANK_64;
}

impl sealed::Sealed for Complex32 {
    const BANK: &'static LocalKey<ScratchBank<Self, PLAN_SCRATCH_ROLE_COUNT>> =
        &TL_PLAN_SCRATCH_BANK_32;
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

impl<C: sealed::Sealed> PlanScratch for C {
    #[inline]
    fn with_2d_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [C]) -> R) -> R {
        C::BANK.with(|bank| bank.with_scratch::<SCRATCH_2D_SLOT, _>(n, f))
    }

    #[inline]
    fn with_3d_y_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [C]) -> R) -> R {
        C::BANK.with(|bank| bank.with_scratch::<SCRATCH_3D_Y_SLOT, _>(n, f))
    }

    #[inline]
    fn with_3d_x_scratch_impl<R>(n: usize, f: impl FnOnce(&mut [C]) -> R) -> R {
        C::BANK.with(|bank| bank.with_scratch::<SCRATCH_3D_X_SLOT, _>(n, f))
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

/// Releases idle capacity from the plan scratch banks on the current thread.
pub(crate) fn release_thread_local_scratch() {
    release_bank::<Complex64>();
    release_bank::<Complex32>();
}

fn release_bank<C: PlanScratch>() {
    C::BANK.with(ScratchBank::release);
}

#[cfg(test)]
pub(crate) fn thread_local_scratch_capacity() -> usize {
    bank_capacity::<Complex64>() + bank_capacity::<Complex32>()
}

#[cfg(test)]
fn bank_capacity<C: PlanScratch>() -> usize {
    C::BANK.with(|bank| {
        bank.capacity::<SCRATCH_2D_SLOT>()
            + bank.capacity::<SCRATCH_3D_Y_SLOT>()
            + bank.capacity::<SCRATCH_3D_X_SLOT>()
    })
}

#[cfg(test)]
mod tests {
    use super::{bank_capacity, release_thread_local_scratch, PlanScratch};
    use eunomia::{Complex32, Complex64};

    /// Every role of the element's bank grows to the borrowed length and
    /// gives it all back on release.
    fn release_reclaims_every_role<C: PlanScratch + Copy>(marker: C) {
        const LEN: usize = 257;
        C::with_2d_scratch_impl(LEN, |scratch| scratch[0] = marker);
        C::with_3d_y_scratch_impl(LEN, |scratch| scratch[0] = marker);
        C::with_3d_x_scratch_impl(LEN, |scratch| scratch[0] = marker);
        assert!(bank_capacity::<C>() >= 3 * LEN);

        release_thread_local_scratch();

        assert_eq!(bank_capacity::<C>(), 0);
    }

    #[test]
    fn release_reclaims_idle_plan_capacity() {
        release_reclaims_every_role(Complex64::new(1.0, -2.0));
        release_reclaims_every_role(Complex32::new(1.0, -2.0));
    }
}
