use eunomia::{Complex32, Complex64};
use mnemosyne::scratch::{ScratchBank, ScratchElement};

const STOCKHAM_SLOT: usize = 0;
const PFA_SLOT: usize = 1;
const RADER_PADDED_SLOT: usize = 2;
const BLUESTEIN_SLOT: usize = 3;
const SCRATCH_ROLE_COUNT: usize = 4;

thread_local! {
    static TL_SCRATCH_BANK_64: ScratchBank<Complex64, SCRATCH_ROLE_COUNT> = const { ScratchBank::new() };
    static TL_SCRATCH_BANK_32: ScratchBank<Complex32, SCRATCH_ROLE_COUNT> = const { ScratchBank::new() };
}

mod sealed {
    pub(crate) trait ScratchDispatchSealed {}
}

/// Maps supported complex element types to their thread-local scratch pools.
pub(crate) trait ScratchDispatch:
    ScratchElement + sealed::ScratchDispatchSealed + 'static
{
    fn with_stockham_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_pfa_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_rader_padded_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_bluestein_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
}

impl sealed::ScratchDispatchSealed for Complex64 {}

impl ScratchDispatch for Complex64 {
    #[inline]
    fn with_stockham_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_SCRATCH_BANK_64.with(|bank| bank.with_scratch::<STOCKHAM_SLOT, _>(n, f))
    }

    #[inline]
    fn with_pfa_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_SCRATCH_BANK_64.with(|bank| bank.with_scratch::<PFA_SLOT, _>(n, f))
    }

    #[inline]
    fn with_rader_padded_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_SCRATCH_BANK_64.with(|bank| bank.with_scratch::<RADER_PADDED_SLOT, _>(n, f))
    }

    #[inline]
    fn with_bluestein_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_SCRATCH_BANK_64.with(|bank| bank.with_scratch::<BLUESTEIN_SLOT, _>(n, f))
    }
}

impl sealed::ScratchDispatchSealed for Complex32 {}

impl ScratchDispatch for Complex32 {
    #[inline]
    fn with_stockham_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_SCRATCH_BANK_32.with(|bank| bank.with_scratch::<STOCKHAM_SLOT, _>(n, f))
    }

    #[inline]
    fn with_pfa_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_SCRATCH_BANK_32.with(|bank| bank.with_scratch::<PFA_SLOT, _>(n, f))
    }

    #[inline]
    fn with_rader_padded_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_SCRATCH_BANK_32.with(|bank| bank.with_scratch::<RADER_PADDED_SLOT, _>(n, f))
    }

    #[inline]
    fn with_bluestein_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_SCRATCH_BANK_32.with(|bank| bank.with_scratch::<BLUESTEIN_SLOT, _>(n, f))
    }
}

#[inline]
pub(crate) fn with_stockham_scratch<C: ScratchDispatch, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_stockham_impl(n, f)
}

#[inline]
pub(crate) fn with_pfa_scratch<C: ScratchDispatch, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_pfa_impl(n, f)
}

#[inline]
pub(crate) fn with_rader_padded_scratch<C: ScratchDispatch, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_rader_padded_impl(n, f)
}

#[inline]
pub(crate) fn with_bluestein_scratch<C: ScratchDispatch, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_bluestein_impl(n, f)
}

/// Releases idle capacity from the mixed-radix transform scratch banks on the
/// current thread.
pub(crate) fn release_thread_local_scratch() {
    TL_SCRATCH_BANK_64.with(|bank| bank.release());
    TL_SCRATCH_BANK_32.with(|bank| bank.release());
}

#[cfg(test)]
pub(crate) fn thread_local_scratch_capacity() -> usize {
    let capacity_64 = TL_SCRATCH_BANK_64.with(|bank| {
        bank.capacity::<STOCKHAM_SLOT>()
            + bank.capacity::<PFA_SLOT>()
            + bank.capacity::<RADER_PADDED_SLOT>()
            + bank.capacity::<BLUESTEIN_SLOT>()
    });
    let capacity_32 = TL_SCRATCH_BANK_32.with(|bank| {
        bank.capacity::<STOCKHAM_SLOT>()
            + bank.capacity::<PFA_SLOT>()
            + bank.capacity::<RADER_PADDED_SLOT>()
            + bank.capacity::<BLUESTEIN_SLOT>()
    });
    capacity_64 + capacity_32
}

#[cfg(test)]
mod tests {
    use super::{release_thread_local_scratch, STOCKHAM_SLOT, TL_SCRATCH_BANK_64};

    #[test]
    fn release_reclaims_idle_mixed_radix_capacity() {
        TL_SCRATCH_BANK_64.with(|bank| {
            bank.with_scratch::<STOCKHAM_SLOT, _>(257, |scratch| {
                scratch[0] = eunomia::Complex64::new(1.0, -2.0);
            });
            assert!(bank.capacity::<STOCKHAM_SLOT>() >= 257);
        });

        release_thread_local_scratch();

        TL_SCRATCH_BANK_64.with(|bank| {
            assert_eq!(bank.capacity::<STOCKHAM_SLOT>(), 0);
        });
    }
}
