use mnemosyne::scratch::{ScratchBank, ScratchElement};
use eunomia::{Complex32, Complex64};

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
