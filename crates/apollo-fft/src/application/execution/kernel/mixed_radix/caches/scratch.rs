use num_complex::{Complex32, Complex64};
use mnemosyne::scratch::{ScratchElement, ScratchPool};

thread_local! {
    // Stockham scratch pools
    static TL_STOCKHAM_SCRATCH_64: ScratchPool<Complex64> = ScratchPool::new();
    static TL_STOCKHAM_SCRATCH_32: ScratchPool<Complex32> = ScratchPool::new();

    // PFA scratch pools
    static TL_PFA_SCRATCH_64: ScratchPool<Complex64> = ScratchPool::new();
    static TL_PFA_SCRATCH_32: ScratchPool<Complex32> = ScratchPool::new();

    // Rader padded scratch pools
    static TL_RADER_PADDED_SCRATCH_64: ScratchPool<Complex64> = ScratchPool::new();
    static TL_RADER_PADDED_SCRATCH_32: ScratchPool<Complex32> = ScratchPool::new();

    // Bluestein chirp scratch pools
    static TL_BLUESTEIN_SCRATCH_64: ScratchPool<Complex64> = ScratchPool::new();
    static TL_BLUESTEIN_SCRATCH_32: ScratchPool<Complex32> = ScratchPool::new();
}

mod sealed {
    pub(crate) trait ScratchDispatchSealed {}
}

/// Maps supported complex element types to their thread-local scratch pools.
pub(crate) trait ScratchDispatch: ScratchElement + sealed::ScratchDispatchSealed + 'static {
    fn with_stockham_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_pfa_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_rader_padded_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_bluestein_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
}

impl sealed::ScratchDispatchSealed for Complex64 {}

impl ScratchDispatch for Complex64 {
    #[inline]
    fn with_stockham_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_STOCKHAM_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_pfa_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_PFA_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_rader_padded_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_RADER_PADDED_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_bluestein_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_BLUESTEIN_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }
}

impl sealed::ScratchDispatchSealed for Complex32 {}

impl ScratchDispatch for Complex32 {
    #[inline]
    fn with_stockham_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_STOCKHAM_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_pfa_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_PFA_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_rader_padded_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_RADER_PADDED_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_bluestein_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_BLUESTEIN_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
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
