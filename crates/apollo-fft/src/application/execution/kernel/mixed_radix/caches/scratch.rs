use num_complex::{Complex32, Complex64};
use mnemosyne::scratch::ScratchPool;

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
    pub(crate) trait ScratchStoreSealed {}
}

/// Sealed trait providing thread-local scratch buffer access per complex type.
///
/// Implemented for `Complex64` and `Complex32`. All access is via the three
/// generic free functions `with_stockham_scratch`, `with_pfa_scratch`, and
/// `with_rader_padded_scratch`.
pub(crate) trait ScratchStore: sealed::ScratchStoreSealed + 'static {
    fn with_stockham_scratch_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_pfa_scratch_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_rader_padded_scratch_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_bluestein_scratch_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
}

impl sealed::ScratchStoreSealed for Complex64 {}
impl ScratchStore for Complex64 {
    #[inline]
    fn with_stockham_scratch_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_STOCKHAM_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_pfa_scratch_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_PFA_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_rader_padded_scratch_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_RADER_PADDED_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_bluestein_scratch_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_BLUESTEIN_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }
}

impl sealed::ScratchStoreSealed for Complex32 {}
impl ScratchStore for Complex32 {
    #[inline]
    fn with_stockham_scratch_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_STOCKHAM_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_pfa_scratch_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_PFA_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_rader_padded_scratch_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_RADER_PADDED_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn with_bluestein_scratch_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_BLUESTEIN_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }
}

#[inline]
pub(crate) fn with_stockham_scratch<C: ScratchStore, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_stockham_scratch_impl(n, f)
}

#[inline]
pub(crate) fn with_pfa_scratch<C: ScratchStore, R, F: FnOnce(&mut [C]) -> R>(n: usize, f: F) -> R {
    C::with_pfa_scratch_impl(n, f)
}

#[inline]
pub(crate) fn with_rader_padded_scratch<C: ScratchStore, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_rader_padded_scratch_impl(n, f)
}

#[inline]
pub(crate) fn with_bluestein_scratch<C: ScratchStore, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_bluestein_scratch_impl(n, f)
}
