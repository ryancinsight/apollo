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

/// Dispatches to the correct thread-local Stockham scratch pool.
#[inline]
pub(crate) fn with_stockham_scratch<C: ScratchElement, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    use core::any::TypeId;
    if TypeId::of::<C>() == TypeId::of::<Complex64>() {
        TL_STOCKHAM_SCRATCH_64.with(|pool| {
            let f = unsafe {
                core::mem::transmute::<
                    *const dyn FnOnce(&mut [C]) -> R,
                    *const dyn FnOnce(&mut [Complex64]) -> R,
                >(&f as *const _)
            };
            pool.with_scratch(n, unsafe { core::ptr::read(f) })
        })
    } else {
        TL_STOCKHAM_SCRATCH_32.with(|pool| {
            let f = unsafe {
                core::mem::transmute::<
                    *const dyn FnOnce(&mut [C]) -> R,
                    *const dyn FnOnce(&mut [Complex32]) -> R,
                >(&f as *const _)
            };
            pool.with_scratch(n, unsafe { core::ptr::read(f) })
        })
    }
}
