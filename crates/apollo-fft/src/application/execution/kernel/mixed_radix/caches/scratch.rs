use eunomia::{Complex32, Complex64};
use mnemosyne::scratch::ScratchBank;
use std::thread::LocalKey;

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
    use super::SCRATCH_ROLE_COUNT;
    use mnemosyne::scratch::{ScratchBank, ScratchElement};
    use std::thread::LocalKey;

    /// Binds a complex element to its thread-local scratch bank.
    ///
    /// Rust has no generic statics, so the bank each element owns is the one
    /// per-element fact; everything done with it is written once, in the
    /// blanket [`ScratchDispatch`](super::ScratchDispatch) implementation.
    pub(crate) trait ScratchDispatchSealed: ScratchElement + Sized + 'static {
        const BANK: &'static LocalKey<ScratchBank<Self, SCRATCH_ROLE_COUNT>>;
    }
}

/// Maps supported complex element types to their thread-local scratch pools.
pub(crate) trait ScratchDispatch: sealed::ScratchDispatchSealed {
    fn with_stockham_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_pfa_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_rader_padded_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_bluestein_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
}

impl sealed::ScratchDispatchSealed for Complex64 {
    const BANK: &'static LocalKey<ScratchBank<Self, SCRATCH_ROLE_COUNT>> = &TL_SCRATCH_BANK_64;
}

impl sealed::ScratchDispatchSealed for Complex32 {
    const BANK: &'static LocalKey<ScratchBank<Self, SCRATCH_ROLE_COUNT>> = &TL_SCRATCH_BANK_32;
}

impl<C: sealed::ScratchDispatchSealed> ScratchDispatch for C {
    #[inline]
    fn with_stockham_impl<R, F: FnOnce(&mut [C]) -> R>(n: usize, f: F) -> R {
        C::BANK.with(|bank| bank.with_scratch::<STOCKHAM_SLOT, _>(n, f))
    }

    #[inline]
    fn with_pfa_impl<R, F: FnOnce(&mut [C]) -> R>(n: usize, f: F) -> R {
        C::BANK.with(|bank| bank.with_scratch::<PFA_SLOT, _>(n, f))
    }

    #[inline]
    fn with_rader_padded_impl<R, F: FnOnce(&mut [C]) -> R>(n: usize, f: F) -> R {
        C::BANK.with(|bank| bank.with_scratch::<RADER_PADDED_SLOT, _>(n, f))
    }

    #[inline]
    fn with_bluestein_impl<R, F: FnOnce(&mut [C]) -> R>(n: usize, f: F) -> R {
        C::BANK.with(|bank| bank.with_scratch::<BLUESTEIN_SLOT, _>(n, f))
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
    release_bank::<Complex64>();
    release_bank::<Complex32>();
}

fn release_bank<C: ScratchDispatch>() {
    C::BANK.with(ScratchBank::release);
}

#[cfg(test)]
pub(crate) fn thread_local_scratch_capacity() -> usize {
    bank_capacity::<Complex64>() + bank_capacity::<Complex32>()
}

#[cfg(test)]
fn bank_capacity<C: ScratchDispatch>() -> usize {
    C::BANK.with(|bank| {
        bank.capacity::<STOCKHAM_SLOT>()
            + bank.capacity::<PFA_SLOT>()
            + bank.capacity::<RADER_PADDED_SLOT>()
            + bank.capacity::<BLUESTEIN_SLOT>()
    })
}

#[cfg(test)]
mod tests {
    use super::{
        bank_capacity, release_thread_local_scratch, with_bluestein_scratch, with_pfa_scratch,
        with_rader_padded_scratch, with_stockham_scratch, ScratchDispatch,
    };
    use eunomia::{Complex32, Complex64};

    /// Every role of the element's bank grows to the borrowed length and
    /// gives it all back on release.
    fn release_reclaims_every_role<C: ScratchDispatch + Copy>(marker: C) {
        const LEN: usize = 257;
        with_stockham_scratch::<C, _, _>(LEN, |scratch| scratch[0] = marker);
        with_pfa_scratch::<C, _, _>(LEN, |scratch| scratch[0] = marker);
        with_rader_padded_scratch::<C, _, _>(LEN, |scratch| scratch[0] = marker);
        with_bluestein_scratch::<C, _, _>(LEN, |scratch| scratch[0] = marker);
        assert!(bank_capacity::<C>() >= 4 * LEN);

        release_thread_local_scratch();

        assert_eq!(bank_capacity::<C>(), 0);
    }

    #[test]
    fn release_reclaims_idle_mixed_radix_capacity() {
        release_reclaims_every_role(Complex64::new(1.0, -2.0));
        release_reclaims_every_role(Complex32::new(1.0, -2.0));
    }
}
