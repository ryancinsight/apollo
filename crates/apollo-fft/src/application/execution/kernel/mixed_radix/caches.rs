//! Thread-local and global twiddle/composite-radix caches for mixed-radix dispatch.

use super::super::radix_shape::factorize_composite;
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

static TWIDDLE_FWD_64_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_64_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_FWD_32_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_32_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static COMPOSITE_RADIX_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Option<Arc<[usize]>>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static TL_FWD_64: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_64: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_FWD_32: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_32: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_COMPOSITE_RADIX: RefCell<HashMap<usize, Option<Arc<[usize]>>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_STOCKHAM_SCRATCH_64: RefCell<Vec<Complex64>> =
        const { RefCell::new(Vec::new()) };
    static TL_STOCKHAM_SCRATCH_32: RefCell<Vec<Complex32>> =
        const { RefCell::new(Vec::new()) };
    static TL_PFA_SCRATCH_64: RefCell<Vec<Vec<Complex64>>> =
        const { RefCell::new(Vec::new()) };
    static TL_PFA_SCRATCH_32: RefCell<Vec<Vec<Complex32>>> =
        const { RefCell::new(Vec::new()) };
    static TL_RADER_SCRATCH_64: RefCell<Vec<Vec<Complex64>>> =
        const { RefCell::new(Vec::new()) };
    static TL_RADER_SCRATCH_32: RefCell<Vec<Vec<Complex32>>> =
        const { RefCell::new(Vec::new()) };
    static TL_RADER_PADDED_SCRATCH_64: RefCell<Vec<Vec<Complex64>>> =
        const { RefCell::new(Vec::new()) };
    static TL_RADER_PADDED_SCRATCH_32: RefCell<Vec<Vec<Complex32>>> =
        const { RefCell::new(Vec::new()) };
}

#[inline]
pub(crate) fn with_stockham_scratch_64<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    TL_STOCKHAM_SCRATCH_64.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if scratch.len() < n {
            let cur = scratch.len();
            scratch.reserve(n.saturating_sub(cur));
            unsafe { scratch.set_len(n) };
        }
        f(&mut scratch[..n])
    })
}

#[inline]
pub(crate) fn with_stockham_scratch_32<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
    TL_STOCKHAM_SCRATCH_32.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if scratch.len() < n {
            let cur = scratch.len();
            scratch.reserve(n.saturating_sub(cur));
            unsafe { scratch.set_len(n) };
        }
        f(&mut scratch[..n])
    })
}

#[inline]
pub(crate) fn with_pfa_scratch_64<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    let mut scratch = TL_PFA_SCRATCH_64.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    if scratch.len() < n {
        let cur = scratch.len();
        scratch.reserve(n.saturating_sub(cur));
        unsafe { scratch.set_len(n) };
    }
    let res = f(&mut scratch[..n]);
    TL_PFA_SCRATCH_64.with(|pool| pool.borrow_mut().push(scratch));
    res
}

#[inline]
pub(crate) fn with_pfa_scratch_32<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
    let mut scratch = TL_PFA_SCRATCH_32.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    if scratch.len() < n {
        let cur = scratch.len();
        scratch.reserve(n.saturating_sub(cur));
        unsafe { scratch.set_len(n) };
    }
    let res = f(&mut scratch[..n]);
    TL_PFA_SCRATCH_32.with(|pool| pool.borrow_mut().push(scratch));
    res
}

#[inline]
pub(crate) fn with_rader_scratch_64<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    let mut scratch = TL_RADER_SCRATCH_64.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    if scratch.len() < n {
        let cur = scratch.len();
        scratch.reserve(n.saturating_sub(cur));
        unsafe { scratch.set_len(n) };
    }
    let res = f(&mut scratch[..n]);
    TL_RADER_SCRATCH_64.with(|pool| pool.borrow_mut().push(scratch));
    res
}

#[inline]
pub(crate) fn with_rader_scratch_32<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
    let mut scratch = TL_RADER_SCRATCH_32.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    if scratch.len() < n {
        let cur = scratch.len();
        scratch.reserve(n.saturating_sub(cur));
        unsafe { scratch.set_len(n) };
    }
    let res = f(&mut scratch[..n]);
    TL_RADER_SCRATCH_32.with(|pool| pool.borrow_mut().push(scratch));
    res
}

#[inline]
pub(crate) fn with_rader_padded_scratch_64<R>(
    n: usize,
    f: impl FnOnce(&mut [Complex64]) -> R,
) -> R {
    let mut scratch =
        TL_RADER_PADDED_SCRATCH_64.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    if scratch.len() < n {
        let cur = scratch.len();
        scratch.reserve(n.saturating_sub(cur));
        unsafe { scratch.set_len(n) };
    }
    let res = f(&mut scratch[..n]);
    TL_RADER_PADDED_SCRATCH_64.with(|pool| pool.borrow_mut().push(scratch));
    res
}

#[inline]
pub(crate) fn with_rader_padded_scratch_32<R>(
    n: usize,
    f: impl FnOnce(&mut [Complex32]) -> R,
) -> R {
    let mut scratch =
        TL_RADER_PADDED_SCRATCH_32.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    if scratch.len() < n {
        let cur = scratch.len();
        scratch.reserve(n.saturating_sub(cur));
        unsafe { scratch.set_len(n) };
    }
    let res = f(&mut scratch[..n]);
    TL_RADER_PADDED_SCRATCH_32.with(|pool| pool.borrow_mut().push(scratch));
    res
}

#[inline]
fn tl_cached<T: Clone>(
    tl: &'static std::thread::LocalKey<RefCell<HashMap<usize, Arc<[T]>>>>,
    global: &'static std::sync::LazyLock<RwLock<HashMap<usize, Arc<[T]>>>>,
    n: usize,
    build_fn: impl FnOnce(usize) -> Vec<T>,
) -> Arc<[T]> {
    if let Some(tw) = tl.with(|c| c.borrow().get(&n).cloned()) {
        return tw;
    }

    let tw = {
        let maybe_cached = global.read().get(&n).cloned();
        if let Some(tw) = maybe_cached {
            tw
        } else {
            let new_tw: Arc<[T]> = Arc::from(build_fn(n));
            global
                .write()
                .entry(n)
                .or_insert_with(|| Arc::clone(&new_tw))
                .clone()
        }
    };
    tl.with(|c| c.borrow_mut().insert(n, Arc::clone(&tw)));
    tw
}

#[inline]
pub(crate) fn cached_twiddle_fwd_64(n: usize) -> Arc<[Complex64]> {
    tl_cached(
        &TL_FWD_64,
        &TWIDDLE_FWD_64_CACHE,
        n,
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_64(n: usize) -> Arc<[Complex64]> {
    tl_cached(
        &TL_INV_64,
        &TWIDDLE_INV_64_CACHE,
        n,
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_inverse_twiddle_table,
    )
}

#[inline]
pub(crate) fn cached_twiddle_fwd_32(n: usize) -> Arc<[Complex32]> {
    tl_cached(
        &TL_FWD_32,
        &TWIDDLE_FWD_32_CACHE,
        n,
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_32(n: usize) -> Arc<[Complex32]> {
    tl_cached(
        &TL_INV_32,
        &TWIDDLE_INV_32_CACHE,
        n,
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_inverse_twiddle_table,
    )
}

#[inline]
pub(crate) fn cached_composite_radices(n: usize) -> Option<Arc<[usize]>> {
    if let Some(radices) = TL_COMPOSITE_RADIX.with(|c| c.borrow().get(&n).cloned()) {
        return radices;
    }

    let radices = {
        let maybe_cached = COMPOSITE_RADIX_CACHE.read().get(&n).cloned();
        if let Some(radices) = maybe_cached {
            radices
        } else {
            let new_radices = factorize_composite(n).map(|rad| Arc::from(rad.into_boxed_slice()));
            COMPOSITE_RADIX_CACHE
                .write()
                .entry(n)
                .or_insert_with(|| match &new_radices {
                    Some(a) => Some(Arc::clone(a)),
                    None => None,
                })
                .clone()
        }
    };

    TL_COMPOSITE_RADIX.with(|c| c.borrow_mut().insert(n, radices.clone()));
    radices
}
