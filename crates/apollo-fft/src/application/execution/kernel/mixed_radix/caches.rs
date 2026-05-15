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
// Key: (n, inverse, g_inv).
static RADER_SPECTRUM_64_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_SPECTRUM_32_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
// Value: split (gather_indices, scatter_indices) — contiguous per-pass arrays.
static RADER_PERM_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), (Arc<[usize]>, Arc<[usize]>)>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

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
    static TL_RADER_SPECTRUM_64: RefCell<HashMap<(usize, usize, usize), Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_RADER_SPECTRUM_32: RefCell<HashMap<(usize, usize, usize), Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_RADER_PERM: RefCell<HashMap<(usize, usize, usize), (Arc<[usize]>, Arc<[usize]>)>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_STOCKHAM_SCRATCH_64: RefCell<Vec<Complex64>> =
        const { RefCell::new(Vec::new()) };
    static TL_STOCKHAM_SCRATCH_32: RefCell<Vec<Complex32>> =
        const { RefCell::new(Vec::new()) };
    static TL_PFA_SCRATCH_64: RefCell<Vec<Vec<Complex64>>> =
        const { RefCell::new(Vec::new()) };
    static TL_PFA_SCRATCH_32: RefCell<Vec<Vec<Complex32>>> =
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
    // Bound pool to 2: one for the outer call, one for a nested Rader sub-call.
    TL_RADER_PADDED_SCRATCH_64.with(|pool| {
        let mut p = pool.borrow_mut();
        if p.len() < 2 {
            p.push(scratch);
        }
    });
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
    // Bound pool to 2: one for the outer call, one for a nested Rader sub-call.
    TL_RADER_PADDED_SCRATCH_32.with(|pool| {
        let mut p = pool.borrow_mut();
        if p.len() < 2 {
            p.push(scratch);
        }
    });
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

/// Three-key cache helper for the Rader spectrum (key: `(n, inverse, g_inv)`).
#[inline]
fn tl_cached_k3<T: Clone>(
    tl: &'static std::thread::LocalKey<RefCell<HashMap<(usize, usize, usize), Arc<[T]>>>>,
    global: &'static std::sync::LazyLock<RwLock<HashMap<(usize, usize, usize), Arc<[T]>>>>,
    key: (usize, usize, usize),
    build_fn: impl FnOnce((usize, usize, usize)) -> Vec<T>,
) -> Arc<[T]> {
    if let Some(v) = tl.with(|c| c.borrow().get(&key).cloned()) {
        return v;
    }
    let v = {
        let maybe_cached = global.read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
            let new_v: Arc<[T]> = Arc::from(build_fn(key));
            global
                .write()
                .entry(key)
                .or_insert_with(|| Arc::clone(&new_v))
                .clone()
        }
    };
    tl.with(|c| c.borrow_mut().insert(key, Arc::clone(&v)));
    v
}

/// Cached Rader convolution kernel spectrum.
/// Key: `(n, inverse, g_inv)`.
#[inline]
pub(crate) fn cached_rader_spectrum_64(
    key: (usize, usize, usize),
    build_fn: impl FnOnce((usize, usize, usize)) -> Vec<Complex64>,
) -> Arc<[Complex64]> {
    tl_cached_k3(
        &TL_RADER_SPECTRUM_64,
        &RADER_SPECTRUM_64_CACHE,
        key,
        build_fn,
    )
}

#[inline]
pub(crate) fn cached_rader_spectrum_32(
    key: (usize, usize, usize),
    build_fn: impl FnOnce((usize, usize, usize)) -> Vec<Complex32>,
) -> Arc<[Complex32]> {
    tl_cached_k3(
        &TL_RADER_SPECTRUM_32,
        &RADER_SPECTRUM_32_CACHE,
        key,
        build_fn,
    )
}

/// Cached split permutation arrays `(gather_indices, scatter_indices)`.
/// Key: `(n, g, g_inv)`.
#[inline]
pub(crate) fn cached_rader_perm(
    key: (usize, usize, usize),
    build_fn: impl FnOnce((usize, usize, usize)) -> (Vec<usize>, Vec<usize>),
) -> (Arc<[usize]>, Arc<[usize]>) {
    if let Some(v) = TL_RADER_PERM.with(|c| c.borrow().get(&key).cloned()) {
        return v;
    }
    let v = {
        let maybe_cached = RADER_PERM_CACHE.read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
            let (gather, scatter) = build_fn(key);
            let pair: (Arc<[usize]>, Arc<[usize]>) = (Arc::from(gather), Arc::from(scatter));
            RADER_PERM_CACHE
                .write()
                .entry(key)
                .or_insert_with(|| pair.clone())
                .clone()
        }
    };
    TL_RADER_PERM.with(|c| c.borrow_mut().insert(key, v.clone()));
    v
}
