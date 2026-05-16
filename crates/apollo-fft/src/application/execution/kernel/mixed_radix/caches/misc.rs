use super::super::super::radix_shape::{coprime_factors, factorize_composite as factorize_prime23, is_prime};
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

static PRIME23_RADIX_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Option<Arc<[usize]>>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_SPECTRUM_64_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_SPECTRUM_32_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_PERM_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), (Arc<[usize]>, Arc<[usize]>)>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static COPRIME_FACTORS_CACHE: std::sync::LazyLock<
    RwLock<HashMap<usize, Option<(usize, usize)>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static IS_PRIME_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, bool>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static PRIMITIVE_ROOT_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, usize>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    pub(super) static TL_PRIME23_RADIX: RefCell<HashMap<usize, Option<Arc<[usize]>>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_SPECTRUM_64: RefCell<HashMap<(usize, usize, usize), Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_SPECTRUM_32: RefCell<HashMap<(usize, usize, usize), Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_PERM: RefCell<HashMap<(usize, usize, usize), (Arc<[usize]>, Arc<[usize]>)>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_COPRIME_FACTORS: RefCell<HashMap<usize, Option<(usize, usize)>>> =
        RefCell::new(HashMap::with_capacity(16));
    pub(super) static TL_IS_PRIME: RefCell<HashMap<usize, bool>> =
        RefCell::new(HashMap::with_capacity(16));
    pub(super) static TL_PRIMITIVE_ROOT: RefCell<HashMap<usize, usize>> =
        RefCell::new(HashMap::with_capacity(16));
}

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

#[inline]
pub(crate) fn cached_prime23_radices(n: usize) -> Option<Arc<[usize]>> {
    if let Some(radices) = TL_PRIME23_RADIX.with(|c| c.borrow().get(&n).cloned()) {
        return radices;
    }
    let radices = {
        let maybe_cached = PRIME23_RADIX_CACHE.read().get(&n).cloned();
        if let Some(radices) = maybe_cached {
            radices
        } else {
            let new_radices = factorize_prime23(n).map(|rad| Arc::from(rad.into_boxed_slice()));
            PRIME23_RADIX_CACHE
                .write()
                .entry(n)
                .or_insert_with(|| match &new_radices {
                    Some(a) => Some(Arc::clone(a)),
                    None => None,
                })
                .clone()
        }
    };
    TL_PRIME23_RADIX.with(|c| c.borrow_mut().insert(n, radices.clone()));
    radices
}

#[inline]
pub(crate) fn cached_coprime_factors(n: usize) -> Option<(usize, usize)> {
    if let Some(v) = TL_COPRIME_FACTORS.with(|c| c.borrow().get(&n).copied()) {
        return v;
    }
    let v = {
        let maybe = COPRIME_FACTORS_CACHE.read().get(&n).copied();
        if let Some(v) = maybe {
            v
        } else {
            let result = coprime_factors(n);
            *COPRIME_FACTORS_CACHE.write().entry(n).or_insert(result)
        }
    };
    TL_COPRIME_FACTORS.with(|c| c.borrow_mut().insert(n, v));
    v
}

#[inline]
pub(crate) fn cached_primitive_root(n: usize) -> usize {
    if let Some(v) = TL_PRIMITIVE_ROOT.with(|c| c.borrow().get(&n).copied()) {
        return v;
    }
    let v = {
        let maybe = PRIMITIVE_ROOT_CACHE.read().get(&n).copied();
        if let Some(v) = maybe {
            v
        } else {
            let result =
                crate::application::execution::kernel::rader::generator::primitive_root(n);
            *PRIMITIVE_ROOT_CACHE.write().entry(n).or_insert(result)
        }
    };
    TL_PRIMITIVE_ROOT.with(|c| c.borrow_mut().insert(n, v));
    v
}

#[inline]
pub(crate) fn cached_is_prime(n: usize) -> bool {
    if let Some(v) = TL_IS_PRIME.with(|c| c.borrow().get(&n).copied()) {
        return v;
    }
    let v = {
        let maybe = IS_PRIME_CACHE.read().get(&n).copied();
        if let Some(v) = maybe {
            v
        } else {
            let result = is_prime(n);
            *IS_PRIME_CACHE.write().entry(n).or_insert(result)
        }
    };
    TL_IS_PRIME.with(|c| c.borrow_mut().insert(n, v));
    v
}

#[inline]
pub(crate) fn cached_rader_spectrum_64(
    key: (usize, usize, usize),
    build_fn: impl FnOnce((usize, usize, usize)) -> Vec<Complex64>,
) -> Arc<[Complex64]> {
    tl_cached_k3(&TL_RADER_SPECTRUM_64, &RADER_SPECTRUM_64_CACHE, key, build_fn)
}

#[inline]
pub(crate) fn cached_rader_spectrum_32(
    key: (usize, usize, usize),
    build_fn: impl FnOnce((usize, usize, usize)) -> Vec<Complex32>,
) -> Arc<[Complex32]> {
    tl_cached_k3(&TL_RADER_SPECTRUM_32, &RADER_SPECTRUM_32_CACHE, key, build_fn)
}

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
