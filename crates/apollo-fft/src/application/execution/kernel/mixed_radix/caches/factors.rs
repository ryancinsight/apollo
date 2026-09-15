//! The coprime-factor and primality caches consulted by the plan dispatch.

use super::super::super::radix_shape::{coprime_factors, is_prime};
use super::direct_mapped::FLAT_CACHE_LIMIT;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::OnceLock;

static COPRIME_FACTORS_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<usize, Option<(usize, usize)>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static IS_PRIME_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, bool>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static COPRIME_FACTORS_FLAT: [OnceLock<Option<(usize, usize)>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];

static IS_PRIME_FLAT: [OnceLock<bool>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];

thread_local! {
    pub(super) static TL_COPRIME_FACTORS: RefCell<FxHashMap<usize, Option<(usize, usize)>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(16, Default::default()));
    pub(super) static TL_IS_PRIME: RefCell<FxHashMap<usize, bool>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(16, Default::default()));
}

#[inline]
pub(crate) fn cached_coprime_factors(n: usize) -> Option<(usize, usize)> {
    if n < FLAT_CACHE_LIMIT {
        if let Some(v) = COPRIME_FACTORS_FLAT[n].get() {
            return *v;
        }
    } else if let Some(v) = TL_COPRIME_FACTORS.with(|c| c.borrow().get(&n).copied()) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().coprime_factors.tl_hit();
        return v;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().coprime_factors.global_hit();
    let v = {
        let maybe = COPRIME_FACTORS_CACHE.read().get(&n).copied();
        if let Some(v) = maybe {
            v
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().coprime_factors.miss();
            let result = coprime_factors(n);
            *COPRIME_FACTORS_CACHE.write().entry(n).or_insert(result)
        }
    };
    if n < FLAT_CACHE_LIMIT {
        COPRIME_FACTORS_FLAT[n].get_or_init(|| v);
    } else {
        TL_COPRIME_FACTORS.with(|c| c.borrow_mut().insert(n, v));
    }
    v
}

#[inline]
pub(crate) fn cached_is_prime(n: usize) -> bool {
    if n < FLAT_CACHE_LIMIT {
        if let Some(v) = IS_PRIME_FLAT[n].get() {
            return *v;
        }
    } else if let Some(v) = TL_IS_PRIME.with(|c| c.borrow().get(&n).copied()) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().is_prime.tl_hit();
        return v;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().is_prime.global_hit();
    let v = {
        let maybe = IS_PRIME_CACHE.read().get(&n).copied();
        if let Some(v) = maybe {
            v
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().is_prime.miss();
            let result = is_prime(n);
            *IS_PRIME_CACHE.write().entry(n).or_insert(result)
        }
    };
    if n < FLAT_CACHE_LIMIT {
        IS_PRIME_FLAT[n].get_or_init(|| v);
    } else {
        TL_IS_PRIME.with(|c| c.borrow_mut().insert(n, v));
    }
    v
}
