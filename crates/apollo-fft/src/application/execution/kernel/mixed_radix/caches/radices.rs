//! The 2-3-smooth radix factorization cache: one lowered radix list per length.

use super::super::super::radix_shape::factorize_composite as factorize_prime23;
use super::direct_mapped::FLAT_CACHE_LIMIT;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

static PRIME23_RADIX_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Option<Arc<[usize]>>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static PRIME23_RADIX_FLAT: [OnceLock<Option<Arc<[usize]>>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];

thread_local! {
    pub(super) static TL_PRIME23_RADIX: RefCell<FxHashMap<usize, Option<Arc<[usize]>>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
}

#[inline]
pub(crate) fn cached_prime23_radices(n: usize) -> Option<Arc<[usize]>> {
    if n < FLAT_CACHE_LIMIT {
        if let Some(radices) = PRIME23_RADIX_FLAT[n].get() {
            return radices.clone();
        }
    } else if let Some(radices) = TL_PRIME23_RADIX.with(|c| c.borrow().get(&n).cloned()) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().prime23_radix.tl_hit();
        return radices;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().prime23_radix.global_hit();
    let radices = {
        let maybe_cached = PRIME23_RADIX_CACHE.read().get(&n).cloned();
        if let Some(radices) = maybe_cached {
            radices
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().prime23_radix.miss();
            let new_radices = factorize_prime23(n).map(lower_and_cache_radices);
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
    if n < FLAT_CACHE_LIMIT {
        PRIME23_RADIX_FLAT[n].get_or_init(|| radices.clone());
    } else {
        TL_PRIME23_RADIX.with(|c| c.borrow_mut().insert(n, radices.clone()));
    }
    radices
}

#[inline]
fn lower_and_cache_radices(radices: Vec<usize>) -> Arc<[usize]> {
    Arc::from(radices.into_boxed_slice())
}
