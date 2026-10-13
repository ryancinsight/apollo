//! The coprime-factor cache consulted by the plan dispatch.

use super::super::super::radix_shape::coprime_factors;
use super::tables::{shared_table, LocalTable, SharedTable};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::OnceLock;

const KEY_COMPONENT_BITS: u32 = 12;

/// Number of input lengths retained in the flat factor table.
const FLAT_CACHE_LIMIT: usize = 1 << KEY_COMPONENT_BITS;

static COPRIME_FACTORS_CACHE: SharedTable<usize, Option<(usize, usize)>> = shared_table();

static COPRIME_FACTORS_FLAT: [OnceLock<Option<(usize, usize)>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];

thread_local! {
    pub(super) static TL_COPRIME_FACTORS: LocalTable<usize, Option<(usize, usize)>> = RefCell::new(FxHashMap::with_capacity_and_hasher(16, Default::default()));
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
