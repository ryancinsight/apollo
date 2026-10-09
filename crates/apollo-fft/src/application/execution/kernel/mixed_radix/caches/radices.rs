//! The 2-3-smooth radix factorization cache: one lowered radix list per length.

use super::super::super::radix_shape::factorize_composite as factorize_prime23;
use super::tables::{shared_table, LocalTable, SharedTable};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{Arc, Weak};

type RadixIndex = Option<Weak<[usize]>>;
static PRIME23_RADIX_CACHE: SharedTable<usize, RadixIndex> = shared_table();

thread_local! {
    pub(super) static TL_PRIME23_RADIX: LocalTable<usize, RadixIndex> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
}

#[inline]
pub(crate) fn cached_prime23_radices(n: usize) -> Option<Arc<[usize]>> {
    if let Some(radices) = TL_PRIME23_RADIX
        .with(|cache| cache.borrow().get(&n).cloned())
        .and_then(upgrade_index)
    {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().prime23_radix.tl_hit();
        return radices;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().prime23_radix.global_hit();
    let radices = {
        let maybe_cached = PRIME23_RADIX_CACHE
            .read()
            .get(&n)
            .cloned()
            .and_then(upgrade_index);
        if let Some(radices) = maybe_cached {
            radices
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().prime23_radix.miss();
            let new_radices = factorize_prime23(n).map(lower_and_cache_radices);
            let mut index = PRIME23_RADIX_CACHE.write();
            if let Some(radices) = index.get(&n).cloned().and_then(upgrade_index) {
                radices
            } else {
                index.insert(n, downgrade_index(&new_radices));
                new_radices
            }
        }
    };
    TL_PRIME23_RADIX.with(|cache| cache.borrow_mut().insert(n, downgrade_index(&radices)));
    radices
}

fn upgrade_index(index: RadixIndex) -> Option<Option<Arc<[usize]>>> {
    match index {
        Some(radices) => radices.upgrade().map(Some),
        None => Some(None),
    }
}

fn downgrade_index(radices: &Option<Arc<[usize]>>) -> RadixIndex {
    radices.as_ref().map(Arc::downgrade)
}

#[inline]
fn lower_and_cache_radices(radices: Vec<usize>) -> Arc<[usize]> {
    Arc::from(radices.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::cached_prime23_radices;
    use std::sync::Arc;

    #[test]
    fn index_reuses_live_factorization_and_releases_after_drop() {
        let first = cached_prime23_radices(539).expect("7 * 7 * 11 is supported");
        let second = cached_prime23_radices(539).expect("the factorization remains supported");
        assert!(Arc::ptr_eq(&first, &second));
        let released = Arc::downgrade(&first);
        drop(first);
        drop(second);
        assert!(released.upgrade().is_none());
    }
}
