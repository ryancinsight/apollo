//! Precision-generic storage for Bluestein kernel spectra.

use super::trait_def::{BluesteinIndex, BluesteinKey, BluesteinStore};
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{Arc, LazyLock};

const SPARSE_INITIAL_CAPACITY: usize = 8;

type Cache<C> = RwLock<FxHashMap<BluesteinKey, BluesteinIndex<C>>>;

static REDUCED_CACHE: LazyLock<Cache<Complex32>> =
    LazyLock::new(|| RwLock::new(FxHashMap::default()));
static PRECISE_CACHE: LazyLock<Cache<Complex64>> =
    LazyLock::new(|| RwLock::new(FxHashMap::default()));
thread_local! {
    static REDUCED_SPARSE: RefCell<FxHashMap<BluesteinKey, BluesteinIndex<Complex32>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(
            SPARSE_INITIAL_CAPACITY,
            Default::default(),
        ));
    static PRECISE_SPARSE: RefCell<FxHashMap<BluesteinKey, BluesteinIndex<Complex64>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(
            SPARSE_INITIAL_CAPACITY,
            Default::default(),
        ));
}

trait CacheSpec: Copy + 'static {
    type Complex: Copy + Send + Sync + 'static;

    fn sparse_get(key: BluesteinKey) -> Option<BluesteinIndex<Self::Complex>>;
    fn sparse_insert(key: BluesteinKey, value: BluesteinIndex<Self::Complex>);
    fn cache() -> &'static Cache<Self::Complex>;

    #[cfg(feature = "cache-profiling")]
    fn record_sparse_hit();
}

impl CacheSpec for f32 {
    type Complex = Complex32;

    fn sparse_get(key: BluesteinKey) -> Option<BluesteinIndex<Self::Complex>> {
        REDUCED_SPARSE.with(|cache| cache.borrow().get(&key).cloned())
    }

    fn sparse_insert(key: BluesteinKey, value: BluesteinIndex<Self::Complex>) {
        REDUCED_SPARSE.with(|cache| cache.borrow_mut().insert(key, value));
    }

    fn cache() -> &'static Cache<Self::Complex> {
        &REDUCED_CACHE
    }

    #[cfg(feature = "cache-profiling")]
    fn record_sparse_hit() {
        crate::application::execution::kernel::mixed_radix::caches::profiler::get()
            .bluestein_reduced
            .tl_hit();
    }
}

impl CacheSpec for f64 {
    type Complex = Complex64;

    fn sparse_get(key: BluesteinKey) -> Option<BluesteinIndex<Self::Complex>> {
        PRECISE_SPARSE.with(|cache| cache.borrow().get(&key).cloned())
    }

    fn sparse_insert(key: BluesteinKey, value: BluesteinIndex<Self::Complex>) {
        PRECISE_SPARSE.with(|cache| cache.borrow_mut().insert(key, value));
    }

    fn cache() -> &'static Cache<Self::Complex> {
        &PRECISE_CACHE
    }

    #[cfg(feature = "cache-profiling")]
    fn record_sparse_hit() {
        crate::application::execution::kernel::mixed_radix::caches::profiler::get()
            .bluestein_precise
            .tl_hit();
    }
}

#[inline]
fn get<T: CacheSpec>(key: BluesteinKey) -> Option<Arc<[T::Complex]>> {
    let result = T::sparse_get(key).and_then(|entry| entry.upgrade());
    #[cfg(feature = "cache-profiling")]
    if result.is_some() {
        T::record_sparse_hit();
    }
    result
}

#[inline]
#[cfg(test)]
fn insert<T: CacheSpec>(key: BluesteinKey, value: &Arc<[T::Complex]>) {
    T::sparse_insert(key, Arc::downgrade(value));
}

impl BluesteinStore for f32 {
    type Cpx = Complex32;

    #[inline]
    fn tl_get(key: BluesteinKey) -> Option<BluesteinIndex<Self::Cpx>> {
        get::<Self>(key).map(|entry| Arc::downgrade(&entry))
    }

    #[inline]
    fn tl_insert(key: BluesteinKey, value: BluesteinIndex<Self::Cpx>) {
        Self::sparse_insert(key, value);
    }

    #[inline]
    fn global() -> &'static Cache<Self::Cpx> {
        Self::cache()
    }
}

impl BluesteinStore for f64 {
    type Cpx = Complex64;

    #[inline]
    fn tl_get(key: BluesteinKey) -> Option<BluesteinIndex<Self::Cpx>> {
        get::<Self>(key).map(|entry| Arc::downgrade(&entry))
    }

    #[inline]
    fn tl_insert(key: BluesteinKey, value: BluesteinIndex<Self::Cpx>) {
        Self::sparse_insert(key, value);
    }

    #[inline]
    fn global() -> &'static Cache<Self::Cpx> {
        Self::cache()
    }
}

#[cfg(test)]
mod tests {
    use super::{get, insert};
    use eunomia::Complex64;
    use std::sync::Arc;

    #[test]
    fn weak_index_distinguishes_generator_inverse_and_releases() {
        let first_key = (4000, false, 1);
        let second_key = (4000, false, 2);
        let first = Arc::<[Complex64]>::from([Complex64::new(1.0, 2.0)]);
        let second = Arc::<[Complex64]>::from([Complex64::new(3.0, 4.0)]);

        insert::<f64>(first_key, &first);
        insert::<f64>(second_key, &second);

        assert_eq!(get::<f64>(first_key), Some(Arc::clone(&first)));
        assert_eq!(get::<f64>(second_key), Some(Arc::clone(&second)));

        let released = Arc::downgrade(&first);
        drop(first);
        assert!(released.upgrade().is_none());
        assert!(get::<f64>(first_key).is_none());
    }
}
