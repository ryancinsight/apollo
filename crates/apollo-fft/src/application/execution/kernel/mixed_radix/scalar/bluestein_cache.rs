//! Precision-generic storage for Bluestein kernel spectra.

use super::trait_def::{BluesteinEntry, BluesteinKey, BluesteinStore};
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{LazyLock, OnceLock};

const FLAT_CACHE_LIMIT: usize = 4096;
const DIRECTIONAL_FLAT_CACHE_LIMIT: usize = 2 * FLAT_CACHE_LIMIT;
const SPARSE_INITIAL_CAPACITY: usize = 8;

type Cache<C> = RwLock<FxHashMap<BluesteinKey, BluesteinEntry<C>>>;
type FlatCache<C> = [OnceLock<BluesteinEntry<C>>; DIRECTIONAL_FLAT_CACHE_LIMIT];

static REDUCED_CACHE: LazyLock<Cache<Complex32>> =
    LazyLock::new(|| RwLock::new(FxHashMap::default()));
static PRECISE_CACHE: LazyLock<Cache<Complex64>> =
    LazyLock::new(|| RwLock::new(FxHashMap::default()));
static REDUCED_FLAT: FlatCache<Complex32> =
    [const { OnceLock::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];
static PRECISE_FLAT: FlatCache<Complex64> =
    [const { OnceLock::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];

thread_local! {
    static REDUCED_SPARSE: RefCell<FxHashMap<BluesteinKey, BluesteinEntry<Complex32>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(
            SPARSE_INITIAL_CAPACITY,
            Default::default(),
        ));
    static PRECISE_SPARSE: RefCell<FxHashMap<BluesteinKey, BluesteinEntry<Complex64>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(
            SPARSE_INITIAL_CAPACITY,
            Default::default(),
        ));
}

trait CacheSpec: Copy + 'static {
    type Complex: Copy + Send + Sync + 'static;

    fn flat() -> &'static FlatCache<Self::Complex>;
    fn sparse_get(key: BluesteinKey) -> Option<BluesteinEntry<Self::Complex>>;
    fn sparse_insert(key: BluesteinKey, value: BluesteinEntry<Self::Complex>);
    fn cache() -> &'static Cache<Self::Complex>;

    #[cfg(feature = "cache-profiling")]
    fn record_sparse_hit();
}

impl CacheSpec for f32 {
    type Complex = Complex32;

    fn flat() -> &'static FlatCache<Self::Complex> {
        &REDUCED_FLAT
    }

    fn sparse_get(key: BluesteinKey) -> Option<BluesteinEntry<Self::Complex>> {
        REDUCED_SPARSE.with(|cache| cache.borrow().get(&key).cloned())
    }

    fn sparse_insert(key: BluesteinKey, value: BluesteinEntry<Self::Complex>) {
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

    fn flat() -> &'static FlatCache<Self::Complex> {
        &PRECISE_FLAT
    }

    fn sparse_get(key: BluesteinKey) -> Option<BluesteinEntry<Self::Complex>> {
        PRECISE_SPARSE.with(|cache| cache.borrow().get(&key).cloned())
    }

    fn sparse_insert(key: BluesteinKey, value: BluesteinEntry<Self::Complex>) {
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
fn get<T: CacheSpec>(key: BluesteinKey) -> Option<BluesteinEntry<T::Complex>> {
    let (length, inverse, _) = key;
    if length < FLAT_CACHE_LIMIT {
        let index = (length << 1) | usize::from(inverse);
        T::flat()[index].get().cloned()
    } else {
        let result = T::sparse_get(key);
        #[cfg(feature = "cache-profiling")]
        if result.is_some() {
            T::record_sparse_hit();
        }
        result
    }
}

#[inline]
fn insert<T: CacheSpec>(key: BluesteinKey, value: BluesteinEntry<T::Complex>) {
    let (length, inverse, _) = key;
    if length < FLAT_CACHE_LIMIT {
        let index = (length << 1) | usize::from(inverse);
        T::flat()[index].get_or_init(|| value);
    } else {
        T::sparse_insert(key, value);
    }
}

impl BluesteinStore for f32 {
    type Cpx = Complex32;

    #[inline]
    fn tl_get(key: BluesteinKey) -> Option<BluesteinEntry<Self::Cpx>> {
        get::<Self>(key)
    }

    #[inline]
    fn tl_insert(key: BluesteinKey, value: BluesteinEntry<Self::Cpx>) {
        insert::<Self>(key, value);
    }

    #[inline]
    fn global() -> &'static Cache<Self::Cpx> {
        Self::cache()
    }
}

impl BluesteinStore for f64 {
    type Cpx = Complex64;

    #[inline]
    fn tl_get(key: BluesteinKey) -> Option<BluesteinEntry<Self::Cpx>> {
        get::<Self>(key)
    }

    #[inline]
    fn tl_insert(key: BluesteinKey, value: BluesteinEntry<Self::Cpx>) {
        insert::<Self>(key, value);
    }

    #[inline]
    fn global() -> &'static Cache<Self::Cpx> {
        Self::cache()
    }
}
