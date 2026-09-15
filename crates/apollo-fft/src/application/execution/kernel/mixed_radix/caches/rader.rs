//! The Rader caches: generator orders, cyclic and negacyclic convolution
//! spectra, and negacyclic twiddles, per precision.

use super::direct_mapped::{
    bounded_directional_coordinates, bounded_pair_coordinates, DirectMappedSlot,
    DIRECTIONAL_FLAT_CACHE_LIMIT, FLAT_CACHE_LIMIT,
};
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

static RADER_SPECTRUM_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<(usize, usize, usize), Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static RADER_SPECTRUM_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<(usize, usize, usize), Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static RADER_ORDER_CACHE: std::sync::LazyLock<RwLock<FxHashMap<(usize, usize), Arc<[usize]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

/// Negacyclic spectrum cache: (cyclic_spectrum, negacyclic_spectrum) per (n, inverse, g_inv).
type NegacyclicEntry<C> = (Arc<[C]>, Arc<[C]>);

static RADER_NEGACYCLIC_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<(usize, usize, usize), NegacyclicEntry<Complex64>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static RADER_NEGACYCLIC_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<(usize, usize, usize), NegacyclicEntry<Complex32>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static RADER_NEG_TWIDDLES_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<usize, Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static RADER_NEG_TWIDDLES_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<usize, Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static RADER_ORDER_FLAT: [DirectMappedSlot<usize, Arc<[usize]>>; FLAT_CACHE_LIMIT] =
    [const { DirectMappedSlot::new() }; FLAT_CACHE_LIMIT];

static RADER_NEG_TWIDDLES_PRECISE_FLAT: [OnceLock<Arc<[Complex64]>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];

static RADER_NEG_TWIDDLES_REDUCED_FLAT: [OnceLock<Arc<[Complex32]>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];

static RADER_SPECTRUM_PRECISE_FLAT: [DirectMappedSlot<usize, Arc<[Complex64]>>;
    DIRECTIONAL_FLAT_CACHE_LIMIT] =
    [const { DirectMappedSlot::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];

static RADER_SPECTRUM_REDUCED_FLAT: [DirectMappedSlot<usize, Arc<[Complex32]>>;
    DIRECTIONAL_FLAT_CACHE_LIMIT] =
    [const { DirectMappedSlot::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];

static RADER_NEGACYCLIC_PRECISE_FLAT: [DirectMappedSlot<usize, NegacyclicEntry<Complex64>>;
    DIRECTIONAL_FLAT_CACHE_LIMIT] =
    [const { DirectMappedSlot::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];

static RADER_NEGACYCLIC_REDUCED_FLAT: [DirectMappedSlot<usize, NegacyclicEntry<Complex32>>;
    DIRECTIONAL_FLAT_CACHE_LIMIT] =
    [const { DirectMappedSlot::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];

thread_local! {
    pub(super) static TL_RADER_SPECTRUM_PRECISE: RefCell<FxHashMap<(usize, usize, usize), Arc<[Complex64]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_SPECTRUM_REDUCED: RefCell<FxHashMap<(usize, usize, usize), Arc<[Complex32]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_ORDER: RefCell<FxHashMap<(usize, usize), Arc<[usize]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_NEGACYCLIC_PRECISE: RefCell<FxHashMap<(usize, usize, usize), NegacyclicEntry<Complex64>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_NEGACYCLIC_REDUCED: RefCell<FxHashMap<(usize, usize, usize), NegacyclicEntry<Complex32>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_NEG_TWIDDLES_PRECISE: RefCell<FxHashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_NEG_TWIDDLES_REDUCED: RefCell<FxHashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
}

declare_cache_store! {
    sealed_mod: sealed,
    sealed_trait: RaderSpectrumStoreSealed,
    store_trait: RaderSpectrumStore,
    extra_bounds: [Clone, 'static],
    key: (usize, usize, usize),
    val_precise: Arc<[Complex64]>,
    val_reduced: Arc<[Complex32]>,
    val_self: Arc<[Self]>,
    tl_get: rader_tl_get,
    tl_insert: rader_tl_insert,
    global: rader_global,
    global_ret_self: RwLock<FxHashMap<(usize, usize, usize), Arc<[Self]>>>,
    tl_precise: TL_RADER_SPECTRUM_PRECISE,
    tl_reduced: TL_RADER_SPECTRUM_REDUCED,
    global_precise: RADER_SPECTRUM_PRECISE_CACHE,
    global_reduced: RADER_SPECTRUM_REDUCED_CACHE,
    flat_precise: RADER_SPECTRUM_PRECISE_FLAT,
    flat_reduced: RADER_SPECTRUM_REDUCED_FLAT,
    flat_coordinates: |key: (usize, usize, usize)| bounded_directional_coordinates(key.0, key.1, key.2),
}

// Rader spectrum cache: dispatches via the sealed `RaderSpectrumStore` trait.
cached_fetch_arc! {
    fn pub(crate) cached_rader_spectrum<RaderSpectrumStore>(
        key: (usize, usize, usize),
        build_fn: build_fn,
    ) -> Arc<[F]>
    using tl_get = rader_tl_get, tl_insert = rader_tl_insert, global = rader_global,
}

#[inline]
pub(crate) fn cached_rader_order(
    key: (usize, usize),
    build_fn: impl FnOnce((usize, usize)) -> Vec<usize>,
) -> Arc<[usize]> {
    let coordinates = bounded_pair_coordinates(key.0, key.1);
    if let Some((index, tag)) = coordinates {
        if let Some(v) = RADER_ORDER_FLAT[index].get(tag) {
            return v;
        }
    }
    if let Some(v) = TL_RADER_ORDER.with(|c| c.borrow().get(&key).cloned()) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().rader_order.tl_hit();
        return v;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().rader_order.global_hit();
    let v = {
        let maybe_cached = RADER_ORDER_CACHE.read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().rader_order.miss();
            let order: Arc<[usize]> = Arc::from(build_fn(key));
            RADER_ORDER_CACHE
                .write()
                .entry(key)
                .or_insert_with(|| Arc::clone(&order))
                .clone()
        }
    };
    if let Some((index, tag)) = coordinates {
        if RADER_ORDER_FLAT[index].insert(tag, Arc::clone(&v)).is_ok() {
            return v;
        }
    }
    TL_RADER_ORDER.with(|c| c.borrow_mut().insert(key, Arc::clone(&v)));
    v
}

// ── Negacyclic spectrum cache ────────────────────────────────────────────────

declare_cache_store! {
    sealed_mod: negacyclic_sealed,
    sealed_trait: NegacyclicSpectrumStoreSealed,
    store_trait: NegacyclicSpectrumStore,
    extra_bounds: [Clone, 'static],
    key: (usize, usize, usize),
    val_precise: NegacyclicEntry<Complex64>,
    val_reduced: NegacyclicEntry<Complex32>,
    val_self: NegacyclicEntry<Self>,
    tl_get: neg_tl_get,
    tl_insert: neg_tl_insert,
    global: neg_global,
    global_ret_self: RwLock<FxHashMap<(usize, usize, usize), NegacyclicEntry<Self>>>,
    tl_precise: TL_RADER_NEGACYCLIC_PRECISE,
    tl_reduced: TL_RADER_NEGACYCLIC_REDUCED,
    global_precise: RADER_NEGACYCLIC_PRECISE_CACHE,
    global_reduced: RADER_NEGACYCLIC_REDUCED_CACHE,
    flat_precise: RADER_NEGACYCLIC_PRECISE_FLAT,
    flat_reduced: RADER_NEGACYCLIC_REDUCED_FLAT,
    flat_coordinates: |key: (usize, usize, usize)| bounded_directional_coordinates(key.0, key.1, key.2),
}

/// Generic negacyclic spectrum cache: dispatches to the correct concrete
/// thread-local and global RwLock cache via the sealed `NegacyclicSpectrumStore` trait.
#[inline]
pub(crate) fn cached_rader_negacyclic_spectra<F: NegacyclicSpectrumStore>(
    key: (usize, usize, usize),
    build_fn: impl FnOnce((usize, usize, usize)) -> (Vec<F>, Vec<F>),
) -> NegacyclicEntry<F> {
    if let Some(v) = F::neg_tl_get(key) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().rader_negacyclic_precise.tl_hit();
        return v;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().rader_negacyclic_precise.global_hit();
    let v = {
        let maybe_cached = F::neg_global().read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().rader_negacyclic_precise.miss();
            let (cyc, neg) = build_fn(key);
            let entry: NegacyclicEntry<F> = (Arc::from(cyc), Arc::from(neg));
            F::neg_global()
                .write()
                .entry(key)
                .or_insert_with(|| entry.clone())
                .clone()
        }
    };
    F::neg_tl_insert(key, v.clone());
    v
}

// ── Negacyclic twiddle cache ─────────────────────────────────────────────────

declare_cache_store! {
    sealed_mod: neg_twiddle_sealed,
    sealed_trait: NegTwiddleStoreSealed,
    store_trait: NegTwiddleStore,
    extra_bounds: [Clone, 'static],
    key: usize,
    val_precise: Arc<[Complex64]>,
    val_reduced: Arc<[Complex32]>,
    val_self: Arc<[Self]>,
    tl_get: neg_tw_tl_get,
    tl_insert: neg_tw_tl_insert,
    global: neg_tw_global,
    global_ret_self: RwLock<FxHashMap<usize, Arc<[Self]>>>,
    tl_precise: TL_RADER_NEG_TWIDDLES_PRECISE,
    tl_reduced: TL_RADER_NEG_TWIDDLES_REDUCED,
    global_precise: RADER_NEG_TWIDDLES_PRECISE_CACHE,
    global_reduced: RADER_NEG_TWIDDLES_REDUCED_CACHE,
    flat_precise: RADER_NEG_TWIDDLES_PRECISE_FLAT,
    flat_reduced: RADER_NEG_TWIDDLES_REDUCED_FLAT,
    flat_coordinates: |key: usize| (key < FLAT_CACHE_LIMIT).then_some((key, 0)),
}

// Negacyclic twiddle cache: dispatches via the sealed `NegTwiddleStore` trait.
cached_fetch_arc! {
    fn pub(crate) cached_rader_neg_twiddles<NegTwiddleStore>(
        m: usize,
        build_fn: build_fn,
    ) -> Arc<[F]>
    using tl_get = neg_tw_tl_get, tl_insert = neg_tw_tl_insert, global = neg_tw_global,
}
