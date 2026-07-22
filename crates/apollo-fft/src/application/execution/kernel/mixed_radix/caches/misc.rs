use super::super::super::radix_shape::{
    coprime_factors, factorize_composite as factorize_prime23, is_prime,
};
use super::direct_mapped::{
    bounded_directional_index, bounded_index, DIRECTIONAL_FLAT_CACHE_LIMIT, FLAT_CACHE_LIMIT,
};
use crate::application::execution::kernel::components::rader::generator::CanonicalRaderGenerator;
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

static PRIME23_RADIX_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Option<Arc<[usize]>>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static RADER_SPECTRUM_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<(usize, usize), Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static RADER_SPECTRUM_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<(usize, usize), Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static RADER_ORDER_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Arc<[usize]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static COPRIME_FACTORS_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<usize, Option<(usize, usize)>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static IS_PRIME_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, bool>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static PFA_PERM_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<(usize, usize), (Arc<[usize]>, Arc<[usize]>)>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
/// Negacyclic spectrum cache: (cyclic_spectrum, negacyclic_spectrum) per (n, inverse, g_inv).
type NegacyclicEntry<C> = (Arc<[C]>, Arc<[C]>);

static RADER_NEGACYCLIC_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<(usize, usize), NegacyclicEntry<Complex64>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static RADER_NEGACYCLIC_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<(usize, usize), NegacyclicEntry<Complex32>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static RADER_NEG_TWIDDLES_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<usize, Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static RADER_NEG_TWIDDLES_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<FxHashMap<usize, Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

static COPRIME_FACTORS_FLAT: [OnceLock<Option<(usize, usize)>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];
static IS_PRIME_FLAT: [OnceLock<bool>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];
static PRIME23_RADIX_FLAT: [OnceLock<Option<Arc<[usize]>>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];
static RADER_ORDER_FLAT: [OnceLock<Arc<[usize]>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];
static RADER_NEG_TWIDDLES_PRECISE_FLAT: [OnceLock<Arc<[Complex64]>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];
static RADER_NEG_TWIDDLES_REDUCED_FLAT: [OnceLock<Arc<[Complex32]>>; FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; FLAT_CACHE_LIMIT];
static RADER_SPECTRUM_PRECISE_FLAT: [OnceLock<Arc<[Complex64]>>; DIRECTIONAL_FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];
static RADER_SPECTRUM_REDUCED_FLAT: [OnceLock<Arc<[Complex32]>>; DIRECTIONAL_FLAT_CACHE_LIMIT] =
    [const { OnceLock::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];
static RADER_NEGACYCLIC_PRECISE_FLAT: [OnceLock<NegacyclicEntry<Complex64>>;
    DIRECTIONAL_FLAT_CACHE_LIMIT] = [const { OnceLock::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];
static RADER_NEGACYCLIC_REDUCED_FLAT: [OnceLock<NegacyclicEntry<Complex32>>;
    DIRECTIONAL_FLAT_CACHE_LIMIT] = [const { OnceLock::new() }; DIRECTIONAL_FLAT_CACHE_LIMIT];

thread_local! {
    pub(super) static TL_PRIME23_RADIX: RefCell<FxHashMap<usize, Option<Arc<[usize]>>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_SPECTRUM_PRECISE: RefCell<FxHashMap<(usize, usize), Arc<[Complex64]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_SPECTRUM_REDUCED: RefCell<FxHashMap<(usize, usize), Arc<[Complex32]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_ORDER: RefCell<FxHashMap<usize, Arc<[usize]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_COPRIME_FACTORS: RefCell<FxHashMap<usize, Option<(usize, usize)>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(16, Default::default()));
    pub(super) static TL_IS_PRIME: RefCell<FxHashMap<usize, bool>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(16, Default::default()));
    pub(super) static TL_PFA_PERM: RefCell<FxHashMap<(usize, usize), (Arc<[usize]>, Arc<[usize]>)>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_NEGACYCLIC_PRECISE: RefCell<FxHashMap<(usize, usize), NegacyclicEntry<Complex64>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    pub(super) static TL_RADER_NEGACYCLIC_REDUCED: RefCell<FxHashMap<(usize, usize), NegacyclicEntry<Complex32>>> =
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
    key: (usize, usize),
    val_precise: Arc<[Complex64]>,
    val_reduced: Arc<[Complex32]>,
    val_self: Arc<[Self]>,
    tl_get: rader_tl_get,
    tl_insert: rader_tl_insert,
    global: rader_global,
    global_ret_self: RwLock<FxHashMap<(usize, usize), Arc<[Self]>>>,
    tl_precise: TL_RADER_SPECTRUM_PRECISE,
    tl_reduced: TL_RADER_SPECTRUM_REDUCED,
    global_precise: RADER_SPECTRUM_PRECISE_CACHE,
    global_reduced: RADER_SPECTRUM_REDUCED_CACHE,
    flat_precise: RADER_SPECTRUM_PRECISE_FLAT,
    flat_reduced: RADER_SPECTRUM_REDUCED_FLAT,
    flat_index: |key: (usize, usize)| bounded_directional_index(key.0, key.1),
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
pub(crate) fn lower_and_cache_radices(radices: Vec<usize>) -> Arc<[usize]> {
    Arc::from(radices.into_boxed_slice())
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

/// Return precomputed Good-Thomas input and output CRT permutation tables for
/// a pair of coprime factors `(n1, n2)`.
///
/// `input_perm[i1 * n2 + i2]  = (i1 * n2 + i2 * n1) % n` — gather index for step 1.
/// `output_perm[k2 * n1 + k1] = (k1 * n2 * inv_n2_n1 + k2 * n1 * inv_n1_n2) % n` — scatter index for step 5.
///
/// Tables are computed once on first use and shared across threads via `Arc`.
#[inline]
pub(crate) fn cached_pfa_perm(n1: usize, n2: usize) -> (Arc<[usize]>, Arc<[usize]>) {
    let key = (n1, n2);
    if let Some(v) = TL_PFA_PERM.with(|c| c.borrow().get(&key).cloned()) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().pfa_perm.tl_hit();
        return v;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().pfa_perm.global_hit();
    let v = {
        let maybe_cached = PFA_PERM_CACHE.read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().pfa_perm.miss();
            let pair = build_pfa_perm(n1, n2);
            PFA_PERM_CACHE
                .write()
                .entry(key)
                .or_insert_with(|| pair.clone())
                .clone()
        }
    };
    TL_PFA_PERM.with(|c| c.borrow_mut().insert(key, v.clone()));
    v
}

fn extended_gcd(a: usize, b: usize) -> (usize, i64, i64) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x, y) = extended_gcd(b % a, a);
    (g, y - (b as i64 / a as i64) * x, x)
}

fn mod_inverse_local(a: usize, m: usize) -> usize {
    let (_, x, _) = extended_gcd(a, m);
    ((x % m as i64 + m as i64) % m as i64) as usize
}

fn build_pfa_perm(n1: usize, n2: usize) -> (Arc<[usize]>, Arc<[usize]>) {
    let n = n1 * n2;
    let inv_n2_n1 = mod_inverse_local(n2, n1);
    let inv_n1_n2 = mod_inverse_local(n1, n2);

    let mut input_perm = vec![0usize; n];
    let mut output_perm = vec![0usize; n];

    for i1 in 0..n1 {
        for i2 in 0..n2 {
            input_perm[i1 * n2 + i2] = (i1 * n2 + i2 * n1) % n;
        }
    }
    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let k_idx = (k1 * n2 * inv_n2_n1 + k2 * n1 * inv_n1_n2) % n;
            output_perm[k2 * n1 + k1] = k_idx;
        }
    }
    (Arc::from(input_perm), Arc::from(output_perm))
}

// Rader spectrum cache: dispatches via the sealed `RaderSpectrumStore` trait.
cached_fetch_arc! {
    fn pub(crate) cached_rader_spectrum<RaderSpectrumStore>(
        key: (usize, usize),
        build_fn: build_fn,
    ) -> Arc<[F]>
    using tl_get = rader_tl_get, tl_insert = rader_tl_insert, global = rader_global,
}

#[inline]
pub(crate) fn cached_rader_order(
    n: usize,
    generator: CanonicalRaderGenerator,
    build_fn: impl FnOnce(usize, usize) -> Vec<usize>,
) -> Arc<[usize]> {
    if let Some(index) = bounded_index(n) {
        if let Some(v) = RADER_ORDER_FLAT[index].get() {
            return Arc::clone(v);
        }
    }
    if let Some(v) = TL_RADER_ORDER.with(|c| c.borrow().get(&n).cloned()) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().rader_order.tl_hit();
        return v;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().rader_order.global_hit();
    let v = {
        let maybe_cached = RADER_ORDER_CACHE.read().get(&n).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().rader_order.miss();
            let order: Arc<[usize]> = Arc::from(build_fn(n, generator.root()));
            RADER_ORDER_CACHE
                .write()
                .entry(n)
                .or_insert_with(|| Arc::clone(&order))
                .clone()
        }
    };
    if let Some(index) = bounded_index(n) {
        drop(RADER_ORDER_FLAT[index].set(Arc::clone(&v)));
        return v;
    }
    TL_RADER_ORDER.with(|c| c.borrow_mut().insert(n, Arc::clone(&v)));
    v
}

// ── Negacyclic spectrum cache ────────────────────────────────────────────────

declare_cache_store! {
    sealed_mod: negacyclic_sealed,
    sealed_trait: NegacyclicSpectrumStoreSealed,
    store_trait: NegacyclicSpectrumStore,
    extra_bounds: [Clone, 'static],
    key: (usize, usize),
    val_precise: NegacyclicEntry<Complex64>,
    val_reduced: NegacyclicEntry<Complex32>,
    val_self: NegacyclicEntry<Self>,
    tl_get: neg_tl_get,
    tl_insert: neg_tl_insert,
    global: neg_global,
    global_ret_self: RwLock<FxHashMap<(usize, usize), NegacyclicEntry<Self>>>,
    tl_precise: TL_RADER_NEGACYCLIC_PRECISE,
    tl_reduced: TL_RADER_NEGACYCLIC_REDUCED,
    global_precise: RADER_NEGACYCLIC_PRECISE_CACHE,
    global_reduced: RADER_NEGACYCLIC_REDUCED_CACHE,
    flat_precise: RADER_NEGACYCLIC_PRECISE_FLAT,
    flat_reduced: RADER_NEGACYCLIC_REDUCED_FLAT,
    flat_index: |key: (usize, usize)| bounded_directional_index(key.0, key.1),
}

/// Generic negacyclic spectrum cache: dispatches to the correct concrete
/// thread-local and global RwLock cache via the sealed `NegacyclicSpectrumStore` trait.
#[inline]
pub(crate) fn cached_rader_negacyclic_spectra<F: NegacyclicSpectrumStore>(
    key: (usize, usize),
    build_fn: impl FnOnce((usize, usize)) -> (Vec<F>, Vec<F>),
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
    flat_index: bounded_index,
}

// Negacyclic twiddle cache: dispatches via the sealed `NegTwiddleStore` trait.
cached_fetch_arc! {
    fn pub(crate) cached_rader_neg_twiddles<NegTwiddleStore>(
        m: usize,
        build_fn: build_fn,
    ) -> Arc<[F]>
    using tl_get = neg_tw_tl_get, tl_insert = neg_tw_tl_insert, global = neg_tw_global,
}
