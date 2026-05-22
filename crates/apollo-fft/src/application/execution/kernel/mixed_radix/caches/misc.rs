use super::super::super::radix_shape::{
    coprime_factors, factorize_composite as factorize_prime23, is_prime,
    lower_radix2_pairs_to_radix4,
};
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

static PRIME23_RADIX_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Option<Arc<[usize]>>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_SPECTRUM_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_SPECTRUM_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_ORDER_CACHE: std::sync::LazyLock<RwLock<HashMap<(usize, usize), Arc<[usize]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static COPRIME_FACTORS_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Option<(usize, usize)>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static IS_PRIME_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, bool>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static PFA_PERM_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize), (Arc<[usize]>, Arc<[usize]>)>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static PFA_CYCLES_CACHE: std::sync::LazyLock<RwLock<HashMap<(usize, usize), Arc<[usize]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

/// Negacyclic spectrum cache: (cyclic_spectrum, negacyclic_spectrum) per (n, inverse, g_inv).
type NegacyclicEntry<C> = (Arc<[C]>, Arc<[C]>);

static RADER_NEGACYCLIC_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), NegacyclicEntry<Complex64>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_NEGACYCLIC_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize, usize), NegacyclicEntry<Complex32>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_NEG_TWIDDLES_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<HashMap<usize, Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static RADER_NEG_TWIDDLES_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<HashMap<usize, Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    pub(super) static TL_PRIME23_RADIX: RefCell<HashMap<usize, Option<Arc<[usize]>>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_SPECTRUM_PRECISE: RefCell<HashMap<(usize, usize, usize), Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_SPECTRUM_REDUCED: RefCell<HashMap<(usize, usize, usize), Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_ORDER: RefCell<HashMap<(usize, usize), Arc<[usize]>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_COPRIME_FACTORS: RefCell<HashMap<usize, Option<(usize, usize)>>> =
        RefCell::new(HashMap::with_capacity(16));
    pub(super) static TL_IS_PRIME: RefCell<HashMap<usize, bool>> =
        RefCell::new(HashMap::with_capacity(16));
    pub(super) static TL_PFA_PERM: RefCell<HashMap<(usize, usize), (Arc<[usize]>, Arc<[usize]>)>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_PFA_CYCLES: RefCell<HashMap<(usize, usize), Arc<[usize]>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_NEGACYCLIC_PRECISE: RefCell<HashMap<(usize, usize, usize), NegacyclicEntry<Complex64>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_NEGACYCLIC_REDUCED: RefCell<HashMap<(usize, usize, usize), NegacyclicEntry<Complex32>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_NEG_TWIDDLES_PRECISE: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    pub(super) static TL_RADER_NEG_TWIDDLES_REDUCED: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
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
    global_ret_self: RwLock<HashMap<(usize, usize, usize), Arc<[Self]>>>,
    tl_precise: TL_RADER_SPECTRUM_PRECISE,
    tl_reduced: TL_RADER_SPECTRUM_REDUCED,
    global_precise: RADER_SPECTRUM_PRECISE_CACHE,
    global_reduced: RADER_SPECTRUM_REDUCED_CACHE,
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
    TL_PRIME23_RADIX.with(|c| c.borrow_mut().insert(n, radices.clone()));
    radices
}

#[inline]
pub(crate) fn lower_and_cache_radices(radices: Vec<usize>) -> Arc<[usize]> {
    Arc::from(lower_radix2_pairs_to_radix4(&radices).into_boxed_slice())
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
        return v;
    }
    let v = {
        let maybe_cached = PFA_PERM_CACHE.read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
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

/// Return precomputed Good-Thomas input permutation cycles for in-place
/// application without runtime cycle-finding.
///
/// Returns a flat array of cycle data: [len1, pos1_0, pos1_1, ..., len2, pos2_0, ...]
/// where each cycle's positions are listed in order. The permutation is applied
/// by rotating each cycle's values by one position (left rotation).
///
/// Tables are computed once on first use and shared across threads via `Arc`.
#[inline]
pub(crate) fn cached_pfa_input_cycles(n1: usize, n2: usize) -> Arc<[usize]> {
    let key = (n1, n2);
    if let Some(v) = TL_PFA_CYCLES.with(|c| c.borrow().get(&key).cloned()) {
        return v;
    }
    let v = {
        let maybe_cached = PFA_CYCLES_CACHE.read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
            let cycles = build_pfa_input_cycles(n1, n2);
            PFA_CYCLES_CACHE
                .write()
                .entry(key)
                .or_insert_with(|| cycles.clone())
                .clone()
        }
    };
    TL_PFA_CYCLES.with(|c| c.borrow_mut().insert(key, v.clone()));
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

/// Build flat cycle representation for PFA input permutation.
/// Format: [len1, pos1_0, pos1_1, ..., len2, pos2_0, ...]
/// Fixed points (len=1) are stored but skipped during application.
///
/// Reuses cached_pfa_perm to avoid duplicate computation on first call.
fn build_pfa_input_cycles(n1: usize, n2: usize) -> Arc<[usize]> {
    let n = n1 * n2;
    // Reuse cached permutation to avoid duplicate work on first call
    let input_perm = cached_pfa_perm(n1, n2).0;

    let mut cycles = Vec::new();
    let mut visited = vec![false; n];

    for i in 0..n {
        if visited[i] {
            continue;
        }

        // Collect this cycle
        let mut cycle = Vec::new();
        let mut j = i;
        loop {
            visited[j] = true;
            cycle.push(j);
            let target = input_perm[j];
            if visited[target] {
                break;
            }
            j = target;
        }

        // Store cycle length followed by all positions
        cycles.push(cycle.len());
        cycles.extend(cycle);
    }

    Arc::from(cycles)
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
    if let Some(v) = TL_RADER_ORDER.with(|c| c.borrow().get(&key).cloned()) {
        return v;
    }
    let v = {
        let maybe_cached = RADER_ORDER_CACHE.read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
            let order: Arc<[usize]> = Arc::from(build_fn(key));
            RADER_ORDER_CACHE
                .write()
                .entry(key)
                .or_insert_with(|| Arc::clone(&order))
                .clone()
        }
    };
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
    global_ret_self: RwLock<HashMap<(usize, usize, usize), NegacyclicEntry<Self>>>,
    tl_precise: TL_RADER_NEGACYCLIC_PRECISE,
    tl_reduced: TL_RADER_NEGACYCLIC_REDUCED,
    global_precise: RADER_NEGACYCLIC_PRECISE_CACHE,
    global_reduced: RADER_NEGACYCLIC_REDUCED_CACHE,
}

/// Generic negacyclic spectrum cache: dispatches to the correct concrete
/// thread-local and global RwLock cache via the sealed `NegacyclicSpectrumStore` trait.
#[inline]
pub(crate) fn cached_rader_negacyclic_spectra<F: NegacyclicSpectrumStore>(
    key: (usize, usize, usize),
    build_fn: impl FnOnce((usize, usize, usize)) -> (Vec<F>, Vec<F>),
) -> NegacyclicEntry<F> {
    if let Some(v) = F::neg_tl_get(key) {
        return v;
    }
    let v = {
        let maybe_cached = F::neg_global().read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
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
    global_ret_self: RwLock<HashMap<usize, Arc<[Self]>>>,
    tl_precise: TL_RADER_NEG_TWIDDLES_PRECISE,
    tl_reduced: TL_RADER_NEG_TWIDDLES_REDUCED,
    global_precise: RADER_NEG_TWIDDLES_PRECISE_CACHE,
    global_reduced: RADER_NEG_TWIDDLES_REDUCED_CACHE,
}

// Negacyclic twiddle cache: dispatches via the sealed `NegTwiddleStore` trait.
cached_fetch_arc! {
    fn pub(crate) cached_rader_neg_twiddles<NegTwiddleStore>(
        m: usize,
        build_fn: build_fn,
    ) -> Arc<[F]>
    using tl_get = neg_tw_tl_get, tl_insert = neg_tw_tl_insert, global = neg_tw_global,
}
