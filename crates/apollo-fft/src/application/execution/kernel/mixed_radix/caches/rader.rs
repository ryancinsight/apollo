//! Weak indexes for Rader permutation and convolution tables.

use super::tables::{shared_table, LocalTable, SharedTable};
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{Arc, Weak};

type DirectionalKey = (usize, usize, usize);
type OrderKey = (usize, usize);

static RADER_SPECTRUM_PRECISE_CACHE: SharedTable<DirectionalKey, Weak<[Complex64]>> =
    shared_table();
static RADER_SPECTRUM_REDUCED_CACHE: SharedTable<DirectionalKey, Weak<[Complex32]>> =
    shared_table();
static RADER_ORDER_CACHE: SharedTable<OrderKey, Weak<[usize]>> = shared_table();

pub(crate) type NegacyclicEntry<C> = (Arc<[C]>, Arc<[C]>);
type NegacyclicIndex<C> = (Weak<[C]>, Weak<[C]>);

static RADER_NEGACYCLIC_PRECISE_CACHE: SharedTable<DirectionalKey, NegacyclicIndex<Complex64>> =
    shared_table();
static RADER_NEGACYCLIC_REDUCED_CACHE: SharedTable<DirectionalKey, NegacyclicIndex<Complex32>> =
    shared_table();
static RADER_NEG_TWIDDLES_PRECISE_CACHE: SharedTable<usize, Weak<[Complex64]>> = shared_table();
static RADER_NEG_TWIDDLES_REDUCED_CACHE: SharedTable<usize, Weak<[Complex32]>> = shared_table();

thread_local! {
    static TL_RADER_SPECTRUM_PRECISE: LocalTable<DirectionalKey, Weak<[Complex64]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_RADER_SPECTRUM_REDUCED: LocalTable<DirectionalKey, Weak<[Complex32]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_RADER_ORDER: LocalTable<OrderKey, Weak<[usize]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_RADER_NEGACYCLIC_PRECISE: LocalTable<DirectionalKey, NegacyclicIndex<Complex64>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_RADER_NEGACYCLIC_REDUCED: LocalTable<DirectionalKey, NegacyclicIndex<Complex32>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_RADER_NEG_TWIDDLES_PRECISE: LocalTable<usize, Weak<[Complex64]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_RADER_NEG_TWIDDLES_REDUCED: LocalTable<usize, Weak<[Complex32]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
}

declare_cache_store! {
    sealed_mod: spectrum_sealed,
    sealed_trait: RaderSpectrumStoreSealed,
    store_trait: RaderSpectrumStore,
    extra_bounds: [Copy, Send, Sync, 'static],
    key: DirectionalKey,
    val_precise: Weak<[Complex64]>,
    val_reduced: Weak<[Complex32]>,
    val_self: Weak<[Self]>,
    tl_get: local_get,
    tl_insert: local_insert,
    global: global,
    global_ret_self: RwLock<FxHashMap<DirectionalKey, Weak<[Self]>>>,
    tl_precise: TL_RADER_SPECTRUM_PRECISE,
    tl_reduced: TL_RADER_SPECTRUM_REDUCED,
    global_precise: RADER_SPECTRUM_PRECISE_CACHE,
    global_reduced: RADER_SPECTRUM_REDUCED_CACHE,
}

pub(crate) fn cached_rader_spectrum<F: RaderSpectrumStore>(
    key: DirectionalKey,
    build: impl FnOnce(DirectionalKey) -> Vec<F>,
) -> Arc<[F]> {
    if let Some(value) = F::local_get(key).and_then(|entry| entry.upgrade()) {
        return value;
    }
    if let Some(value) = F::global().read().get(&key).and_then(Weak::upgrade) {
        F::local_insert(key, Arc::downgrade(&value));
        return value;
    }
    let built: Arc<[F]> = build(key).into();
    let value = replace_expired(F::global(), key, built);
    F::local_insert(key, Arc::downgrade(&value));
    value
}

fn replace_expired<K: Copy + Eq + std::hash::Hash, T: ?Sized>(
    index: &RwLock<FxHashMap<K, Weak<T>>>,
    key: K,
    built: Arc<T>,
) -> Arc<T> {
    let mut index = index.write();
    if let Some(value) = index.get(&key).and_then(Weak::upgrade) {
        value
    } else {
        index.insert(key, Arc::downgrade(&built));
        built
    }
}

#[inline]
pub(crate) fn cached_rader_order(
    key: OrderKey,
    build: impl FnOnce(OrderKey) -> Vec<usize>,
) -> Arc<[usize]> {
    if let Some(value) =
        TL_RADER_ORDER.with(|cache| cache.borrow().get(&key).and_then(Weak::upgrade))
    {
        return value;
    }
    if let Some(value) = RADER_ORDER_CACHE.read().get(&key).and_then(Weak::upgrade) {
        TL_RADER_ORDER.with(|cache| cache.borrow_mut().insert(key, Arc::downgrade(&value)));
        return value;
    }
    let value = replace_expired(&RADER_ORDER_CACHE, key, Arc::from(build(key)));
    TL_RADER_ORDER.with(|cache| cache.borrow_mut().insert(key, Arc::downgrade(&value)));
    value
}

declare_cache_store! {
    sealed_mod: negacyclic_sealed,
    sealed_trait: NegacyclicSpectrumStoreSealed,
    store_trait: NegacyclicSpectrumStore,
    extra_bounds: [Copy, Send, Sync, 'static],
    key: DirectionalKey,
    val_precise: NegacyclicIndex<Complex64>,
    val_reduced: NegacyclicIndex<Complex32>,
    val_self: NegacyclicIndex<Self>,
    tl_get: negacyclic_local_get,
    tl_insert: negacyclic_local_insert,
    global: negacyclic_global,
    global_ret_self: RwLock<FxHashMap<DirectionalKey, NegacyclicIndex<Self>>>,
    tl_precise: TL_RADER_NEGACYCLIC_PRECISE,
    tl_reduced: TL_RADER_NEGACYCLIC_REDUCED,
    global_precise: RADER_NEGACYCLIC_PRECISE_CACHE,
    global_reduced: RADER_NEGACYCLIC_REDUCED_CACHE,
}

pub(crate) fn cached_rader_negacyclic_spectra<F: NegacyclicSpectrumStore>(
    key: DirectionalKey,
    build: impl FnOnce(DirectionalKey) -> (Vec<F>, Vec<F>),
) -> NegacyclicEntry<F> {
    if let Some(value) = F::negacyclic_local_get(key).and_then(upgrade_negacyclic) {
        return value;
    }
    if let Some(value) = F::negacyclic_global()
        .read()
        .get(&key)
        .cloned()
        .and_then(upgrade_negacyclic)
    {
        F::negacyclic_local_insert(key, downgrade_negacyclic(&value));
        return value;
    }
    let (cyclic, negacyclic) = build(key);
    let built: NegacyclicEntry<F> = (cyclic.into(), negacyclic.into());
    let value = {
        let mut index = F::negacyclic_global().write();
        if let Some(value) = index.get(&key).cloned().and_then(upgrade_negacyclic) {
            value
        } else {
            index.insert(key, downgrade_negacyclic(&built));
            built
        }
    };
    F::negacyclic_local_insert(key, downgrade_negacyclic(&value));
    value
}

fn upgrade_negacyclic<C>(entry: NegacyclicIndex<C>) -> Option<NegacyclicEntry<C>> {
    Some((entry.0.upgrade()?, entry.1.upgrade()?))
}

fn downgrade_negacyclic<C>(entry: &NegacyclicEntry<C>) -> NegacyclicIndex<C> {
    (Arc::downgrade(&entry.0), Arc::downgrade(&entry.1))
}

declare_cache_store! {
    sealed_mod: twiddle_sealed,
    sealed_trait: NegTwiddleStoreSealed,
    store_trait: NegTwiddleStore,
    extra_bounds: [Copy, Send, Sync, 'static],
    key: usize,
    val_precise: Weak<[Complex64]>,
    val_reduced: Weak<[Complex32]>,
    val_self: Weak<[Self]>,
    tl_get: twiddle_local_get,
    tl_insert: twiddle_local_insert,
    global: twiddle_global,
    global_ret_self: RwLock<FxHashMap<usize, Weak<[Self]>>>,
    tl_precise: TL_RADER_NEG_TWIDDLES_PRECISE,
    tl_reduced: TL_RADER_NEG_TWIDDLES_REDUCED,
    global_precise: RADER_NEG_TWIDDLES_PRECISE_CACHE,
    global_reduced: RADER_NEG_TWIDDLES_REDUCED_CACHE,
}

pub(crate) fn cached_rader_neg_twiddles<F: NegTwiddleStore>(
    key: usize,
    build: impl FnOnce(usize) -> Vec<F>,
) -> Arc<[F]> {
    if let Some(value) = F::twiddle_local_get(key).and_then(|entry| entry.upgrade()) {
        return value;
    }
    if let Some(value) = F::twiddle_global().read().get(&key).and_then(Weak::upgrade) {
        F::twiddle_local_insert(key, Arc::downgrade(&value));
        return value;
    }
    let value = replace_expired(F::twiddle_global(), key, Arc::from(build(key)));
    F::twiddle_local_insert(key, Arc::downgrade(&value));
    value
}

#[cfg(test)]
mod tests {
    use super::{
        cached_rader_neg_twiddles, cached_rader_negacyclic_spectra, cached_rader_order,
        cached_rader_spectrum,
    };
    use eunomia::Complex64;
    use std::sync::Arc;

    #[test]
    fn indexes_reuse_only_live_owners() {
        let order_key = (usize::MAX - 1, usize::MAX - 2);
        let spectrum_key = (usize::MAX - 3, 0, usize::MAX - 4);
        let negacyclic_key = (usize::MAX - 5, 0, usize::MAX - 6);
        let twiddle_key = usize::MAX - 7;

        let order = cached_rader_order(order_key, |_| vec![1, 2, 3]);
        assert!(Arc::ptr_eq(
            &order,
            &cached_rader_order(order_key, |_| unreachable!())
        ));
        let released_order = Arc::downgrade(&order);
        drop(order);
        assert!(released_order.upgrade().is_none());

        let spectrum =
            cached_rader_spectrum::<Complex64>(spectrum_key, |_| vec![Complex64::new(1.0, 0.0)]);
        assert!(Arc::ptr_eq(
            &spectrum,
            &cached_rader_spectrum::<Complex64>(spectrum_key, |_| unreachable!())
        ));
        let released_spectrum = Arc::downgrade(&spectrum);
        drop(spectrum);
        assert!(released_spectrum.upgrade().is_none());

        let negacyclic = cached_rader_negacyclic_spectra::<Complex64>(negacyclic_key, |_| {
            (
                vec![Complex64::new(1.0, 0.0)],
                vec![Complex64::new(0.0, 1.0)],
            )
        });
        let reused =
            cached_rader_negacyclic_spectra::<Complex64>(negacyclic_key, |_| unreachable!());
        assert!(Arc::ptr_eq(&negacyclic.0, &reused.0));
        assert!(Arc::ptr_eq(&negacyclic.1, &reused.1));
        drop(reused);
        let released_cyclic = Arc::downgrade(&negacyclic.0);
        let released_negacyclic = Arc::downgrade(&negacyclic.1);
        drop(negacyclic);
        assert!(released_cyclic.upgrade().is_none());
        assert!(released_negacyclic.upgrade().is_none());

        let twiddles =
            cached_rader_neg_twiddles::<Complex64>(twiddle_key, |_| vec![Complex64::new(0.0, 1.0)]);
        let released_twiddles = Arc::downgrade(&twiddles);
        drop(twiddles);
        assert!(released_twiddles.upgrade().is_none());
    }
}
