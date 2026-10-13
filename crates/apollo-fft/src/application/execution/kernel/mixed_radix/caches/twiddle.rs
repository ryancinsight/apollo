use super::super::super::twiddle_table::{build_twiddle_table, TwiddleOutput};
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{Arc, Weak};
use std::thread::LocalKey;

static TWIDDLE_FWD_PRECISE_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Weak<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static TWIDDLE_INV_PRECISE_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Weak<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static TWIDDLE_FWD_REDUCED_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Weak<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static TWIDDLE_INV_REDUCED_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Weak<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

thread_local! {
    static TL_FWD_PRECISE: RefCell<FxHashMap<usize, Weak<[Complex64]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_INV_PRECISE: RefCell<FxHashMap<usize, Weak<[Complex64]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_FWD_REDUCED: RefCell<FxHashMap<usize, Weak<[Complex32]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_INV_REDUCED: RefCell<FxHashMap<usize, Weak<[Complex32]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));

    #[expect(clippy::missing_const_for_thread_local, reason = "Counterbalanced mixed-radix benchmarks favor lazy cache initialization")]
    static TL_FWD_PRECISE_POW2: Pow2Tables<Complex64> = Pow2Tables(RefCell::new([const { None }; 32]));
    #[expect(clippy::missing_const_for_thread_local, reason = "Counterbalanced mixed-radix benchmarks favor lazy cache initialization")]
    static TL_INV_PRECISE_POW2: Pow2Tables<Complex64> = Pow2Tables(RefCell::new([const { None }; 32]));
    #[expect(clippy::missing_const_for_thread_local, reason = "Counterbalanced mixed-radix benchmarks favor lazy cache initialization")]
    static TL_FWD_REDUCED_POW2: Pow2Tables<Complex32> = Pow2Tables(RefCell::new([const { None }; 32]));
    #[expect(clippy::missing_const_for_thread_local, reason = "Counterbalanced mixed-radix benchmarks favor lazy cache initialization")]
    static TL_INV_REDUCED_POW2: Pow2Tables<Complex32> = Pow2Tables(RefCell::new([const { None }; 32]));
}

declare_cache_store! {
    sealed_mod: fwd_sealed,
    sealed_trait: TwiddleFwdStoreSealed,
    store_trait: TwiddleFwdStore,
    extra_bounds: [Clone, 'static],
    key: usize,
    val_precise: Weak<[Complex64]>,
    val_reduced: Weak<[Complex32]>,
    val_self: Weak<[Self]>,
    tl_get: twiddle_tl_fwd_get,
    tl_insert: twiddle_tl_fwd_insert,
    global: twiddle_global_fwd,
    global_ret_self: RwLock<FxHashMap<usize, Weak<[Self]>>>,
    tl_precise: TL_FWD_PRECISE,
    tl_reduced: TL_FWD_REDUCED,
    global_precise: TWIDDLE_FWD_PRECISE_CACHE,
    global_reduced: TWIDDLE_FWD_REDUCED_CACHE,
}

declare_cache_store! {
    sealed_mod: inv_sealed,
    sealed_trait: TwiddleInvStoreSealed,
    store_trait: TwiddleInvStore,
    extra_bounds: [Clone, 'static],
    key: usize,
    val_precise: Weak<[Complex64]>,
    val_reduced: Weak<[Complex32]>,
    val_self: Weak<[Self]>,
    tl_get: twiddle_tl_inv_get,
    tl_insert: twiddle_tl_inv_insert,
    global: twiddle_global_inv,
    global_ret_self: RwLock<FxHashMap<usize, Weak<[Self]>>>,
    tl_precise: TL_INV_PRECISE,
    tl_reduced: TL_INV_REDUCED,
    global_precise: TWIDDLE_INV_PRECISE_CACHE,
    global_reduced: TWIDDLE_INV_REDUCED_CACHE,
}

/// Per-thread weak handles to power-of-two tables, indexed by `log2 n`.
pub(crate) struct Pow2Tables<C>(RefCell<[Option<Weak<[C]>>; 32]>);

/// The thread-local twiddle slots one transform direction of one element owns.
///
/// Every key names a `thread_local!` declared above.
pub(crate) struct TwiddleSlotKeys<C: 'static> {
    pow2: &'static LocalKey<Pow2Tables<C>>,
}

mod slots_sealed {
    /// Binds a complex element to its thread-local twiddle slots.
    ///
    /// Rust has no generic statics, so the slots each element owns are the
    /// one per-element fact; everything done with them is written once, in
    /// the blanket [`TwiddleStore`](super::TwiddleStore) implementation.
    pub(crate) trait TwiddleSlots: Sized + 'static {
        const FWD: super::TwiddleSlotKeys<Self>;
        const INV: super::TwiddleSlotKeys<Self>;
    }
}

impl slots_sealed::TwiddleSlots for Complex64 {
    const FWD: TwiddleSlotKeys<Self> = TwiddleSlotKeys {
        pow2: &TL_FWD_PRECISE_POW2,
    };
    const INV: TwiddleSlotKeys<Self> = TwiddleSlotKeys {
        pow2: &TL_INV_PRECISE_POW2,
    };
}

impl slots_sealed::TwiddleSlots for Complex32 {
    const FWD: TwiddleSlotKeys<Self> = TwiddleSlotKeys {
        pow2: &TL_FWD_REDUCED_POW2,
    };
    const INV: TwiddleSlotKeys<Self> = TwiddleSlotKeys {
        pow2: &TL_INV_REDUCED_POW2,
    };
}

/// Combined twiddle-cache trait: inherits fwd+inv cache dispatch and adds
/// the table builders and the thread-local slot accessors.
pub(crate) trait TwiddleStore: TwiddleFwdStore + TwiddleInvStore {
    fn build_twiddle_fwd(n: usize) -> Vec<Self>;
    fn build_twiddle_inv(n: usize) -> Vec<Self>;

    fn twiddle_tl_fwd_get_pow2(idx: usize) -> Option<Arc<[Self]>>;
    fn twiddle_tl_fwd_insert_pow2(idx: usize, v: Arc<[Self]>);
    fn twiddle_tl_inv_get_pow2(idx: usize) -> Option<Arc<[Self]>>;
    fn twiddle_tl_inv_insert_pow2(idx: usize, v: Arc<[Self]>);
}

#[inline]
fn slot_get_pow2<C>(keys: &TwiddleSlotKeys<C>, idx: usize) -> Option<Arc<[C]>> {
    keys.pow2
        .with(|c| c.0.borrow()[idx].as_ref().and_then(Weak::upgrade))
}

#[inline]
fn slot_insert_pow2<C>(keys: &TwiddleSlotKeys<C>, idx: usize, v: Arc<[C]>) {
    keys.pow2.with(|c| {
        c.0.borrow_mut()[idx] = Some(Arc::downgrade(&v));
    });
}

impl<C> TwiddleStore for C
where
    C: TwiddleFwdStore + TwiddleInvStore + slots_sealed::TwiddleSlots + TwiddleOutput,
{
    #[inline]
    fn build_twiddle_fwd(n: usize) -> Vec<C> {
        build_twiddle_table(n, -1.0)
    }
    #[inline]
    fn build_twiddle_inv(n: usize) -> Vec<C> {
        build_twiddle_table(n, 1.0)
    }
    #[inline]
    fn twiddle_tl_fwd_get_pow2(idx: usize) -> Option<Arc<[C]>> {
        slot_get_pow2(&C::FWD, idx)
    }
    #[inline]
    fn twiddle_tl_fwd_insert_pow2(idx: usize, v: Arc<[C]>) {
        slot_insert_pow2(&C::FWD, idx, v);
    }
    #[inline]
    fn twiddle_tl_inv_get_pow2(idx: usize) -> Option<Arc<[C]>> {
        slot_get_pow2(&C::INV, idx)
    }
    #[inline]
    fn twiddle_tl_inv_insert_pow2(idx: usize, v: Arc<[C]>) {
        slot_insert_pow2(&C::INV, idx, v);
    }
}

#[inline]
pub(crate) fn cached_twiddle_fwd<C: TwiddleStore>(n: usize) -> Arc<[C]> {
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            if let Some(tw) = C::twiddle_tl_fwd_get_pow2(idx) {
                #[cfg(feature = "cache-profiling")]
                super::profiler::get().twiddle_fwd_precise.tl_hit();
                return tw;
            }
        }
    }
    if let Some(tw) = C::twiddle_tl_fwd_get(n).and_then(|table| table.upgrade()) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().twiddle_fwd_precise.tl_hit();
        return tw;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().twiddle_fwd_precise.global_hit();
    let tw = {
        let maybe = C::twiddle_global_fwd()
            .read()
            .get(&n)
            .and_then(Weak::upgrade);
        if let Some(tw) = maybe {
            tw
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().twiddle_fwd_precise.miss();
            let new_tw: Arc<[C]> = Arc::from(C::build_twiddle_fwd(n));
            let mut cache = C::twiddle_global_fwd().write();
            if let Some(tw) = cache.get(&n).and_then(Weak::upgrade) {
                tw
            } else {
                cache.retain(|_, table| table.strong_count() != 0);
                cache.insert(n, Arc::downgrade(&new_tw));
                new_tw
            }
        }
    };
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            C::twiddle_tl_fwd_insert_pow2(idx, Arc::clone(&tw));
        }
    }
    C::twiddle_tl_fwd_insert(n, Arc::downgrade(&tw));
    tw
}

#[inline]
pub(crate) fn cached_twiddle_inv<C: TwiddleStore>(n: usize) -> Arc<[C]> {
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            if let Some(tw) = C::twiddle_tl_inv_get_pow2(idx) {
                #[cfg(feature = "cache-profiling")]
                super::profiler::get().twiddle_inv_precise.tl_hit();
                return tw;
            }
        }
    }
    if let Some(tw) = C::twiddle_tl_inv_get(n).and_then(|table| table.upgrade()) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().twiddle_inv_precise.tl_hit();
        return tw;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().twiddle_inv_precise.global_hit();
    let tw = {
        let maybe = C::twiddle_global_inv()
            .read()
            .get(&n)
            .and_then(Weak::upgrade);
        if let Some(tw) = maybe {
            tw
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().twiddle_inv_precise.miss();
            let new_tw: Arc<[C]> = Arc::from(C::build_twiddle_inv(n));
            let mut cache = C::twiddle_global_inv().write();
            if let Some(tw) = cache.get(&n).and_then(Weak::upgrade) {
                tw
            } else {
                cache.retain(|_, table| table.strong_count() != 0);
                cache.insert(n, Arc::downgrade(&new_tw));
                new_tw
            }
        }
    };
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            C::twiddle_tl_inv_insert_pow2(idx, Arc::clone(&tw));
        }
    }
    C::twiddle_tl_inv_insert(n, Arc::downgrade(&tw));
    tw
}

#[inline]
pub(crate) fn with_twiddle_fwd<C: TwiddleStore, R>(n: usize, f: impl FnOnce(&[C]) -> R) -> R {
    let tw = cached_twiddle_fwd::<C>(n);
    f(&tw)
}

#[inline]
pub(crate) fn with_twiddle_inv<C: TwiddleStore, R>(n: usize, f: impl FnOnce(&[C]) -> R) -> R {
    let tw = cached_twiddle_inv::<C>(n);
    f(&tw)
}

#[cfg(test)]
mod tests {
    use super::{cached_twiddle_fwd, cached_twiddle_inv, with_twiddle_fwd, with_twiddle_inv};
    use eunomia::Complex64;
    use std::sync::Arc;

    #[test]
    fn borrowed_access_is_tied_to_a_live_owner() {
        let forward = cached_twiddle_fwd::<Complex64>(64);
        let released_forward = Arc::downgrade(&forward);
        let expected_forward = forward[7];
        let observed_forward = with_twiddle_fwd::<Complex64, _>(64, |table| {
            assert_eq!(table.as_ptr(), forward.as_ptr());
            table[7]
        });
        assert_eq!(observed_forward, expected_forward);
        drop(forward);
        assert!(released_forward.upgrade().is_none());
        assert_eq!(
            with_twiddle_fwd::<Complex64, _>(64, |table| table[7]),
            expected_forward,
            "a stale weak index must rebuild the same table"
        );

        let inverse = cached_twiddle_inv::<Complex64>(32);
        let released_inverse = Arc::downgrade(&inverse);
        let expected_inverse = inverse[11];
        assert_eq!(
            with_twiddle_inv::<Complex64, _>(32, |table| table[11]),
            expected_inverse
        );
        drop(inverse);
        assert!(released_inverse.upgrade().is_none());
    }
}
