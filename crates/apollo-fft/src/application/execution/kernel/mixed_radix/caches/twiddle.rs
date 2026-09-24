use super::super::super::twiddle_table::{build_twiddle_table, TwiddleOutput};
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::thread::LocalKey;

static TWIDDLE_FWD_PRECISE_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static TWIDDLE_INV_PRECISE_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static TWIDDLE_FWD_REDUCED_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));
static TWIDDLE_INV_REDUCED_CACHE: std::sync::LazyLock<RwLock<FxHashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

thread_local! {
    static TL_FWD_PRECISE: RefCell<FxHashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_INV_PRECISE: RefCell<FxHashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_FWD_REDUCED: RefCell<FxHashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_INV_REDUCED: RefCell<FxHashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));

    #[expect(clippy::missing_const_for_thread_local, reason = "Counterbalanced mixed-radix benchmarks favor lazy cache initialization")]
    static TL_FWD_PRECISE_POW2: Pow2Tables<Complex64> = Pow2Tables(RefCell::new([const { None }; 32]));
    #[expect(clippy::missing_const_for_thread_local, reason = "Counterbalanced mixed-radix benchmarks favor lazy cache initialization")]
    static TL_INV_PRECISE_POW2: Pow2Tables<Complex64> = Pow2Tables(RefCell::new([const { None }; 32]));
    #[expect(clippy::missing_const_for_thread_local, reason = "Counterbalanced mixed-radix benchmarks favor lazy cache initialization")]
    static TL_FWD_REDUCED_POW2: Pow2Tables<Complex32> = Pow2Tables(RefCell::new([const { None }; 32]));
    #[expect(clippy::missing_const_for_thread_local, reason = "Counterbalanced mixed-radix benchmarks favor lazy cache initialization")]
    static TL_INV_REDUCED_POW2: Pow2Tables<Complex32> = Pow2Tables(RefCell::new([const { None }; 32]));

    // INVARIANT (raw-cache immortality): every pointer stored in the `_RAW`
    // caches below is derived from an `Arc<[C]>` allocation that
    // `cached_twiddle_{fwd,inv}` has already inserted into the corresponding
    // global `static` map (`TWIDDLE_*_CACHE`). All twiddle caches — the global
    // maps, the TLS `Arc` maps, and the TLS pow2 arrays — are append-only:
    // no cache surface exposes remove/clear/eviction, and cached tables are
    // never mutated after construction. The global map therefore keeps each
    // pointed-to allocation alive and immutable for the remainder of the
    // program, so a non-null raw entry can never dangle. Every `unsafe`
    // dereference of these pointers relies on this invariant.
    static TL_FWD_PRECISE_POW2_RAW: Cell<[*const [Complex64]; 32]> = const { Cell::new([std::ptr::slice_from_raw_parts(std::ptr::null(), 0); 32]) };
    static TL_INV_PRECISE_POW2_RAW: Cell<[*const [Complex64]; 32]> = const { Cell::new([std::ptr::slice_from_raw_parts(std::ptr::null(), 0); 32]) };
    static TL_FWD_REDUCED_POW2_RAW: Cell<[*const [Complex32]; 32]> = const { Cell::new([std::ptr::slice_from_raw_parts(std::ptr::null(), 0); 32]) };
    static TL_INV_REDUCED_POW2_RAW: Cell<[*const [Complex32]; 32]> = const { Cell::new([std::ptr::slice_from_raw_parts(std::ptr::null(), 0); 32]) };

    static TL_FWD_PRECISE_RAW: RefCell<FxHashMap<usize, *const [Complex64]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_INV_PRECISE_RAW: RefCell<FxHashMap<usize, *const [Complex64]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_FWD_REDUCED_RAW: RefCell<FxHashMap<usize, *const [Complex32]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
    static TL_INV_REDUCED_RAW: RefCell<FxHashMap<usize, *const [Complex32]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
}

declare_cache_store! {
    sealed_mod: fwd_sealed,
    sealed_trait: TwiddleFwdStoreSealed,
    store_trait: TwiddleFwdStore,
    extra_bounds: [Clone, 'static],
    key: usize,
    val_precise: Arc<[Complex64]>,
    val_reduced: Arc<[Complex32]>,
    val_self: Arc<[Self]>,
    tl_get: twiddle_tl_fwd_get,
    tl_insert: twiddle_tl_fwd_insert,
    global: twiddle_global_fwd,
    global_ret_self: RwLock<FxHashMap<usize, Arc<[Self]>>>,
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
    val_precise: Arc<[Complex64]>,
    val_reduced: Arc<[Complex32]>,
    val_self: Arc<[Self]>,
    tl_get: twiddle_tl_inv_get,
    tl_insert: twiddle_tl_inv_insert,
    global: twiddle_global_inv,
    global_ret_self: RwLock<FxHashMap<usize, Arc<[Self]>>>,
    tl_precise: TL_INV_PRECISE,
    tl_reduced: TL_INV_REDUCED,
    global_precise: TWIDDLE_INV_PRECISE_CACHE,
    global_reduced: TWIDDLE_INV_REDUCED_CACHE,
}

/// Per-thread `Arc` handles to the power-of-two tables, indexed by `log2 n`.
pub(crate) struct Pow2Tables<C>(RefCell<[Option<Arc<[C]>>; 32]>);

/// The thread-local twiddle slots one transform direction of one element owns.
///
/// Every key names a `thread_local!` declared above; the raw-cache immortality
/// invariant stated there covers the `pow2_raw` and `raw` slots.
pub(crate) struct TwiddleSlotKeys<C: 'static> {
    pow2: &'static LocalKey<Pow2Tables<C>>,
    pow2_raw: &'static LocalKey<Cell<[*const [C]; 32]>>,
    raw: &'static LocalKey<RefCell<FxHashMap<usize, *const [C]>>>,
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
        pow2_raw: &TL_FWD_PRECISE_POW2_RAW,
        raw: &TL_FWD_PRECISE_RAW,
    };
    const INV: TwiddleSlotKeys<Self> = TwiddleSlotKeys {
        pow2: &TL_INV_PRECISE_POW2,
        pow2_raw: &TL_INV_PRECISE_POW2_RAW,
        raw: &TL_INV_PRECISE_RAW,
    };
}

impl slots_sealed::TwiddleSlots for Complex32 {
    const FWD: TwiddleSlotKeys<Self> = TwiddleSlotKeys {
        pow2: &TL_FWD_REDUCED_POW2,
        pow2_raw: &TL_FWD_REDUCED_POW2_RAW,
        raw: &TL_FWD_REDUCED_RAW,
    };
    const INV: TwiddleSlotKeys<Self> = TwiddleSlotKeys {
        pow2: &TL_INV_REDUCED_POW2,
        pow2_raw: &TL_INV_REDUCED_POW2_RAW,
        raw: &TL_INV_REDUCED_RAW,
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

    fn twiddle_tl_fwd_get_pow2_raw(idx: usize) -> *const [Self];
    fn twiddle_tl_fwd_insert_pow2_raw(idx: usize, ptr: *const [Self]);
    fn twiddle_tl_inv_get_pow2_raw(idx: usize) -> *const [Self];
    fn twiddle_tl_inv_insert_pow2_raw(idx: usize, ptr: *const [Self]);

    fn twiddle_tl_fwd_get_raw(n: usize) -> Option<*const [Self]>;
    fn twiddle_tl_fwd_insert_raw(n: usize, ptr: *const [Self]);
    fn twiddle_tl_inv_get_raw(n: usize) -> Option<*const [Self]>;
    fn twiddle_tl_inv_insert_raw(n: usize, ptr: *const [Self]);
}

#[inline]
fn slot_get_pow2<C>(keys: &TwiddleSlotKeys<C>, idx: usize) -> Option<Arc<[C]>> {
    keys.pow2.with(|c| c.0.borrow()[idx].clone())
}

#[inline]
fn slot_insert_pow2<C>(keys: &TwiddleSlotKeys<C>, idx: usize, v: Arc<[C]>) {
    keys.pow2.with(|c| {
        c.0.borrow_mut()[idx] = Some(v);
    });
}

#[inline]
fn slot_get_pow2_raw<C>(keys: &TwiddleSlotKeys<C>, idx: usize) -> *const [C] {
    keys.pow2_raw.with(|c| c.get()[idx])
}

#[inline]
fn slot_insert_pow2_raw<C>(keys: &TwiddleSlotKeys<C>, idx: usize, ptr: *const [C]) {
    keys.pow2_raw.with(|c| {
        let mut arr = c.get();
        arr[idx] = ptr;
        c.set(arr);
    });
}

#[inline]
fn slot_get_raw<C>(keys: &TwiddleSlotKeys<C>, n: usize) -> Option<*const [C]> {
    keys.raw.with(|c| c.borrow().get(&n).copied())
}

#[inline]
fn slot_insert_raw<C>(keys: &TwiddleSlotKeys<C>, n: usize, ptr: *const [C]) {
    keys.raw.with(|c| {
        c.borrow_mut().insert(n, ptr);
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

    #[inline]
    fn twiddle_tl_fwd_get_pow2_raw(idx: usize) -> *const [C] {
        slot_get_pow2_raw(&C::FWD, idx)
    }
    #[inline]
    fn twiddle_tl_fwd_insert_pow2_raw(idx: usize, ptr: *const [C]) {
        slot_insert_pow2_raw(&C::FWD, idx, ptr);
    }
    #[inline]
    fn twiddle_tl_inv_get_pow2_raw(idx: usize) -> *const [C] {
        slot_get_pow2_raw(&C::INV, idx)
    }
    #[inline]
    fn twiddle_tl_inv_insert_pow2_raw(idx: usize, ptr: *const [C]) {
        slot_insert_pow2_raw(&C::INV, idx, ptr);
    }

    #[inline]
    fn twiddle_tl_fwd_get_raw(n: usize) -> Option<*const [C]> {
        slot_get_raw(&C::FWD, n)
    }
    #[inline]
    fn twiddle_tl_fwd_insert_raw(n: usize, ptr: *const [C]) {
        slot_insert_raw(&C::FWD, n, ptr);
    }
    #[inline]
    fn twiddle_tl_inv_get_raw(n: usize) -> Option<*const [C]> {
        slot_get_raw(&C::INV, n)
    }
    #[inline]
    fn twiddle_tl_inv_insert_raw(n: usize, ptr: *const [C]) {
        slot_insert_raw(&C::INV, n, ptr);
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
    if let Some(tw) = C::twiddle_tl_fwd_get(n) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().twiddle_fwd_precise.tl_hit();
        return tw;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().twiddle_fwd_precise.global_hit();
    let tw = {
        let maybe = C::twiddle_global_fwd().read().get(&n).cloned();
        if let Some(tw) = maybe {
            tw
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().twiddle_fwd_precise.miss();
            let new_tw: Arc<[C]> = Arc::from(C::build_twiddle_fwd(n));
            C::twiddle_global_fwd()
                .write()
                .entry(n)
                .or_insert_with(|| Arc::clone(&new_tw))
                .clone()
        }
    };
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            C::twiddle_tl_fwd_insert_pow2(idx, Arc::clone(&tw));
        }
    }
    C::twiddle_tl_fwd_insert(n, Arc::clone(&tw));
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
    if let Some(tw) = C::twiddle_tl_inv_get(n) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().twiddle_inv_precise.tl_hit();
        return tw;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().twiddle_inv_precise.global_hit();
    let tw = {
        let maybe = C::twiddle_global_inv().read().get(&n).cloned();
        if let Some(tw) = maybe {
            tw
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().twiddle_inv_precise.miss();
            let new_tw: Arc<[C]> = Arc::from(C::build_twiddle_inv(n));
            C::twiddle_global_inv()
                .write()
                .entry(n)
                .or_insert_with(|| Arc::clone(&new_tw))
                .clone()
        }
    };
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            C::twiddle_tl_inv_insert_pow2(idx, Arc::clone(&tw));
        }
    }
    C::twiddle_tl_inv_insert(n, Arc::clone(&tw));
    tw
}

#[inline]
pub(crate) fn with_twiddle_fwd<C: TwiddleStore, R>(n: usize, f: impl FnOnce(&[C]) -> R) -> R {
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            let ptr = C::twiddle_tl_fwd_get_pow2_raw(idx);
            if !ptr.is_null() {
                // SAFETY: valid because cache entries are never evicted; see
                // the raw-cache immortality invariant at the `_RAW`
                // thread-local declarations.
                return f(unsafe { &*ptr });
            }
        }
    } else if let Some(ptr) = C::twiddle_tl_fwd_get_raw(n) {
        // SAFETY: valid because cache entries are never evicted; see the
        // raw-cache immortality invariant at the `_RAW` thread-local
        // declarations.
        return f(unsafe { &*ptr });
    }
    let tw = cached_twiddle_fwd::<C>(n);
    let ptr = std::ptr::from_ref::<[C]>(tw.as_ref());
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            C::twiddle_tl_fwd_insert_pow2_raw(idx, ptr);
        }
    } else {
        C::twiddle_tl_fwd_insert_raw(n, ptr);
    }
    f(&tw)
}

#[inline]
pub(crate) fn with_twiddle_inv<C: TwiddleStore, R>(n: usize, f: impl FnOnce(&[C]) -> R) -> R {
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            let ptr = C::twiddle_tl_inv_get_pow2_raw(idx);
            if !ptr.is_null() {
                // SAFETY: valid because cache entries are never evicted; see
                // the raw-cache immortality invariant at the `_RAW`
                // thread-local declarations.
                return f(unsafe { &*ptr });
            }
        }
    } else if let Some(ptr) = C::twiddle_tl_inv_get_raw(n) {
        // SAFETY: valid because cache entries are never evicted; see the
        // raw-cache immortality invariant at the `_RAW` thread-local
        // declarations.
        return f(unsafe { &*ptr });
    }
    let tw = cached_twiddle_inv::<C>(n);
    let ptr = std::ptr::from_ref::<[C]>(tw.as_ref());
    if n.is_power_of_two() {
        let idx = n.trailing_zeros() as usize;
        if idx < 32 {
            C::twiddle_tl_inv_insert_pow2_raw(idx, ptr);
        }
    } else {
        C::twiddle_tl_inv_insert_raw(n, ptr);
    }
    f(&tw)
}
