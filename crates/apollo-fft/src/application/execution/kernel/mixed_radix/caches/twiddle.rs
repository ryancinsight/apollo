use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::Arc;

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

    static TL_FWD_PRECISE_POW2: RefCell<[Option<Arc<[Complex64]>>; 32]> = RefCell::new([const { None }; 32]);
    static TL_INV_PRECISE_POW2: RefCell<[Option<Arc<[Complex64]>>; 32]> = RefCell::new([const { None }; 32]);
    static TL_FWD_REDUCED_POW2: RefCell<[Option<Arc<[Complex32]>>; 32]> = RefCell::new([const { None }; 32]);
    static TL_INV_REDUCED_POW2: RefCell<[Option<Arc<[Complex32]>>; 32]> = RefCell::new([const { None }; 32]);

    static TL_FWD_PRECISE_POW2_RAW: std::cell::Cell<[*const [Complex64]; 32]> = const { std::cell::Cell::new([std::ptr::slice_from_raw_parts(std::ptr::null(), 0); 32]) };
    static TL_INV_PRECISE_POW2_RAW: std::cell::Cell<[*const [Complex64]; 32]> = const { std::cell::Cell::new([std::ptr::slice_from_raw_parts(std::ptr::null(), 0); 32]) };
    static TL_FWD_REDUCED_POW2_RAW: std::cell::Cell<[*const [Complex32]; 32]> = const { std::cell::Cell::new([std::ptr::slice_from_raw_parts(std::ptr::null(), 0); 32]) };
    static TL_INV_REDUCED_POW2_RAW: std::cell::Cell<[*const [Complex32]; 32]> = const { std::cell::Cell::new([std::ptr::slice_from_raw_parts(std::ptr::null(), 0); 32]) };

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

/// Combined twiddle-cache trait: inherits fwd+inv cache dispatch and adds
/// precision-specific build helpers.
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

impl TwiddleStore for Complex64 {
    #[inline]
    fn build_twiddle_fwd(n: usize) -> Vec<Complex64> {
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(n)
    }
    #[inline]
    fn build_twiddle_inv(n: usize) -> Vec<Complex64> {
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_inverse_twiddle_table(n)
    }
    #[inline]
    fn twiddle_tl_fwd_get_pow2(idx: usize) -> Option<Arc<[Complex64]>> {
        TL_FWD_PRECISE_POW2.with(|c| c.borrow()[idx].clone())
    }
    #[inline]
    fn twiddle_tl_fwd_insert_pow2(idx: usize, v: Arc<[Complex64]>) {
        TL_FWD_PRECISE_POW2.with(|c| {
            c.borrow_mut()[idx] = Some(v);
        });
    }
    #[inline]
    fn twiddle_tl_inv_get_pow2(idx: usize) -> Option<Arc<[Complex64]>> {
        TL_INV_PRECISE_POW2.with(|c| c.borrow()[idx].clone())
    }
    #[inline]
    fn twiddle_tl_inv_insert_pow2(idx: usize, v: Arc<[Complex64]>) {
        TL_INV_PRECISE_POW2.with(|c| {
            c.borrow_mut()[idx] = Some(v);
        });
    }

    #[inline]
    fn twiddle_tl_fwd_get_pow2_raw(idx: usize) -> *const [Complex64] {
        TL_FWD_PRECISE_POW2_RAW.with(|c| c.get()[idx])
    }
    #[inline]
    fn twiddle_tl_fwd_insert_pow2_raw(idx: usize, ptr: *const [Complex64]) {
        TL_FWD_PRECISE_POW2_RAW.with(|c| {
            let mut arr = c.get();
            arr[idx] = ptr;
            c.set(arr);
        });
    }
    #[inline]
    fn twiddle_tl_inv_get_pow2_raw(idx: usize) -> *const [Complex64] {
        TL_INV_PRECISE_POW2_RAW.with(|c| c.get()[idx])
    }
    #[inline]
    fn twiddle_tl_inv_insert_pow2_raw(idx: usize, ptr: *const [Complex64]) {
        TL_INV_PRECISE_POW2_RAW.with(|c| {
            let mut arr = c.get();
            arr[idx] = ptr;
            c.set(arr);
        });
    }

    #[inline]
    fn twiddle_tl_fwd_get_raw(n: usize) -> Option<*const [Complex64]> {
        TL_FWD_PRECISE_RAW.with(|c| c.borrow().get(&n).copied())
    }
    #[inline]
    fn twiddle_tl_fwd_insert_raw(n: usize, ptr: *const [Complex64]) {
        TL_FWD_PRECISE_RAW.with(|c| {
            c.borrow_mut().insert(n, ptr);
        });
    }
    #[inline]
    fn twiddle_tl_inv_get_raw(n: usize) -> Option<*const [Complex64]> {
        TL_INV_PRECISE_RAW.with(|c| c.borrow().get(&n).copied())
    }
    #[inline]
    fn twiddle_tl_inv_insert_raw(n: usize, ptr: *const [Complex64]) {
        TL_INV_PRECISE_RAW.with(|c| {
            c.borrow_mut().insert(n, ptr);
        });
    }
}

impl TwiddleStore for Complex32 {
    #[inline]
    fn build_twiddle_fwd(n: usize) -> Vec<Complex32> {
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(n)
    }
    #[inline]
    fn build_twiddle_inv(n: usize) -> Vec<Complex32> {
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_inverse_twiddle_table(n)
    }
    #[inline]
    fn twiddle_tl_fwd_get_pow2(idx: usize) -> Option<Arc<[Complex32]>> {
        TL_FWD_REDUCED_POW2.with(|c| c.borrow()[idx].clone())
    }
    #[inline]
    fn twiddle_tl_fwd_insert_pow2(idx: usize, v: Arc<[Complex32]>) {
        TL_FWD_REDUCED_POW2.with(|c| {
            c.borrow_mut()[idx] = Some(v);
        });
    }
    #[inline]
    fn twiddle_tl_inv_get_pow2(idx: usize) -> Option<Arc<[Complex32]>> {
        TL_INV_REDUCED_POW2.with(|c| c.borrow()[idx].clone())
    }
    #[inline]
    fn twiddle_tl_inv_insert_pow2(idx: usize, v: Arc<[Complex32]>) {
        TL_INV_REDUCED_POW2.with(|c| {
            c.borrow_mut()[idx] = Some(v);
        });
    }

    #[inline]
    fn twiddle_tl_fwd_get_pow2_raw(idx: usize) -> *const [Complex32] {
        TL_FWD_REDUCED_POW2_RAW.with(|c| c.get()[idx])
    }
    #[inline]
    fn twiddle_tl_fwd_insert_pow2_raw(idx: usize, ptr: *const [Complex32]) {
        TL_FWD_REDUCED_POW2_RAW.with(|c| {
            let mut arr = c.get();
            arr[idx] = ptr;
            c.set(arr);
        });
    }
    #[inline]
    fn twiddle_tl_inv_get_pow2_raw(idx: usize) -> *const [Complex32] {
        TL_INV_REDUCED_POW2_RAW.with(|c| c.get()[idx])
    }
    #[inline]
    fn twiddle_tl_inv_insert_pow2_raw(idx: usize, ptr: *const [Complex32]) {
        TL_INV_REDUCED_POW2_RAW.with(|c| {
            let mut arr = c.get();
            arr[idx] = ptr;
            c.set(arr);
        });
    }

    #[inline]
    fn twiddle_tl_fwd_get_raw(n: usize) -> Option<*const [Complex32]> {
        TL_FWD_REDUCED_RAW.with(|c| c.borrow().get(&n).copied())
    }
    #[inline]
    fn twiddle_tl_fwd_insert_raw(n: usize, ptr: *const [Complex32]) {
        TL_FWD_REDUCED_RAW.with(|c| {
            c.borrow_mut().insert(n, ptr);
        });
    }
    #[inline]
    fn twiddle_tl_inv_get_raw(n: usize) -> Option<*const [Complex32]> {
        TL_INV_REDUCED_RAW.with(|c| c.borrow().get(&n).copied())
    }
    #[inline]
    fn twiddle_tl_inv_insert_raw(n: usize, ptr: *const [Complex32]) {
        TL_INV_REDUCED_RAW.with(|c| {
            c.borrow_mut().insert(n, ptr);
        });
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
                return f(unsafe { &*ptr });
            }
        }
    } else if let Some(ptr) = C::twiddle_tl_fwd_get_raw(n) {
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
                return f(unsafe { &*ptr });
            }
        }
    } else if let Some(ptr) = C::twiddle_tl_inv_get_raw(n) {
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
