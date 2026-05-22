use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

static TWIDDLE_FWD_PRECISE_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_PRECISE_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_FWD_REDUCED_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_REDUCED_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static TL_FWD_PRECISE: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_PRECISE: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_FWD_REDUCED: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_REDUCED: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
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
    global_ret_self: RwLock<HashMap<usize, Arc<[Self]>>>,
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
    global_ret_self: RwLock<HashMap<usize, Arc<[Self]>>>,
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
}

#[inline]
pub(crate) fn cached_twiddle_fwd<C: TwiddleStore>(n: usize) -> Arc<[C]> {
    if let Some(tw) = C::twiddle_tl_fwd_get(n) {
        return tw;
    }
    let tw = {
        let maybe = C::twiddle_global_fwd().read().get(&n).cloned();
        if let Some(tw) = maybe {
            tw
        } else {
            let new_tw: Arc<[C]> = Arc::from(C::build_twiddle_fwd(n));
            C::twiddle_global_fwd()
                .write()
                .entry(n)
                .or_insert_with(|| Arc::clone(&new_tw))
                .clone()
        }
    };
    C::twiddle_tl_fwd_insert(n, Arc::clone(&tw));
    tw
}

#[inline]
pub(crate) fn cached_twiddle_inv<C: TwiddleStore>(n: usize) -> Arc<[C]> {
    if let Some(tw) = C::twiddle_tl_inv_get(n) {
        return tw;
    }
    let tw = {
        let maybe = C::twiddle_global_inv().read().get(&n).cloned();
        if let Some(tw) = maybe {
            tw
        } else {
            let new_tw: Arc<[C]> = Arc::from(C::build_twiddle_inv(n));
            C::twiddle_global_inv()
                .write()
                .entry(n)
                .or_insert_with(|| Arc::clone(&new_tw))
                .clone()
        }
    };
    C::twiddle_tl_inv_insert(n, Arc::clone(&tw));
    tw
}
