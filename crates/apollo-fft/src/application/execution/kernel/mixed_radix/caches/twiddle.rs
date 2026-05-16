use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

static TWIDDLE_FWD_64_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_64_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_FWD_32_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_32_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static TL_FWD_64: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_64: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_FWD_32: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_32: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
}

#[inline]
pub(super) fn tl_cached<T: Clone>(
    tl: &'static std::thread::LocalKey<RefCell<HashMap<usize, Arc<[T]>>>>,
    global: &'static std::sync::LazyLock<RwLock<HashMap<usize, Arc<[T]>>>>,
    n: usize,
    build_fn: impl FnOnce(usize) -> Vec<T>,
) -> Arc<[T]> {
    if let Some(tw) = tl.with(|c| c.borrow().get(&n).cloned()) {
        return tw;
    }
    let tw = {
        let maybe_cached = global.read().get(&n).cloned();
        if let Some(tw) = maybe_cached {
            tw
        } else {
            let new_tw: Arc<[T]> = Arc::from(build_fn(n));
            global
                .write()
                .entry(n)
                .or_insert_with(|| Arc::clone(&new_tw))
                .clone()
        }
    };
    tl.with(|c| c.borrow_mut().insert(n, Arc::clone(&tw)));
    tw
}

#[inline]
pub(crate) fn cached_twiddle_fwd_64(n: usize) -> Arc<[Complex64]> {
    tl_cached(
        &TL_FWD_64,
        &TWIDDLE_FWD_64_CACHE,
        n,
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_64(n: usize) -> Arc<[Complex64]> {
    tl_cached(
        &TL_INV_64,
        &TWIDDLE_INV_64_CACHE,
        n,
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_inverse_twiddle_table,
    )
}

#[inline]
pub(crate) fn cached_twiddle_fwd_32(n: usize) -> Arc<[Complex32]> {
    tl_cached(
        &TL_FWD_32,
        &TWIDDLE_FWD_32_CACHE,
        n,
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_32(n: usize) -> Arc<[Complex32]> {
    tl_cached(
        &TL_INV_32,
        &TWIDDLE_INV_32_CACHE,
        n,
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_inverse_twiddle_table,
    )
}
