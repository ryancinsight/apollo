use super::super::super::twiddle_table::TwiddleOutput;
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

static FOUR_STEP_TW_PRECISE_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize), Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static FOUR_STEP_TW_REDUCED_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize), Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static TL_FOUR_STEP_TW_PRECISE: RefCell<HashMap<(usize, usize), Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(4));
    static TL_FOUR_STEP_TW_REDUCED: RefCell<HashMap<(usize, usize), Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(4));
}

declare_cache_store! {
    sealed_mod: sealed,
    sealed_trait: FourStepStoreSealed,
    store_trait: FourStepStore,
    extra_bounds: [TwiddleOutput, Clone, 'static],
    key: (usize, usize),
    val_precise: Arc<[Complex64]>,
    val_reduced: Arc<[Complex32]>,
    val_self: Arc<[Self]>,
    tl_get: four_step_tl_get,
    tl_insert: four_step_tl_insert,
    global: four_step_global,
    global_ret_self: RwLock<HashMap<(usize, usize), Arc<[Self]>>>,
    tl_precise: TL_FOUR_STEP_TW_PRECISE,
    tl_reduced: TL_FOUR_STEP_TW_REDUCED,
    global_precise: FOUR_STEP_TW_PRECISE_CACHE,
    global_reduced: FOUR_STEP_TW_REDUCED_CACHE,
}

#[inline]
pub(crate) fn cached_four_step_twiddles<C: FourStepStore, const INVERSE: bool>(
    n: usize,
    n1: usize,
    n2: usize,
) -> Arc<[C]> {
    let key = (n, INVERSE as usize);
    if let Some(v) = C::four_step_tl_get(key) {
        return v;
    }
    let v = {
        let maybe = C::four_step_global().read().get(&key).cloned();
        if let Some(v) = maybe {
            v
        } else {
            let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
            let entries = n1
                .checked_mul(n2)
                .expect("invariant: twiddle matrix size fits usize");
            // Entry (j, k) is W_n^{j*k} = exp(sign * 2πi * j * k / n), through
            // the shared evaluation authority: mod-`n` reduction first and one
            // direct `sin_cos` per entry, so no entry carries a recurrence's
            // O(n1 + n2) roundings, which the `O(log N * u)` forward-error
            // bound (Higham, section 24.1) does not admit.
            let new_v: Arc<[C]> = (0..entries)
                .map(|index| {
                    let (sin, cos) =
                        crate::application::execution::kernel::twiddle_table::twiddle_components(
                            sign,
                            (index / n1) * (index % n1),
                            n,
                        );
                    C::from_components(cos, sin)
                })
                .collect();
            C::four_step_global()
                .write()
                .entry(key)
                .or_insert_with(|| Arc::clone(&new_v))
                .clone()
        }
    };
    C::four_step_tl_insert(key, Arc::clone(&v));
    v
}
