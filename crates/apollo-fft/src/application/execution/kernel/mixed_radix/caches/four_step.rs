use super::super::super::twiddle_table::TwiddleOutput;
use super::tables::{shared_table, LocalTable, SharedTable};
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::Arc;

static FOUR_STEP_TW_PRECISE_CACHE: SharedTable<(usize, usize), Arc<[Complex64]>> = shared_table();
static FOUR_STEP_TW_REDUCED_CACHE: SharedTable<(usize, usize), Arc<[Complex32]>> = shared_table();

thread_local! {
    static TL_FOUR_STEP_TW_PRECISE: LocalTable<(usize, usize), Arc<[Complex64]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(4, Default::default()));
    static TL_FOUR_STEP_TW_REDUCED: LocalTable<(usize, usize), Arc<[Complex32]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(4, Default::default()));
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
    global_ret_self: RwLock<FxHashMap<(usize, usize), Arc<[Self]>>>,
    tl_precise: TL_FOUR_STEP_TW_PRECISE,
    tl_reduced: TL_FOUR_STEP_TW_REDUCED,
    global_precise: FOUR_STEP_TW_PRECISE_CACHE,
    global_reduced: FOUR_STEP_TW_REDUCED_CACHE,
}

/// The forward four-step matrix `W_N^{j k}`, `N = n1 n2`, row-major
/// `n2 x n1`, cached per shape.
///
/// Only the forward matrix is kept. An inverse reads each entry conjugated
/// at the multiply: `twiddle_components` negates the angle, and sine is odd
/// and cosine even, so the inverse entry is the forward one with its
/// imaginary part negated, and a negation is exact, so the product is
/// bitwise what a stored inverse matrix gave.
#[inline]
pub(crate) fn cached_four_step_twiddles<C: FourStepStore>(n1: usize, n2: usize) -> Arc<[C]> {
    let n = n1
        .checked_mul(n2)
        .expect("invariant: twiddle matrix size fits usize");
    let key = (n1, n2);
    if let Some(v) = C::four_step_tl_get(key) {
        return v;
    }
    let v = {
        let maybe = C::four_step_global().read().get(&key).cloned();
        if let Some(v) = maybe {
            v
        } else {
            let sign = -1.0_f64;
            let entries = n;
            // Entry (j, k) is W_n^{j*k} = exp(-2πi * j * k / n), through
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
