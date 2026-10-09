use super::super::super::twiddle_table::TwiddleOutput;
use super::tables::{shared_table, LocalTable, SharedTable};
use eunomia::{Complex32, Complex64};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{Arc, Weak};

static FOUR_STEP_TW_PRECISE_CACHE: SharedTable<(usize, usize), Weak<[Complex64]>> = shared_table();
static FOUR_STEP_TW_REDUCED_CACHE: SharedTable<(usize, usize), Weak<[Complex32]>> = shared_table();

thread_local! {
    static TL_FOUR_STEP_TW_PRECISE: LocalTable<(usize, usize), Weak<[Complex64]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(4, Default::default()));
    static TL_FOUR_STEP_TW_REDUCED: LocalTable<(usize, usize), Weak<[Complex32]>> = RefCell::new(FxHashMap::with_capacity_and_hasher(4, Default::default()));
}

declare_cache_store! {
    sealed_mod: sealed,
    sealed_trait: FourStepStoreSealed,
    store_trait: FourStepStore,
    extra_bounds: [TwiddleOutput, Clone, Send, Sync, 'static],
    key: (usize, usize),
    val_precise: Weak<[Complex64]>,
    val_reduced: Weak<[Complex32]>,
    val_self: Weak<[Self]>,
    tl_get: four_step_tl_get,
    tl_insert: four_step_tl_insert,
    global: four_step_global,
    global_ret_self: RwLock<FxHashMap<(usize, usize), Weak<[Self]>>>,
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
    if let Some(v) = C::four_step_tl_get(key).and_then(|entry| entry.upgrade()) {
        return v;
    }
    if let Some(v) = C::four_step_global()
        .read()
        .get(&key)
        .and_then(Weak::upgrade)
    {
        C::four_step_tl_insert(key, Arc::downgrade(&v));
        return v;
    }
    let sign = -1.0_f64;
    let new_v: Arc<[C]> = (0..n)
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
    let v = {
        let mut index = C::four_step_global().write();
        if let Some(v) = index.get(&key).and_then(Weak::upgrade) {
            v
        } else {
            index.insert(key, Arc::downgrade(&new_v));
            new_v
        }
    };
    C::four_step_tl_insert(key, Arc::downgrade(&v));
    v
}

#[cfg(test)]
mod tests {
    use super::cached_four_step_twiddles;
    use eunomia::Complex64;
    use std::sync::Arc;

    #[test]
    fn index_reuses_live_owner_and_releases_after_drop() {
        let first = cached_four_step_twiddles::<Complex64>(1, 1);
        let second = cached_four_step_twiddles::<Complex64>(1, 1);
        assert!(Arc::ptr_eq(&first, &second));
        let released = Arc::downgrade(&first);
        drop(first);
        drop(second);
        assert!(released.upgrade().is_none());
    }
}
