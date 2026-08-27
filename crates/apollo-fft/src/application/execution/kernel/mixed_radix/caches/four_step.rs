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

/// Builds the `W_N^(j*k)` matrix without touching the caches.
///
/// The batched kernel's planar-plane cache builds from this and stores only
/// its own representation; routing it through [`cached_four_step_twiddles`]
/// would cache the interleaved matrix as a dead side effect beside the planes
/// — the doubled-footprint defect the planes' consolidation note recorded.
pub(crate) fn build_four_step_twiddles<C: TwiddleOutput, const INVERSE: bool>(
    n: usize,
    n1: usize,
    n2: usize,
) -> Vec<C> {
    let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
    let scale = sign * std::f64::consts::TAU / n as f64;

    // Entry (j, k) is W_n^{j*k} = exp(sign * 2πi * j * k / n), evaluated
    // directly. The exponent is reduced modulo `n` first — `j*k` reaches
    // `(n2-1)*(n1-1)`, just under `n` — so every angle stays inside one period.
    //
    // The superseded form built row 1 by a recurrence over `k` and each later
    // row by multiplying the row above it, so entry (j, k) carried the rounding
    // of `j + k` complex multiplications, up to `O(n1 + n2) = O(2*sqrt(n))`.
    // Twiddle error of that size defeats the `O(log N * u)` FFT forward-error
    // bound (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
    // section 24.1), which holds only for accurately computed twiddles. The
    // cost is `n1 * n2` `sin_cos` calls once per `(n, direction)`, behind the
    // thread-local and global caches above.
    (0..n2)
        .flat_map(|j| {
            (0..n1).map(move |k| {
                let reduced = (j * k) % n;
                let (sin, cos) = (scale * reduced as f64).sin_cos();
                C::from_components(cos, sin)
            })
        })
        .collect()
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
            let new_v: Arc<[C]> = Arc::from(build_four_step_twiddles::<C, INVERSE>(n, n1, n2));
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
