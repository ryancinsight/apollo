use super::super::super::twiddle_table::TwiddleOutput;
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

static FOUR_STEP_TW_64_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize), Arc<[Complex64]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static FOUR_STEP_TW_32_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize), Arc<[Complex32]>>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static TL_FOUR_STEP_TW_64: RefCell<HashMap<(usize, usize), Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(4));
    static TL_FOUR_STEP_TW_32: RefCell<HashMap<(usize, usize), Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(4));
}

declare_cache_store! {
    sealed_mod: sealed,
    sealed_trait: FourStepStoreSealed,
    store_trait: FourStepStore,
    extra_bounds: [TwiddleOutput, Clone, 'static],
    key: (usize, usize),
    val64: Arc<[Complex64]>,
    val32: Arc<[Complex32]>,
    val_self: Arc<[Self]>,
    tl_get: four_step_tl_get,
    tl_insert: four_step_tl_insert,
    global: four_step_global,
    global_ret_self: RwLock<HashMap<(usize, usize), Arc<[Self]>>>,
    tl64: TL_FOUR_STEP_TW_64,
    tl32: TL_FOUR_STEP_TW_32,
    global64: FOUR_STEP_TW_64_CACHE,
    global32: FOUR_STEP_TW_32_CACHE,
}

fn build_four_step_twiddles<C: TwiddleOutput>(n: usize, n1: usize, n2: usize, sign: f64) -> Vec<C> {
    let total = n1 * n2;
    let mut re_tbl = vec![0.0f64; total];
    let mut im_tbl = vec![0.0f64; total];

    for k in 0..n1 {
        re_tbl[k] = 1.0;
    }
    if n2 == 1 {
        return re_tbl
            .iter()
            .zip(im_tbl.iter())
            .map(|(&r, &i)| C::from_components(r, i))
            .collect();
    }

    let base_angle = sign * std::f64::consts::TAU / n as f64;
    let w_re = base_angle.cos();
    let w_im = base_angle.sin();
    let mut cur_re = 1.0_f64;
    let mut cur_im = 0.0_f64;
    for k in 0..n1 {
        re_tbl[n1 + k] = cur_re;
        im_tbl[n1 + k] = cur_im;
        let nr = cur_re * w_re - cur_im * w_im;
        let ni = cur_re * w_im + cur_im * w_re;
        cur_re = nr;
        cur_im = ni;
    }

    for j in 2..n2 {
        for k in 0..n1 {
            let pr = re_tbl[(j - 1) * n1 + k];
            let pi = im_tbl[(j - 1) * n1 + k];
            let br = re_tbl[n1 + k];
            let bi = im_tbl[n1 + k];
            re_tbl[j * n1 + k] = pr * br - pi * bi;
            im_tbl[j * n1 + k] = pr * bi + pi * br;
        }
    }

    re_tbl
        .iter()
        .zip(im_tbl.iter())
        .map(|(&r, &i)| C::from_components(r, i))
        .collect()
}

#[inline]
pub(crate) fn cached_four_step_twiddles<C: FourStepStore>(
    n: usize,
    n1: usize,
    n2: usize,
    inverse: bool,
) -> Arc<[C]> {
    let key = (n, inverse as usize);
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    if let Some(v) = C::four_step_tl_get(key) {
        return v;
    }
    let v = {
        let maybe = C::four_step_global().read().get(&key).cloned();
        if let Some(v) = maybe {
            v
        } else {
            let new_v: Arc<[C]> = Arc::from(build_four_step_twiddles::<C>(n, n1, n2, sign));
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
