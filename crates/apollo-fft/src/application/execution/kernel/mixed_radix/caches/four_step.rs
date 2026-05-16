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
    pub(super) static TL_FOUR_STEP_TW_64: RefCell<HashMap<(usize, usize), Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(4));
    pub(super) static TL_FOUR_STEP_TW_32: RefCell<HashMap<(usize, usize), Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(4));
}

#[inline]
fn tl_cached_k2<T: Clone>(
    tl: &'static std::thread::LocalKey<RefCell<HashMap<(usize, usize), Arc<[T]>>>>,
    global: &'static std::sync::LazyLock<RwLock<HashMap<(usize, usize), Arc<[T]>>>>,
    key: (usize, usize),
    build_fn: impl FnOnce((usize, usize)) -> Vec<T>,
) -> Arc<[T]> {
    if let Some(v) = tl.with(|c| c.borrow().get(&key).cloned()) {
        return v;
    }
    let v = {
        let maybe = global.read().get(&key).cloned();
        if let Some(v) = maybe {
            v
        } else {
            let new_v: Arc<[T]> = Arc::from(build_fn(key));
            global.write().entry(key).or_insert_with(|| Arc::clone(&new_v)).clone()
        }
    };
    tl.with(|c| c.borrow_mut().insert(key, Arc::clone(&v)));
    v
}

fn build_four_step_twiddles<C: TwiddleOutput>(n: usize, n1: usize, n2: usize, sign: f64) -> Vec<C> {
    let total = n1 * n2;
    let mut re_tbl = vec![0.0f64; total];
    let mut im_tbl = vec![0.0f64; total];

    for k in 0..n1 {
        re_tbl[k] = 1.0;
    }
    if n2 == 1 {
        return re_tbl.iter().zip(im_tbl.iter()).map(|(&r, &i)| C::from_components(r, i)).collect();
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

    re_tbl.iter().zip(im_tbl.iter()).map(|(&r, &i)| C::from_components(r, i)).collect()
}

#[inline]
pub(crate) fn cached_four_step_twiddles_64(
    n: usize,
    n1: usize,
    n2: usize,
    inverse: bool,
) -> Arc<[Complex64]> {
    let key = (n, inverse as usize);
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    tl_cached_k2(
        &TL_FOUR_STEP_TW_64,
        &FOUR_STEP_TW_64_CACHE,
        key,
        |_| build_four_step_twiddles::<Complex64>(n, n1, n2, sign),
    )
}

#[inline]
pub(crate) fn cached_four_step_twiddles_32(
    n: usize,
    n1: usize,
    n2: usize,
    inverse: bool,
) -> Arc<[Complex32]> {
    let key = (n, inverse as usize);
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    tl_cached_k2(
        &TL_FOUR_STEP_TW_32,
        &FOUR_STEP_TW_32_CACHE,
        key,
        |_| build_four_step_twiddles::<Complex32>(n, n1, n2, sign),
    )
}
