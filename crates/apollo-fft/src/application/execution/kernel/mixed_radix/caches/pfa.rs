use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

static PFA_PERM_CACHE: std::sync::LazyLock<
    RwLock<HashMap<(usize, usize), (Arc<[usize]>, Arc<[usize]>)>>,
> = std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    pub(super) static TL_PFA_PERM: RefCell<HashMap<(usize, usize), (Arc<[usize]>, Arc<[usize]>)>> =
        RefCell::new(HashMap::with_capacity(8));
}

fn build_pfa_perm(n1: usize, n2: usize) -> (Vec<usize>, Vec<usize>) {
    let n = n1 * n2;
    fn mod_inv(a: usize, m: usize) -> usize {
        if m == 1 {
            return 0;
        }
        let mut m0 = m as i64;
        let (mut y, mut x, mut a) = (0i64, 1i64, a as i64);
        while a > 1 {
            let q = a / m0;
            let t = m0;
            m0 = a % m0;
            a = t;
            let t = y;
            y = x - q * y;
            x = t;
        }
        if x < 0 {
            x += m as i64;
        }
        x as usize
    }
    let inv_n2_n1 = mod_inv(n2, n1);
    let inv_n1_n2 = mod_inv(n1, n2);

    let mut gather = vec![0usize; n];
    let mut scatter = vec![0usize; n];

    for i1 in 0..n1 {
        for i2 in 0..n2 {
            gather[i1 * n2 + i2] = (i1 * n2 + i2 * n1) % n;
        }
    }
    for k2 in 0..n2 {
        for k1 in 0..n1 {
            scatter[k2 * n1 + k1] = (k1 * n2 * inv_n2_n1 + k2 * n1 * inv_n1_n2) % n;
        }
    }
    (gather, scatter)
}

#[inline]
pub(crate) fn cached_pfa_perm(n1: usize, n2: usize) -> (Arc<[usize]>, Arc<[usize]>) {
    let key = (n1, n2);
    if let Some(v) = TL_PFA_PERM.with(|c| c.borrow().get(&key).cloned()) {
        return v;
    }
    let v = {
        let maybe = PFA_PERM_CACHE.read().get(&key).cloned();
        if let Some(v) = maybe {
            v
        } else {
            let (g, s) = build_pfa_perm(n1, n2);
            let pair: (Arc<[usize]>, Arc<[usize]>) = (Arc::from(g), Arc::from(s));
            PFA_PERM_CACHE.write().entry(key).or_insert_with(|| pair.clone()).clone()
        }
    };
    TL_PFA_PERM.with(|c| c.borrow_mut().insert(key, v.clone()));
    v
}
