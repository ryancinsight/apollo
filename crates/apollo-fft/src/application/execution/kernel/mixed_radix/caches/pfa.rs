//! The Good-Thomas CRT permutation cache: input gather and output scatter
//! tables per coprime pair.

use super::tables::{shared_table, LocalTable, SharedTable};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::Arc;

/// The input gather and output scatter tables of one coprime pair.
pub(crate) type PfaPermutation = (Arc<[usize]>, Arc<[usize]>);

static PFA_PERM_CACHE: SharedTable<(usize, usize), PfaPermutation> = shared_table();

thread_local! {
    pub(super) static TL_PFA_PERM: LocalTable<(usize, usize), PfaPermutation> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
}

/// Return precomputed Good-Thomas input and output CRT permutation tables for
/// a pair of coprime factors `(n1, n2)`.
///
/// `input_perm[i1 * n2 + i2]  = (i1 * n2 + i2 * n1) % n` — gather index for step 1.
/// `output_perm[k2 * n1 + k1] = (k1 * n2 * inv_n2_n1 + k2 * n1 * inv_n1_n2) % n` — scatter index for step 5.
///
/// Tables are computed once on first use and shared across threads via `Arc`.
#[inline]
pub(crate) fn cached_pfa_perm(n1: usize, n2: usize) -> PfaPermutation {
    let key = (n1, n2);
    if let Some(v) = TL_PFA_PERM.with(|c| c.borrow().get(&key).cloned()) {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().pfa_perm.tl_hit();
        return v;
    }
    #[cfg(feature = "cache-profiling")]
    super::profiler::get().pfa_perm.global_hit();
    let v = {
        let maybe_cached = PFA_PERM_CACHE.read().get(&key).cloned();
        if let Some(v) = maybe_cached {
            v
        } else {
            #[cfg(feature = "cache-profiling")]
            super::profiler::get().pfa_perm.miss();
            let pair = build_pfa_perm(n1, n2);
            PFA_PERM_CACHE
                .write()
                .entry(key)
                .or_insert_with(|| pair.clone())
                .clone()
        }
    };
    TL_PFA_PERM.with(|c| c.borrow_mut().insert(key, v.clone()));
    v
}

fn extended_gcd(a: usize, b: usize) -> (usize, i64, i64) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x, y) = extended_gcd(b % a, a);
    (g, y - (b as i64 / a as i64) * x, x)
}

fn mod_inverse_local(a: usize, m: usize) -> usize {
    let (_, x, _) = extended_gcd(a, m);
    ((x % m as i64 + m as i64) % m as i64) as usize
}

fn build_pfa_perm(n1: usize, n2: usize) -> PfaPermutation {
    let n = n1 * n2;
    let inv_n2_n1 = mod_inverse_local(n2, n1);
    let inv_n1_n2 = mod_inverse_local(n1, n2);

    let mut input_perm = vec![0usize; n];
    let mut output_perm = vec![0usize; n];

    for i1 in 0..n1 {
        for i2 in 0..n2 {
            input_perm[i1 * n2 + i2] = (i1 * n2 + i2 * n1) % n;
        }
    }
    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let k_idx = (k1 * n2 * inv_n2_n1 + k2 * n1 * inv_n1_n2) % n;
            output_perm[k2 * n1 + k1] = k_idx;
        }
    }
    // The PFA kernel gathers and scatters through these values unchecked;
    // the build-time check is the release-mode half of that contract.
    assert!(
        input_perm
            .iter()
            .chain(&output_perm)
            .all(|&index| index < n),
        "invariant: PFA permutation entries stay below n = n1 * n2"
    );
    (Arc::from(input_perm), Arc::from(output_perm))
}
