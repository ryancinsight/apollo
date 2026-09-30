//! Weak index for Good-Thomas CRT permutation tables.

use super::tables::{shared_table, LocalTable, SharedTable};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::{Arc, Weak};

/// Input gather and output scatter tables for one coprime pair.
pub(crate) struct PfaPermutation {
    pub(crate) input: Box<[usize]>,
    pub(crate) output: Box<[usize]>,
}

type PfaIndex = Weak<PfaPermutation>;

static PFA_PERM_CACHE: SharedTable<(usize, usize), PfaIndex> = shared_table();

thread_local! {
    static TL_PFA_PERM: LocalTable<(usize, usize), PfaIndex> = RefCell::new(FxHashMap::with_capacity_and_hasher(8, Default::default()));
}

/// Returns the reusable Good-Thomas input and output CRT permutations.
#[inline]
pub(crate) fn cached_pfa_perm(n1: usize, n2: usize) -> Arc<PfaPermutation> {
    let key = (n1, n2);
    if let Some(tables) = TL_PFA_PERM.with(|cache| cache.borrow().get(&key).and_then(Weak::upgrade))
    {
        #[cfg(feature = "cache-profiling")]
        super::profiler::get().pfa_perm.tl_hit();
        return tables;
    }

    #[cfg(feature = "cache-profiling")]
    super::profiler::get().pfa_perm.global_hit();
    if let Some(tables) = PFA_PERM_CACHE.read().get(&key).and_then(Weak::upgrade) {
        TL_PFA_PERM.with(|cache| cache.borrow_mut().insert(key, Arc::downgrade(&tables)));
        return tables;
    }

    #[cfg(feature = "cache-profiling")]
    super::profiler::get().pfa_perm.miss();
    let built = Arc::new(build_pfa_perm(n1, n2));
    let tables = {
        let mut cache = PFA_PERM_CACHE.write();
        if let Some(tables) = cache.get(&key).and_then(Weak::upgrade) {
            tables
        } else {
            cache.insert(key, Arc::downgrade(&built));
            built
        }
    };
    TL_PFA_PERM.with(|cache| cache.borrow_mut().insert(key, Arc::downgrade(&tables)));
    tables
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

    let mut input = vec![0usize; n];
    let mut output = vec![0usize; n];
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            input[i1 * n2 + i2] = (i1 * n2 + i2 * n1) % n;
        }
    }
    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let index = (k1 * n2 * inv_n2_n1 + k2 * n1 * inv_n1_n2) % n;
            output[k2 * n1 + k1] = index;
        }
    }
    assert!(
        input.iter().chain(&output).all(|&index| index < n),
        "invariant: PFA permutation entries stay below n = n1 * n2"
    );
    PfaPermutation {
        input: input.into_boxed_slice(),
        output: output.into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use super::cached_pfa_perm;
    use std::sync::Arc;

    #[test]
    fn index_reuses_only_live_permutation_owners() {
        let first = cached_pfa_perm(5, 8);
        let again = cached_pfa_perm(5, 8);
        assert!(Arc::ptr_eq(&first, &again));

        let released = Arc::downgrade(&first);
        drop(first);
        drop(again);
        assert!(released.upgrade().is_none());
    }
}
