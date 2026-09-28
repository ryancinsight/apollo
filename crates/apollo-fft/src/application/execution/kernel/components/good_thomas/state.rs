//! Plan-owned Good-Thomas permutation tables.

use super::pfa_fft_with_owner;
use crate::application::execution::kernel::mixed_radix::caches::pfa::PfaPermutation;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

/// Prepared state for one reusable Good-Thomas factorization.
pub(crate) struct PfaState<F: MixedRadixScalar> {
    n1: usize,
    n2: usize,
    tables: OnceLock<Arc<PfaPermutation>>,
    scalar: PhantomData<F>,
}

impl<F: MixedRadixScalar<Complex = eunomia::Complex<F>>> PfaState<F> {
    pub(crate) fn new(n1: usize, n2: usize) -> Self {
        Self {
            n1,
            n2,
            tables: OnceLock::new(),
            scalar: PhantomData,
        }
    }

    pub(crate) fn execute<const INVERSE: bool>(&self, data: &mut [F::Complex]) {
        pfa_fft_with_owner::<F, INVERSE>(data, self.n1, self.n2, &self.tables);
    }
}
