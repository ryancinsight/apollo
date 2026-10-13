//! Plan-owned composite radix tables.

use super::cache::{CompositeCache, CompositeTables};
use super::{
    forward_inplace_with_tables, inverse_inplace_unnorm_with_tables, inverse_inplace_with_tables,
};
use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;
use eunomia::Complex;
use std::sync::{Arc, OnceLock};

/// Tables retained by a reusable composite radix plan.
pub(crate) struct CompositeState<F: CompositeCache> {
    forward: Arc<CompositeTables<Complex<F>>>,
    inverse: OnceLock<Arc<CompositeTables<Complex<F>>>>,
}

impl<F> CompositeState<F>
where
    F: CompositeCache + ShortWinogradScalar + 'static,
{
    pub(crate) fn new(radices: &[usize]) -> Self {
        Self {
            forward: F::cached_tables::<false>(radices),
            inverse: OnceLock::new(),
        }
    }

    pub(crate) fn forward(&self, data: &mut [Complex<F>]) {
        forward_inplace_with_tables(data, &self.forward.radices, &self.forward);
    }

    pub(crate) fn forward_with_pointwise(
        &self,
        data: &mut [Complex<F>],
        pointwise_spectrum: &[Complex<F>],
    ) {
        super::core::composite_core_with_radices::<
            moirai::AdaptiveWithThreshold<
                { crate::application::execution::kernel::tuning::RADIX_PARALLEL_CHUNK_THRESHOLD },
            >,
            F,
            false,
        >(
            data,
            &self.forward.radices,
            &self.forward,
            Some(pointwise_spectrum),
        );
    }

    pub(crate) fn inverse_unnorm(&self, data: &mut [Complex<F>]) {
        let tables = self
            .inverse
            .get_or_init(|| F::cached_tables::<true>(&self.forward.radices));
        inverse_inplace_unnorm_with_tables(data, &tables.radices, tables);
    }

    pub(crate) fn inverse(&self, data: &mut [Complex<F>]) {
        let tables = self
            .inverse
            .get_or_init(|| F::cached_tables::<true>(&self.forward.radices));
        inverse_inplace_with_tables(data, &tables.radices, tables);
    }
}
