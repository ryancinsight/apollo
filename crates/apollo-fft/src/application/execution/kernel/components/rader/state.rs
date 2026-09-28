//! Plan-owned Rader permutation and convolution tables.

use super::bluestein::{cached_bluestein_entry, rader_bluestein_convolve_with};
use super::convolution::{rader_convolve_inplace, rader_negacyclic_convolve_inplace};
use super::{cached_generator_order, prefers_bluestein_for_rader, prefers_half_cyclic_for_rader};
use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use std::sync::{Arc, OnceLock};

enum RaderTables<F: MixedRadixScalar> {
    Full(Arc<[F::Complex]>),
    Half {
        cyclic: Arc<[F::Complex]>,
        negacyclic: Arc<[F::Complex]>,
        twiddles: Arc<[F::Complex]>,
    },
    Bluestein(Arc<[F::Complex]>),
}

impl<F> RaderTables<F>
where
    F: MixedRadixScalar<Complex = eunomia::Complex<F>> + ShortWinogradScalar,
{
    fn new<const INVERSE: bool>(n: usize, generator_inverse: usize) -> Self {
        if prefers_bluestein_for_rader(n) {
            Self::Bluestein(cached_bluestein_entry::<F, INVERSE>(n, generator_inverse))
        } else if prefers_half_cyclic_for_rader::<F>(n) {
            let (cyclic, negacyclic) =
                F::cached_rader_negacyclic_spectra::<INVERSE>(n, generator_inverse);
            Self::Half {
                cyclic,
                negacyclic,
                twiddles: F::cached_rader_neg_twiddles((n - 1) / 2),
            }
        } else {
            Self::Full(F::cached_rader_spectrum::<INVERSE>(n, generator_inverse))
        }
    }

    fn convolve(&self, data: &mut [F::Complex], n: usize) {
        match self {
            Self::Full(spectrum) => rader_convolve_inplace::<F>(data, spectrum),
            Self::Half {
                cyclic,
                negacyclic,
                twiddles,
            } => rader_negacyclic_convolve_inplace::<F>(data, cyclic, negacyclic, twiddles),
            Self::Bluestein(kernel) => rader_bluestein_convolve_with::<F>(data, n, kernel),
        }
    }
}

/// Prepared state for one reusable prime-length Rader transform.
pub(crate) struct RaderState<F: MixedRadixScalar> {
    n: usize,
    generator_inverse: usize,
    gather: Arc<[usize]>,
    forward: OnceLock<RaderTables<F>>,
    inverse: OnceLock<RaderTables<F>>,
}

impl<F> RaderState<F>
where
    F: MixedRadixScalar<Complex = eunomia::Complex<F>> + ShortWinogradScalar,
{
    pub(crate) fn new(n: usize) -> Self {
        let (generator, generator_inverse) = super::generator::primitive_root_and_inverse(n);
        Self {
            n,
            generator_inverse,
            gather: cached_generator_order(n, generator),
            forward: OnceLock::new(),
            inverse: OnceLock::new(),
        }
    }

    pub(crate) fn execute<const INVERSE: bool>(&self, data: &mut [F::Complex]) {
        let tables = if INVERSE {
            self.inverse
                .get_or_init(|| RaderTables::new::<true>(self.n, self.generator_inverse))
        } else {
            self.forward
                .get_or_init(|| RaderTables::new::<false>(self.n, self.generator_inverse))
        };
        let x0 = data[0];
        F::with_rader_padded_scratch(self.n - 1, |padded| {
            let sum_x = super::gather_sum_slice::<F>(data, padded, &self.gather);
            tables.convolve(padded, self.n);
            data[0] = x0 + sum_x;
            super::scatter_slice::<F>(data, padded, x0, &self.gather);
        });
    }
}
