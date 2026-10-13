//! Plan-owned Rader permutation and convolution tables.

use super::bluestein::{cached_bluestein_entry, rader_bluestein_convolve_with};
use super::convolution::{
    rader_convolve_with_state, rader_negacyclic_convolve_with_state, ConvolutionState,
};
use super::{cached_generator_order, prefers_bluestein_for_rader, prefers_half_cyclic_for_rader};
use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use std::sync::{Arc, OnceLock};

enum RaderTables<F: MixedRadixScalar> {
    Full {
        spectrum: Arc<[F::Complex]>,
        convolution: ConvolutionState<F>,
    },
    Half {
        spectra: crate::application::execution::kernel::mixed_radix::caches::rader::NegacyclicEntry<
            F::Complex,
        >,
        twiddles: Arc<[F::Complex]>,
        convolution: ConvolutionState<F>,
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
            let spectra = F::cached_rader_negacyclic_spectra::<INVERSE>(n, generator_inverse);
            Self::Half {
                spectra,
                twiddles: F::cached_rader_neg_twiddles((n - 1) / 2),
                convolution: ConvolutionState::new((n - 1) / 2),
            }
        } else {
            Self::Full {
                spectrum: F::cached_rader_spectrum::<INVERSE>(n, generator_inverse),
                convolution: ConvolutionState::new(n - 1),
            }
        }
    }

    fn convolve(&self, data: &mut [F::Complex], n: usize) {
        match self {
            Self::Full {
                spectrum,
                convolution,
            } => rader_convolve_with_state::<F>(data, spectrum, convolution),
            Self::Half {
                spectra,
                twiddles,
                convolution,
            } => rader_negacyclic_convolve_with_state::<F>(
                data,
                &spectra.0,
                &spectra.1,
                twiddles,
                convolution,
            ),
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
        Self::from_generator(n, generator, generator_inverse)
    }

    pub(crate) fn from_generator(n: usize, generator: usize, generator_inverse: usize) -> Self {
        Self {
            n,
            generator_inverse,
            gather: cached_generator_order(n, generator),
            forward: OnceLock::new(),
            inverse: OnceLock::new(),
        }
    }

    pub(crate) fn execute<const INVERSE: bool>(&self, data: &mut [F::Complex]) {
        let tables = self.tables::<INVERSE>();
        let x0 = data[0];
        F::with_rader_padded_scratch(self.n - 1, |padded| {
            let sum_x = super::gather_sum_slice::<F>(data, padded, &self.gather);
            tables.convolve(padded, self.n);
            data[0] = x0 + sum_x;
            super::scatter_slice::<F>(data, padded, x0, &self.gather);
        });
    }

    pub(crate) fn execute_ordered<const INVERSE: bool>(&self, data: &mut [F::Complex]) {
        assert!(
            data.len() >= self.n,
            "invariant: the ordered Rader buffer covers its transform"
        );
        let data = &mut data[..self.n];
        let (head, nonzero) = data.split_at_mut(1);
        let x0 = head[0];
        let sum_x: F::Complex = nonzero.iter().copied().sum();
        self.tables::<INVERSE>().convolve(nonzero, self.n);
        head[0] = x0 + sum_x;
        nonzero.iter_mut().for_each(|x| *x = x0 + *x);
    }

    pub(crate) fn order(&self) -> &[usize] {
        &self.gather
    }

    fn tables<const INVERSE: bool>(&self) -> &RaderTables<F> {
        if INVERSE {
            self.inverse
                .get_or_init(|| RaderTables::new::<true>(self.n, self.generator_inverse))
        } else {
            self.forward
                .get_or_init(|| RaderTables::new::<false>(self.n, self.generator_inverse))
        }
    }
}
