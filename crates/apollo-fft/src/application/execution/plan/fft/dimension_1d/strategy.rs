use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::kernel::pot::{SizedPoT, StockhamAutosort};
use core::marker::PhantomData;
use std::borrow::Cow;
use std::sync::Arc;

/// Selects four-step only after the direct and sized plan routes end.
///
/// Both plan forms have dedicated power-of-two entries through 1024. Their
/// scalar-specific sized policy and base kernels remain independent of the
/// generic route's crossover.
#[inline]
pub(in crate::application::execution::plan::fft) fn generic_four_step_applies(n: usize) -> bool {
    const LARGEST_SIZED_PLAN_LENGTH: usize = 1024;
    n > LARGEST_SIZED_PLAN_LENGTH
        && crate::application::execution::kernel::pot::one_dimensional_uses_four_step(n)
}

pub(crate) type CompositeRadices = Cow<'static, [usize]>;

#[inline]
pub(crate) fn arc_to_cow(arc: Arc<[usize]>) -> CompositeRadices {
    Cow::Owned(arc.to_vec())
}

/// Reusable 1D FFT plan strategy generic over `MixedRadixScalar`.
pub(crate) enum PlanStrategy<F: MixedRadixScalar> {
    Identity,
    ShortWinograd,
    /// The route acquires its own row, matrix, and optional combine tables.
    FourStep,
    PowerOfTwo {
        twiddle_fwd: Option<Arc<[F::Complex]>>,
        log2: u32,
        pot: PhantomData<SizedPoT<StockhamAutosort, 0>>,
    },
    GoodThomas {
        n1: usize,
        n2: usize,
    },
    Composite {
        radices: Cow<'static, [usize]>,
    },
    Rader,
    /// Chirp-z. Serves any length, so it is the terminal route for lengths no
    /// shaped strategy accepts.
    Bluestein,
}

impl<F: MixedRadixScalar> Clone for PlanStrategy<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::ShortWinograd => Self::ShortWinograd,
            Self::FourStep => Self::FourStep,
            Self::PowerOfTwo {
                twiddle_fwd,
                log2,
                pot,
            } => Self::PowerOfTwo {
                twiddle_fwd: twiddle_fwd.clone(),
                log2: *log2,
                pot: *pot,
            },
            Self::GoodThomas { n1, n2 } => Self::GoodThomas { n1: *n1, n2: *n2 },
            Self::Composite { radices } => Self::Composite {
                radices: Cow::clone(radices),
            },
            Self::Rader => Self::Rader,
            Self::Bluestein => Self::Bluestein,
        }
    }
}
