use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::kernel::pot::{SizedPoT, StockhamAutosort};
use core::marker::PhantomData;
use std::borrow::Cow;
use std::sync::Arc;

pub(crate) type CompositeRadices = Cow<'static, [usize]>;

#[inline]
pub(crate) fn arc_to_cow(arc: Arc<[usize]>) -> CompositeRadices {
    Cow::Owned(arc.to_vec())
}

/// Reusable 1D FFT plan strategy generic over `MixedRadixScalar`.
pub(crate) enum PlanStrategy<F: MixedRadixScalar> {
    Identity,
    ShortWinograd,
    PowerOfTwo {
        twiddle_fwd: Arc<[F::Complex]>,
        twiddle_inv: Arc<[F::Complex]>,
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
}

impl<F: MixedRadixScalar> Clone for PlanStrategy<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::ShortWinograd => Self::ShortWinograd,
            Self::PowerOfTwo {
                twiddle_fwd,
                twiddle_inv,
                log2,
                pot,
            } => Self::PowerOfTwo {
                twiddle_fwd: twiddle_fwd.clone(),
                twiddle_inv: twiddle_inv.clone(),
                log2: *log2,
                pot: *pot,
            },
            Self::GoodThomas { n1, n2 } => Self::GoodThomas { n1: *n1, n2: *n2 },
            Self::Composite { radices } => Self::Composite {
                radices: Cow::clone(radices),
            },
            Self::Rader => Self::Rader,
        }
    }
}
