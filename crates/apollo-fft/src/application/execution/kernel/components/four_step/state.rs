//! Operation-owned tables for one four-step transform length.

use crate::application::execution::kernel::components::batched::{planar_applies, PlanarState};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use std::sync::{Arc, OnceLock};

pub(super) struct RowTables<C> {
    pub(super) first: Arc<[C]>,
    pub(super) second: Arc<[C]>,
}

pub(super) enum Decomposition<F: MixedRadixScalar> {
    Planar(PlanarState<F>),
    Odd {
        child: Box<FourStepState<F>>,
        combine: OnceLock<Arc<[F::Complex]>>,
    },
    Matrix {
        first_len: usize,
        second_len: usize,
        rows: [OnceLock<RowTables<F::Complex>>; 2],
        matrix: OnceLock<Arc<[F::Complex]>>,
    },
}

/// Lazily initialized tables retained by a four-step plan or operation.
pub(crate) struct FourStepState<F: MixedRadixScalar> {
    pub(super) len: usize,
    pub(super) decomposition: Decomposition<F>,
}

impl<F: MixedRadixScalar<Complex = eunomia::Complex<F>>> FourStepState<F> {
    #[must_use]
    pub(crate) fn new(len: usize) -> Self {
        assert!(
            len.is_power_of_two() && len >= 4,
            "invariant: four-step state requires a power of two of at least four"
        );
        let decomposition = if planar_applies(len) {
            Decomposition::Planar(PlanarState::new(len))
        } else if len.trailing_zeros() % 2 == 1 && len >= 512 {
            Decomposition::Odd {
                child: Box::new(Self::new(len / 2)),
                combine: OnceLock::new(),
            }
        } else {
            let log2 = len.trailing_zeros();
            let first_len = 1usize << (log2 / 2);
            let second_len = 1usize << (log2 - log2 / 2);
            Decomposition::Matrix {
                first_len,
                second_len,
                rows: [OnceLock::new(), OnceLock::new()],
                matrix: OnceLock::new(),
            }
        };
        Self { len, decomposition }
    }

    pub(super) fn row_tables<const INVERSE: bool>(
        rows: &[OnceLock<RowTables<F::Complex>>; 2],
        first_len: usize,
        second_len: usize,
    ) -> &RowTables<F::Complex> {
        rows[usize::from(INVERSE)].get_or_init(|| RowTables {
            first: if INVERSE {
                F::cached_twiddle_inv(first_len)
            } else {
                F::cached_twiddle_fwd(first_len)
            },
            second: if INVERSE {
                F::cached_twiddle_inv(second_len)
            } else {
                F::cached_twiddle_fwd(second_len)
            },
        })
    }
}
