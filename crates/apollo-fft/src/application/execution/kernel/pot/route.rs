//! Power-of-two execution routes, carried as zero-sized types.
//!
//! [`strategies`](super::strategies) already models a route as a ZST but only
//! as a marker; the choice between routes lived as a bare `if` against a tuning
//! constant, which made it impossible to run both routes at one length in one
//! process. That is not a stylistic complaint. It is why two instruments
//! disagreed by an order of magnitude about where the crossover sits
//! (`gap_audit.md#crossover-contradiction`): each measured one route in its own
//! process against the other in a different one, and the difference between
//! processes was larger than the difference between routes.
//!
//! Giving each route a ZST with an executable contract lets the crossover
//! instrument instantiate both at the same length, interleaved, against the
//! same cache state. Selection stays a single branch per transform — not per
//! element — and the types carry no data, so monomorphization leaves nothing
//! behind at run time.

use super::strategies::{PoTStrategy, StockhamAutosort};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;

/// A complete power-of-two transform route.
///
/// Implementors are zero-sized. `run` is generic over the scalar and over both
/// direction flags, so every instantiation compiles to the same code the route
/// would have emitted when it was written inline.
pub(crate) trait PotRoute: PoTStrategy + Default {
    /// Whether this route admits `n`.
    ///
    /// A route that does not admit a length is not merely slower there — it is
    /// incorrect, so this is a precondition rather than a preference.
    fn admits(n: usize) -> bool;

    /// Transforms `data` in place.
    ///
    /// # Panics
    ///
    /// If [`admits`](Self::admits) is false for `data.len()`.
    fn run<F, const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [F::Complex],
        twiddles: &[F::Complex],
    ) where
        F: MixedRadixScalar<Complex = Complex<F>>;
}

impl PotRoute for StockhamAutosort {
    fn admits(n: usize) -> bool {
        n.is_power_of_two()
    }

    fn run<F, const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [F::Complex],
        twiddles: &[F::Complex],
    ) where
        F: MixedRadixScalar<Complex = Complex<F>>,
    {
        let n = data.len();
        debug_assert!(Self::admits(n), "stockham route requires a power of two");
        match n {
            // SAFETY: each arm's const length equals `n`, which is the
            // precondition `small_pot_inplace_sized` documents.
            2 => unsafe { F::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data) },
            4 => unsafe { F::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data) },
            8 => unsafe { F::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data) },
            16 => unsafe { F::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data) },
            32 => unsafe { F::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data) },
            64 => unsafe { F::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data) },
            _ => <F as MixedRadixScalar>::with_scratch(n, |scratch| {
                if INVERSE && NORMALIZE {
                    F::stockham_forward_normalized(data, scratch, twiddles, n);
                } else {
                    F::stockham_forward(data, scratch, twiddles);
                }
            }),
        }
    }
}

/// Four-step decomposition: `N = N1 x N2` with both sub-transforms cache
/// resident, reaching the batched layout below the threading threshold.
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct FourStep;

impl PoTStrategy for FourStep {}

impl PotRoute for FourStep {
    /// Square splits only. An odd `log2` would need an asymmetric split, whose
    /// cost this route has never been measured at.
    fn admits(n: usize) -> bool {
        n.is_power_of_two() && n >= 4 && n.trailing_zeros() % 2 == 0
    }

    fn run<F, const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [F::Complex],
        _twiddles: &[F::Complex],
    ) where
        F: MixedRadixScalar<Complex = Complex<F>>,
    {
        let n = data.len();
        debug_assert!(Self::admits(n), "four-step route requires a square split");
        crate::application::execution::kernel::components::four_step::four_step_fft::<F, INVERSE>(
            data,
        );
        if INVERSE && NORMALIZE {
            F::normalize(data, n);
        }
    }
}

/// Whether a standalone one-dimensional plan of length `n` takes the four-step
/// route, against the crossover the decision record carries.
///
/// The single place the threshold is consulted, so the constant, the routes and
/// the instrument that measures them cannot drift apart.
#[inline]
pub(crate) fn one_dimensional_uses_four_step(n: usize) -> bool {
    FourStep::admits(n)
        && n >= crate::application::execution::kernel::tuning::ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD
}
