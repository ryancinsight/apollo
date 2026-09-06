//! Shared axis-plan ownership of direction-specific power-of-two twiddles.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use std::sync::Arc;

/// Retains the selected direction's table only for nontrivial power-of-two axes.
#[inline]
pub(super) fn cached_power_of_two_twiddle<F, const FORWARD: bool>(
    n: usize,
) -> Option<Arc<[F::Complex]>>
where
    F: MixedRadixScalar,
{
    if n <= 1 || !n.is_power_of_two() {
        return None;
    }
    Some(if FORWARD {
        F::cached_twiddle_fwd(n)
    } else {
        F::cached_twiddle_inv(n)
    })
}
