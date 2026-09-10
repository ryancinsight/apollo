//! Per-length batched-transform plan: the stage-major twiddle table both
//! stage sets read.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

/// Per-length batched-transform plan: the stage-major twiddle table, a
/// build-time cost.
///
/// Both stage sets read this one table. The time-decimated set walks its
/// stages upward and the frequency-decimated set downward, over the same
/// values.
pub(crate) struct BatchedPlan<T> {
    pub(super) len: usize,
    /// Stage-major twiddles: stage `s` (sub-transform length `2^(s+1)`) occupies
    /// `2^s` entries, so the table totals `len - 1`.
    pub(super) tw: Vec<(T, T)>,
}

impl<T: MixedRadixScalar> BatchedPlan<T> {
    pub(super) fn new<const INVERSE: bool>(len: usize) -> Self {
        assert!(
            len.is_power_of_two(),
            "batched plan requires a power of two"
        );
        let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        let mut tw = Vec::with_capacity(len.saturating_sub(1));
        let mut l = 2usize;
        while l <= len {
            for j in 0..l / 2 {
                // Direct evaluation per entry: a recurrence here would carry
                // O(N·u) twiddle error, which the forward-error bound this
                // crate documents does not admit.
                let (sin, cos) = (sign * core::f64::consts::TAU * j as f64 / l as f64).sin_cos();
                tw.push((T::from_precise(cos), T::from_precise(sin)));
            }
            l <<= 1;
        }
        Self { len, tw }
    }
}
