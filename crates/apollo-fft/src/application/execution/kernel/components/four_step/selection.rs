use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

/// Runs the shared four-step route when the length admits the selected split.
///
/// Selection lives here so one-dimensional plans and the general mixed-radix
/// dispatcher cannot diverge on split or normalization semantics. The caller
/// supplies its measured workload crossover.
#[inline]
pub(crate) fn try_four_step<
    F: MixedRadixScalar<Complex = eunomia::Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [F::Complex],
    minimum_len: usize,
) -> bool {
    use crate::application::execution::kernel::pot::{FourStep, PotRoute};
    // Admission is the route's own property, defined once on `FourStep`, so the
    // general dispatcher and one-dimensional plans cannot drift apart on which
    // lengths the split is valid for. Only the crossover differs between them,
    // and that is what the caller supplies.
    let n = data.len();
    if n < minimum_len || !FourStep::admits(n) {
        return false;
    }

    FourStep::run::<F, INVERSE, NORMALIZE>(data, &[]);
    true
}
