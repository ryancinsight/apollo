/// Crossover for executing independent four-step rows through Moirai.
///
/// This remains separate from the algorithm-selection crossovers: scheduler
/// economics may change without proving that a different transform route wins.
pub(crate) const PARALLEL_ROW_THRESHOLD: usize = 65_536;

/// Required complex elements for the complete four-step call tree.
///
/// Planar kernels own padded planes, two pairs for an odd power. A generic
/// square uses one transpose buffer. An unfused odd split past the planar
/// domain holds its gathered input while its halves reuse a child workspace
/// sequentially. No nested call acquires storage.
/// Invalid lengths or an unrepresentable combined capacity return `None`.
pub(crate) fn scratch_len(n: usize) -> Option<usize> {
    use crate::application::execution::kernel::components::batched;
    use crate::application::execution::kernel::pot::{FourStep, PotRoute};

    if !FourStep::admits(n) {
        return None;
    }
    if batched::planar_applies(n) {
        return Some(batched::scratch_len(n));
    }
    if n.trailing_zeros() % 2 == 1 && n >= 512 {
        return n.checked_add(scratch_len(n / 2)?);
    }
    Some(n)
}
