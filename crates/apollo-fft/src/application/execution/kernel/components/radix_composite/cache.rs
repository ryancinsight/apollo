use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;
use crate::application::execution::kernel::components::winograd::WinogradScalar;
use eunomia::Complex;
use std::cell::RefCell;
use std::sync::{Arc, Weak};
use std::thread::LocalKey;

mod planar;

/// Prepared stage tables for one composite radix sequence and direction.
pub struct CompositeTables<C> {
    pub(crate) radices: Box<[usize]>,
    pub(crate) twiddles: Box<[C]>,
    pub(crate) offsets: Box<[usize]>,
}

struct CompositeTwiddleEntry<C> {
    tables: Weak<CompositeTables<C>>,
}

pub trait CompositeCache: WinogradScalar + ShortWinogradScalar + hermes_simd::LaneScalar {
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R;

    /// Runs the batched-layout four-step transform, reporting whether it applied.
    ///
    /// The batched kernel needs bounds (`LaneScalar`, `Pod`) that this trait's
    /// generic callers do not carry, so the concrete scalars route to it here.
    /// Returns `false` when the length is outside the path's domain, leaving the
    /// caller to take its existing route. The caller supplies the complete
    /// padded workspace; this operation never acquires thread-local scratch.
    fn try_four_step_batched<const INVERSE: bool>(
        data: &mut [Complex<Self>],
        scratch: &mut [Complex<Self>],
    ) -> bool;
    fn cached_tables<const INVERSE: bool>(radices: &[usize])
        -> Arc<CompositeTables<Complex<Self>>>;
}

thread_local! {
    static TL_TWIDDLES_FWD_64: RefCell<Vec<CompositeTwiddleEntry<eunomia::Complex64>>> = const { RefCell::new(Vec::new()) };
    static TL_TWIDDLES_INV_64: RefCell<Vec<CompositeTwiddleEntry<eunomia::Complex64>>> = const { RefCell::new(Vec::new()) };

    static TL_TWIDDLES_FWD_32: RefCell<Vec<CompositeTwiddleEntry<eunomia::Complex32>>> = const { RefCell::new(Vec::new()) };
    static TL_TWIDDLES_INV_32: RefCell<Vec<CompositeTwiddleEntry<eunomia::Complex32>>> = const { RefCell::new(Vec::new()) };

    static TL_COMPOSITE_SCRATCH_64: mnemosyne::scratch::ScratchPool<eunomia::Complex64> =
        const { mnemosyne::scratch::ScratchPool::new() };
    static TL_COMPOSITE_SCRATCH_32: mnemosyne::scratch::ScratchPool<eunomia::Complex32> =
        const { mnemosyne::scratch::ScratchPool::new() };
}

/// Releases idle batched four-step scratch on the current thread.
pub(super) fn release_thread_local_scratch() {
    TL_COMPOSITE_SCRATCH_64.with(|pool| pool.release());
    TL_COMPOSITE_SCRATCH_32.with(|pool| pool.release());
}

#[cfg(test)]
pub(super) fn thread_local_scratch_capacity() -> usize {
    TL_COMPOSITE_SCRATCH_64.with(|pool| pool.capacity())
        + TL_COMPOSITE_SCRATCH_32.with(|pool| pool.capacity())
}

/// Whether the batched-layout four-step covers this length.
///
/// Square splits only, which is what the four-step gate already admits and what
/// lets the middle transpose run in place. The upper bound is where the
/// superseded path begins distributing its row transforms across threads: the
/// batched kernel is single-threaded, and its batch dimension is the innermost
/// one, so splitting it across workers hands each a strided view rather than a
/// contiguous chunk. Extending past this bound is a measurement plus a
/// partitioning design, not a constant change.
#[inline]
fn build_composite_twiddles<F: WinogradScalar, const INVERSE: bool>(
    radices: &[usize],
) -> (Vec<Complex<F>>, Vec<usize>) {
    let sign: f64 = if INVERSE { 1.0 } else { -1.0 };
    // Per-arm layout: (R-1)*prev_len entries per stage.
    // Arm k (k=1..R-1) at stage_offset + (k-1)*prev_len: W^{k*j} for j=0..prev_len-1.
    // Radix-2 stages are unchanged ((2-1)*L = L).
    let total_twiddles: usize = radices
        .iter()
        .scan(1usize, |p, &r| {
            let out = *p * (r - 1);
            *p *= r;
            Some(out)
        })
        .sum();
    let one = Complex::new(F::from_precise(1.0), F::from_precise(0.0));
    let mut all_twiddles = vec![one; total_twiddles];
    let mut stage_offsets = vec![0usize; radices.len()];
    let mut prev_len = 1usize;
    let mut tw_idx = 0;
    let mut offset_idx = 0;
    for &r in radices {
        let stage_len = prev_len * r;
        // SAFETY: `offset_idx` counts the radices walked so far, below `radices.len()`, the
        // length `stage_offsets` was built with.
        unsafe { *stage_offsets.get_unchecked_mut(offset_idx) = tw_idx };
        offset_idx += 1;
        // Arms 1..R-1: arm-k[j] = W^{k*j}, evaluated directly.
        //
        // The exponent is reduced modulo `stage_len` before the angle is
        // formed, so every argument stays inside one period and the
        // library's own argument reduction is never the accuracy limit.
        //
        // The superseded form built arm 1 by the recurrence
        // `tw[j] = tw[j-1] * W_base` and each later arm by multiplying the
        // previous arm, so entry `j` of the last stage carried the rounding
        // of `j` complex multiplications. That is `O(N * u)` twiddle error,
        // and it propagates directly into the transform: the `O(log N * u)`
        // FFT forward-error bound (Higham, *Accuracy and Stability of
        // Numerical Algorithms*, 2nd ed., section 24.1) holds only for
        // accurately computed twiddles and degrades to `O(N * u)` without
        // them. Direct evaluation costs one `sin_cos` per entry at plan
        // build, which a plan amortizes; the recurrence saved that at the
        // cost of the bound.
        for k in 1..r {
            for j in 0..prev_len {
                let (sin, cos) =
                    crate::application::execution::kernel::twiddle_table::twiddle_components(
                        sign,
                        k * j,
                        stage_len,
                    );
                // SAFETY: `tw_idx` counts the entries written so far; the loops write exactly
                // `prev_len * (r - 1)` per stage, the sum `total_twiddles` was built with.
                unsafe {
                    *all_twiddles.get_unchecked_mut(tw_idx) =
                        Complex::new(F::from_precise(cos), F::from_precise(sin));
                }
                tw_idx += 1;
            }
        }
        prev_len = stage_len;
    }
    debug_assert_eq!(tw_idx, total_twiddles);
    debug_assert_eq!(offset_idx, radices.len());
    (all_twiddles, stage_offsets)
}

fn cached_tables<
    F: WinogradScalar + ShortWinogradScalar + hermes_simd::LaneScalar + 'static,
    const INVERSE: bool,
>(
    radices: &[usize],
    local: &'static LocalKey<RefCell<Vec<CompositeTwiddleEntry<Complex<F>>>>>,
) -> Arc<CompositeTables<Complex<F>>> {
    if let Some(tables) = local.with(|cache| {
        cache
            .borrow()
            .iter()
            .filter_map(|entry| entry.tables.upgrade())
            .find(|tables| tables.radices.as_ref() == radices)
    }) {
        return tables;
    }

    let (twiddles, offsets) = build_composite_twiddles::<F, INVERSE>(radices);
    let tables = Arc::new(CompositeTables {
        radices: radices.into(),
        twiddles: twiddles.into_boxed_slice(),
        offsets: offsets.into_boxed_slice(),
    });
    local.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.retain(|entry| entry.tables.strong_count() != 0);
        cache.push(CompositeTwiddleEntry {
            tables: Arc::downgrade(&tables),
        });
    });
    tables
}

impl CompositeCache for f64 {
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R {
        TL_COMPOSITE_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn try_four_step_batched<const INVERSE: bool>(
        data: &mut [Complex<Self>],
        scratch: &mut [Complex<Self>],
    ) -> bool {
        planar::try_four_step::<Self, INVERSE>(data, scratch)
    }

    #[inline]
    fn cached_tables<const INVERSE: bool>(
        radices: &[usize],
    ) -> Arc<CompositeTables<Complex<Self>>> {
        let tl = if INVERSE {
            &TL_TWIDDLES_INV_64
        } else {
            &TL_TWIDDLES_FWD_64
        };
        cached_tables::<Self, INVERSE>(radices, tl)
    }
}

impl CompositeCache for f32 {
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R {
        TL_COMPOSITE_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn try_four_step_batched<const INVERSE: bool>(
        data: &mut [Complex<Self>],
        scratch: &mut [Complex<Self>],
    ) -> bool {
        planar::try_four_step::<Self, INVERSE>(data, scratch)
    }

    #[inline]
    fn cached_tables<const INVERSE: bool>(
        radices: &[usize],
    ) -> Arc<CompositeTables<Complex<Self>>> {
        let tl = if INVERSE {
            &TL_TWIDDLES_INV_32
        } else {
            &TL_TWIDDLES_FWD_32
        };
        cached_tables::<Self, INVERSE>(radices, tl)
    }
}

#[cfg(test)]
mod tests {
    use super::CompositeCache;
    use std::sync::Arc;

    #[test]
    fn table_index_reuses_prepared_state() {
        const RADICES: &[usize] = &[4, 5, 5];
        let first = <f64 as CompositeCache>::cached_tables::<false>(RADICES);
        let again = <f64 as CompositeCache>::cached_tables::<false>(RADICES);
        assert!(Arc::ptr_eq(&first, &again), "prepared state must be reused");
        let released = Arc::downgrade(&first);
        drop(first);
        drop(again);
        assert!(released.upgrade().is_none());
    }
}
