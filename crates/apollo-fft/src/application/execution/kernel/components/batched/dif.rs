//! Decimation-in-frequency stage set for the planar four-step's second axis.
//!
//! ## Why a second stage set exists
//!
//! The deinterleave earns bit-reversed rows for free by writing each row to
//! `rev(row)`, which is exactly the order a decimation-in-time stage set wants.
//! The transpose between the two axes then destroys that order, so the route
//! used to run a whole pass restoring it — 4.9 to 6.2% of the route, on the
//! order of the transpose itself, and its entire content was undoing the pass
//! before it (`gap_audit.md#planar-pass-attribution`).
//!
//! Decimation in frequency inverts both ends of that. It consumes natural
//! order, which is what the transpose leaves, and produces bit-reversed order,
//! which the sink absorbs for free by reading `rev(row)` — the same trick the
//! deinterleave already uses on the source side. The repair pass has nowhere
//! left to be.
//!
//! ## Why it is a sibling rather than a parameter
//!
//! The two are different algorithms, not two configurations of one. DIT
//! multiplies the odd operand *before* the butterfly and DIF multiplies the
//! difference *after* it, so nearly every arithmetic line differs; a const
//! parameter would monomorphize to these same two bodies while making both
//! harder to read. What they genuinely share — the twiddle table, the load and
//! store helpers, the fold contract — is shared.
//!
//! The table is shared exactly: stage sub-length `l` occupies `l / 2` entries
//! at offset `l / 2 - 1`, holding `W_l^j`. DIT walks those stages upward and
//! DIF downward, over the same values.

use super::radix::Lane;
use super::sweep::{sweep_frequency, sweep_lengths_descending};
use super::FourStepFold;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel};

/// All stages of `batch` independent length-`len` transforms over planar data,
/// decimated in frequency.
///
/// Input rows in natural order; output rows bit-reversed. Dispatch happens once
/// for the whole stage set, as [`super::BatchedStages`] documents.
pub(super) struct BatchedStagesDif<'a, T> {
    pub(super) re: &'a mut [T],
    pub(super) im: &'a mut [T],
    pub(super) tw: &'a [(T, T)],
    /// The four-step twiddle tables multiplied into the first stage's loads,
    /// or `None`; rows in the same natural order the data rows carry, the
    /// mirror of the bit-reversed planes the decimation-in-time set required.
    pub(super) fold: Option<&'a FourStepFold<T>>,
    /// Interleaved output written by the last pass in place of the planes,
    /// or `None` to leave the result in the planes. Rows of `batch`
    /// complexes as `2 * batch` reals; plane row `p` lands in output row
    /// `rev(p)`, the permutation the reinterleave pass used to absorb. Writing
    /// here deletes that pass: the last stage pair stores every element
    /// exactly once, and one register interleave plus two interleaved stores
    /// replace the two plane stores.
    pub(super) sink: Option<&'a mut [T]>,
    /// One tile block of interleaved rows for the sink, as reals.
    pub(super) staging: &'a mut [T],
    pub(super) batch: usize,
    pub(super) stride: usize,
    pub(super) len: usize,
}

impl<T> LaneKernel<T> for BatchedStagesDif<'_, T>
where
    T: LaneScalar + MixedRadixScalar + Lane,
{
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "large LaneKernel::call body must fold into the dispatcher's target-feature scope"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let Self {
            re,
            im,
            tw,
            fold,
            mut sink,
            staging,
            batch: b,
            stride: s,
            len,
        } = self;
        // Widest stage first, in sweeps of up to `SWEEP_STAGES` stages per
        // trip through the planes, two per pass while two remain and then
        // one: the mirror of the time-decimated set's grouping over the
        // same L1-resident tiles (see [`super::sweep`]), with the odd
        // remainder taken first so the sink rides a full tile. The four-step
        // twiddle rides the pass over stage `len` and the interleaved sink
        // the pass over stage 2. Stage `l` holds `W_l^j` at `l / 2 - 1`.
        let mut l_top = len;
        for (index, stages) in sweep_lengths_descending(len.trailing_zeros()).enumerate() {
            sect!(super::FREQUENCY_SWEEPS[index], {
                sweep_frequency(
                    re,
                    im,
                    tw,
                    fold,
                    sink.as_deref_mut(),
                    staging,
                    b,
                    s,
                    len,
                    l_top,
                    stages,
                    simd,
                );
            });
            l_top >>= stages;
        }
    }
}
