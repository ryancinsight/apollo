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

use super::radix::{butterfly_rows, Dif2, Dif4, Lane, Rows, Seams};
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
    /// Planar four-step twiddles multiplied into the first stage's loads, or
    /// `None`. Row-major with row stride `batch`, rows in the same *natural*
    /// order the data rows now carry — the mirror of the bit-reversed planes
    /// the decimation-in-time set required.
    pub(super) fold: Option<(&'a [T], &'a [T])>,
    /// Interleaved output written by the last pass in place of the planes,
    /// or `None` to leave the result in the planes. Rows of `batch`
    /// complexes as `2 * batch` reals; plane row `p` lands in output row
    /// `rev(p)`, the permutation the reinterleave pass used to absorb. Writing
    /// here deletes that pass: the last stage pair stores every element
    /// exactly once, and one register interleave plus two interleaved stores
    /// replace the two plane stores.
    pub(super) sink: Option<&'a mut [T]>,
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
            batch: b,
            stride: s,
            len,
        } = self;
        let row_bits = len.trailing_zeros();
        let mut l = len;

        // Widest stage first, two per pass while two remain, then one: the
        // mirror of the time-decimated set's grouping (see its note on why
        // not three). The four-step twiddle rides the first pass, `l == len`,
        // and the interleaved sink the last, whichever radix that turns out
        // to be. Stage `l` holds `W_l^j` at `l / 2 - 1`.
        while l >= 4 {
            let quarter = l >> 2;
            let groups = len / l;
            let wide = (l >> 1) - 1;
            let narrow = quarter - 1;
            let pass_fold = if l == len { fold } else { None };
            // The pair `(4, 2)` is the last pass whenever the stage count is
            // even; an odd count leaves one radix-2 pass after it.
            let last = l == 4;
            for j in 0..quarter {
                let tws = [tw[wide + j], tw[wide + j + quarter], tw[narrow + j]];
                let twv = tws.map(|(wr, wi)| (simd.splat(wr), simd.splat(wi)));
                for g in 0..groups {
                    let rows = Rows {
                        first: g * l + j,
                        step: quarter,
                    };
                    butterfly_rows::<T, A, Dif4, 4, 3>(
                        re,
                        im,
                        rows,
                        s,
                        b,
                        row_bits,
                        &tws,
                        &twv,
                        Seams::frequency(pass_fold, if last { sink.as_deref_mut() } else { None }),
                    );
                }
            }
            l >>= 2;
        }

        if l >= 2 {
            let half = l >> 1;
            let groups = len / l;
            let base = half - 1;
            let pass_fold = if l == len { fold } else { None };
            for j in 0..half {
                let tws = [tw[base + j]];
                let twv = tws.map(|(wr, wi)| (simd.splat(wr), simd.splat(wi)));
                for g in 0..groups {
                    let rows = Rows {
                        first: g * l + j,
                        step: half,
                    };
                    butterfly_rows::<T, A, Dif2, 2, 1>(
                        re,
                        im,
                        rows,
                        s,
                        b,
                        row_bits,
                        &tws,
                        &twv,
                        Seams::frequency(pass_fold, sink.as_deref_mut()),
                    );
                }
            }
        }
    }
}
