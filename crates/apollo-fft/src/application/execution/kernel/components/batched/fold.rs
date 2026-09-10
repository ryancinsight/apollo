//! The four-step twiddle as the fold tables the second stage set
//! multiplies into its first loads.

use super::lane_order::LaneOrder;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

/// Shortest transform whose fold table is two-level.
///
/// The compact table costs a second complex multiply per element and two
/// broadcasts per row per register, which the fold sweep pays back only
/// where the full matrix would stream from beyond L2: at 262144 `f64` the
/// fold sweep fell from 635k to 369k cycles, at 65536 and 16384 it moved
/// within noise or lost (48k to 58k at 65536 `f32`; ADR 0059). One host's
/// 3 MiB L2 sets the bound; re-measure before moving it.
const COMPACT_FOLD_MIN_LEN: usize = 1 << 18;

/// The four-step twiddle `W_N^(p k)` for data row `p` of `rows` and plane
/// column `k` of `cols` (the second stage set's plane shape, `cols × rows`
/// for an odd power),
/// as two tables whose product is the entry, or as the full matrix below
/// [`COMPACT_FOLD_MIN_LEN`] (the fine table one full row wide, the coarse
/// table one).
///
/// Column `k` holds memory column `c = order(k)`, and `c = G F + f` with `F`
/// the lane group and `f = order(k mod F)`, so `W_N^(p c) = W_N^(p F G) ·
/// W_N^(p f)`: a coarse table of one scalar per row and lane group and a
/// fine table of one register per row, `m (F + m / F)` entries in place of
/// the `m^2` the full matrix held. At 65536 `f64` that is 128 KiB against
/// 1 MiB, the size of the data itself, which the fold sweep streamed from
/// L2 on every call (ADR 0059). Each entry is one direct evaluation, so the
/// product carries the two factors' roundings and the multiply's own:
/// within `4u` of the exact twiddle, against `u` for the full matrix, which
/// the transform's `O(log N · u)` bound absorbs.
pub(crate) struct FourStepFold<T> {
    /// Lanes per group, `F`: the fine table's row width.
    pub(crate) lanes: usize,
    /// `W_N^(p order(f))` for `f < F`, row-major by `p`.
    pub(crate) fine_re: Box<[T]>,
    pub(crate) fine_im: Box<[T]>,
    /// `W_N^(p F G)` for `G < cols / F`, row-major by `p`.
    pub(crate) coarse_re: Box<[T]>,
    pub(crate) coarse_im: Box<[T]>,
}

impl<T: MixedRadixScalar> FourStepFold<T> {
    pub(super) fn new<const INVERSE: bool>(
        n: usize,
        rows: usize,
        cols: usize,
        order: LaneOrder,
    ) -> Self {
        use crate::application::execution::kernel::twiddle_table::twiddle_components;
        let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        // The two levels pay a second complex multiply per element, which
        // wins only once the full matrix would stream from beyond L2; below
        // that the fine table is the full row and the coarse table is one.
        let lanes = if n >= COMPACT_FOLD_MIN_LEN {
            order.lanes().min(cols)
        } else {
            cols
        };
        let groups = cols / lanes;
        // Direct evaluation per entry through the shared authority, as the
        // full matrix was built: mod-`n` reduction and one `sin_cos` each.
        let entry = |exponent: usize| {
            let (sin, cos) = twiddle_components(sign, exponent, n);
            (T::from_precise(cos), T::from_precise(sin))
        };
        let (fine_re, fine_im): (Vec<T>, Vec<T>) = (0..rows)
            .flat_map(|p| (0..lanes).map(move |f| (p, f)))
            .map(|(p, f)| entry(p * order.column(f)))
            .unzip();
        let (coarse_re, coarse_im): (Vec<T>, Vec<T>) = (0..rows)
            .flat_map(|p| (0..groups).map(move |g| (p, g)))
            .map(|(p, g)| entry(p * lanes * g))
            .unzip();
        Self {
            lanes,
            fine_re: fine_re.into_boxed_slice(),
            fine_im: fine_im.into_boxed_slice(),
            coarse_re: coarse_re.into_boxed_slice(),
            coarse_im: coarse_im.into_boxed_slice(),
        }
    }
}
