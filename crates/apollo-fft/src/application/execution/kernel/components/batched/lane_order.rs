//! The column order of the planes: the dispatched backend's sub-lane order.
//!
//! A plane row holds one batch column per element. Nothing in the stage sets
//! cares which column sits in which lane, since every column is an
//! independent transform, but the two seams do: the source deinterleaves the
//! caller's `[re, im, re, im, ...]` rows into the planes and the sink
//! interleaves them back. On x86 the one-instruction interleave is the
//! `unpack`, which weaves within 128-bit sub-lanes, so its output holds the
//! columns of a lane group in sub-lane order rather than in memory order; the
//! flat interleave hermes offers restores memory order with a cross-lane
//! permute per operand, which ADR 0056's attribution put on the critical
//! path of both seam sweeps.
//!
//! The planes therefore hold each aligned group of `lanes` columns in the
//! order the sub-lane deinterleave produces, and every kernel that reads or
//! writes a plane by column index goes through [`LaneOrder::column`]: the
//! seams are then the sub-lane unpacks alone, the fold planes are built in
//! the same order, the transpose permutes its tile rows to keep data rows in
//! natural order, and the odd-power decimation and combine index through it.
//! [`LaneOrder::column`] maps a plane column to its memory column and
//! [`LaneOrder::plane`] back; the order is an involution at the AVX2
//! widths (`0, 2, 1, 3`) and not at the AVX-512 ones (`0, 4, 1, 5, 2, 6,
//! 3, 7` for eight `f64` lanes), so both maps exist and the transpose,
//! whose two sides carry one each, never assumes one serves for both.
//!
//! Which order applies is a property of the backend `vectorize` selects for
//! the scalar type on this host, and every planar kernel dispatches through
//! that same selector, so the probe here and the seams agree by
//! construction. Where the batch is narrower than a register the vector
//! path never runs and the order is the identity.

use core::marker::PhantomData;

use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdPermute, SimdStorage};

/// The permutation between memory column order and plane lane order within
/// each aligned group of `1 << shift` columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaneOrder {
    /// Log2 of the columns per group, the dispatched register's lane count;
    /// zero where the order is the identity.
    shift: u32,
    /// The memory column of each plane column within a group.
    table: [u8; LANE_GROUP_CAPACITY],
    /// The plane column of each memory column within a group: the inverse
    /// of `table`.
    inverse: [u8; LANE_GROUP_CAPACITY],
}

/// Widest lane group a plane can be ordered by: sixteen `f32` lanes
/// (AVX-512), the widest register this crate's scalars dispatch to.
pub(crate) const LANE_GROUP_CAPACITY: usize = 16;

/// Reads the dispatched backend's lane geometry.
struct Geometry<T>(PhantomData<T>);

impl<T: LaneScalar> LaneKernel<T> for Geometry<T> {
    type Output = (usize, usize);

    #[inline]
    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> (usize, usize) {
        (
            <A as SimdStorage<T>>::LANE_COUNT,
            <A as SimdPermute<T>>::SUBLANE_LANES,
        )
    }
}

/// The memory column at plane lane `lane` of a `lanes`-lane register whose
/// sub-lanes hold `sublane` lanes: sub-lane `s` position `p` of the
/// deinterleaved register carries memory column `s * sublane / 2 + p %
/// (sublane / 2)`, offset by half the group when `p` is in the sub-lane's
/// second half.
const fn column_of(lanes: usize, sublane: usize, lane: usize) -> usize {
    if sublane == lanes {
        return lane;
    }
    let half = sublane / 2;
    let (sub, pos) = (lane / sublane, lane % sublane);
    let offset = if pos < half { 0 } else { lanes / 2 };
    offset + sub * half + pos % half
}

/// The plane column order of a `LANES`-lane register in sub-lanes of
/// `sublane` lanes, as a table: the form a kernel that knows its backend at
/// compile time folds into constant row offsets.
pub(crate) const fn sublane_order<const LANES: usize>(sublane: usize) -> [usize; LANES] {
    let mut table = [0usize; LANES];
    let mut lane = 0;
    while lane < LANES {
        table[lane] = column_of(LANES, sublane, lane);
        lane += 1;
    }
    table
}

/// The inverse of [`sublane_order`]: the plane lane holding each memory
/// column of a `LANES`-lane register in sub-lanes of `sublane` lanes.
pub(crate) const fn sublane_inverse<const LANES: usize>(sublane: usize) -> [usize; LANES] {
    let order = sublane_order::<LANES>(sublane);
    let mut inverse = [0usize; LANES];
    let mut lane = 0;
    while lane < LANES {
        inverse[order[lane]] = lane;
        lane += 1;
    }
    inverse
}

impl LaneOrder {
    /// Memory order: every column in its own place.
    pub(crate) const IDENTITY: Self = Self {
        shift: 0,
        table: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        inverse: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    };

    /// The order of a plane whose rows hold `batch` columns, under the
    /// backend `vectorize` selects for `T`.
    pub(crate) fn for_batch<T: LaneScalar>(batch: usize) -> Self {
        let (lanes, sublane) = hermes_simd::vectorize(Geometry::<T>(PhantomData));
        if batch < lanes {
            Self::IDENTITY
        } else {
            Self::from_geometry(lanes, sublane)
        }
    }

    /// The order of a register of `lanes` lanes in sub-lanes of `sublane`
    /// ([`column_of`] per lane): `0, 2, 1, 3` for four `f64` lanes of
    /// two-lane sub-lanes, `0, 1, 4, 5, 2, 3, 6, 7` for eight `f32` lanes
    /// of four, `0, 4, 1, 5, 2, 6, 3, 7` for eight `f64` lanes of two and
    /// `0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15` for sixteen
    /// `f32` lanes of four.
    ///
    /// # Panics
    ///
    /// If `lanes` is not a power of two within [`LANE_GROUP_CAPACITY`] or
    /// `sublane` does not divide it evenly.
    pub(crate) fn from_geometry(lanes: usize, sublane: usize) -> Self {
        if sublane == lanes {
            // Memory order, but the group is still the register: the fold
            // table's fine level is one register wide.
            return Self {
                shift: lanes.trailing_zeros(),
                table: Self::IDENTITY.table,
                inverse: Self::IDENTITY.inverse,
            };
        }
        assert!(
            lanes.is_power_of_two() && lanes <= LANE_GROUP_CAPACITY && lanes % sublane == 0,
            "invariant: dispatched registers hold a power of two of at most {LANE_GROUP_CAPACITY} lanes in whole sub-lanes"
        );
        let mut table = Self::IDENTITY.table;
        let mut inverse = Self::IDENTITY.inverse;
        for lane in 0..lanes {
            let column = column_of(lanes, sublane, lane);
            table[lane] = u8::try_from(column).expect("invariant: a column index below sixteen");
            inverse[column] = u8::try_from(lane).expect("invariant: a lane index below sixteen");
        }
        Self {
            shift: lanes.trailing_zeros(),
            table,
            inverse,
        }
    }

    /// Columns per lane group: the dispatched register's lanes, or one
    /// where the batch is narrower than a register.
    pub(crate) fn lanes(self) -> usize {
        1 << self.shift
    }

    /// The memory column held at plane column `physical`.
    #[inline]
    pub(crate) fn column(self, physical: usize) -> usize {
        let mask = (1usize << self.shift) - 1;
        (physical & !mask) | usize::from(self.table[physical & mask])
    }

    /// The plane column holding memory column `column`: the inverse of
    /// [`column`](Self::column).
    #[inline]
    pub(crate) fn plane(self, column: usize) -> usize {
        let mask = (1usize << self.shift) - 1;
        (column & !mask) | usize::from(self.inverse[column & mask])
    }
}

#[cfg(test)]
mod tests {
    use super::LaneOrder;
    use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

    /// Deinterleaves the sample indices `0..lanes` and reports which memory
    /// column each plane lane received.
    struct Received<T>(core::marker::PhantomData<T>);

    impl<T: LaneScalar + From<u8> + Into<f64>> LaneKernel<T> for Received<T> {
        type Output = Vec<usize>;

        #[inline]
        fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> Vec<usize> {
            let lanes = <A as SimdStorage<T>>::LANE_COUNT;
            // Sample `c` as `(c, c + 100)`, so the real plane reads the column
            // and the imaginary plane must read it plus one hundred.
            let interleaved: Vec<T> = (0..lanes)
                .flat_map(|c| {
                    let c = u8::try_from(c).expect("lane count fits u8");
                    [T::from(c), T::from(c + 100)]
                })
                .collect();
            let a = Vector::<T, A>::load_unaligned_from_slice(&interleaved[..lanes])
                .expect("invariant: the slice holds one register");
            let b = Vector::<T, A>::load_unaligned_from_slice(&interleaved[lanes..])
                .expect("invariant: the slice holds one register");
            let (re, im) = a.deinterleave_sublanes(b);
            let mut re_out = vec![T::from(0); lanes];
            let mut im_out = vec![T::from(0); lanes];
            re.store_unaligned_to_slice(&mut re_out)
                .expect("invariant: the slice holds one register");
            im.store_unaligned_to_slice(&mut im_out)
                .expect("invariant: the slice holds one register");
            re_out
                .into_iter()
                .zip(im_out)
                .map(|(r, i)| {
                    let (r, i): (f64, f64) = (r.into(), i.into());
                    assert!((i - r - 100.0).abs() < 0.5, "planes disagree on a column");
                    r as usize
                })
                .collect()
        }
    }

    fn order_matches_the_dispatched_deinterleave<T: LaneScalar + From<u8> + Into<f64>>() {
        let received = hermes_simd::vectorize(Received::<T>(core::marker::PhantomData));
        let lanes = received.len();
        let order = LaneOrder::for_batch::<T>(lanes);
        for (physical, &column) in received.iter().enumerate() {
            assert_eq!(order.column(physical), column, "lane {physical}");
            assert_eq!(order.plane(column), physical, "inverse at {physical}");
        }
        // Groups repeat: the second group is the first shifted by `lanes`.
        for physical in 0..lanes {
            assert_eq!(
                order.column(physical + lanes),
                order.column(physical) + lanes
            );
        }
        assert_eq!(LaneOrder::for_batch::<T>(lanes / 2), LaneOrder::IDENTITY);
    }

    #[test]
    fn order_matches_the_dispatched_deinterleave_in_both_precisions() {
        order_matches_the_dispatched_deinterleave::<f64>();
        order_matches_the_dispatched_deinterleave::<f32>();
    }

    #[test]
    fn constant_tables_match_the_runtime_order() {
        assert_eq!(super::sublane_order::<4>(2), [0, 2, 1, 3]);
        assert_eq!(super::sublane_order::<8>(4), [0, 1, 4, 5, 2, 3, 6, 7]);
        assert_eq!(super::sublane_order::<4>(4), [0, 1, 2, 3]);
        for (lanes, sublane) in [(2, 2), (4, 2), (8, 4), (8, 2), (16, 4)] {
            let order = LaneOrder::from_geometry(lanes, sublane);
            for lane in 0..lanes {
                assert_eq!(order.column(lane), super::column_of(lanes, sublane, lane));
                assert_eq!(order.plane(order.column(lane)), lane);
                assert_eq!(order.column(order.plane(lane)), lane);
            }
        }
        assert_eq!(super::sublane_inverse::<8>(2), [0, 2, 4, 6, 1, 3, 5, 7]);
        assert_eq!(
            super::sublane_inverse::<16>(4),
            [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]
        );
    }

    /// The register relabeling the vector transpose applies, on a scalar
    /// model of the tile: rows fed in `order`, transposed, read back in the
    /// inverse, must land `out[k][l] = in[order(l)][inverse(k)]` at every
    /// width, the AVX-512 ones included, where the order is no involution.
    fn relabeled_transpose_holds<const LANES: usize>(sublane: usize) {
        let order = super::sublane_order::<LANES>(sublane);
        let inverse = super::sublane_inverse::<LANES>(sublane);
        let tile: [[usize; LANES]; LANES] =
            core::array::from_fn(|r| core::array::from_fn(|c| 100 * r + c));
        let ordered: [[usize; LANES]; LANES] = core::array::from_fn(|k| tile[order[k]]);
        let transposed: [[usize; LANES]; LANES] =
            core::array::from_fn(|r| core::array::from_fn(|c| ordered[c][r]));
        let out: [[usize; LANES]; LANES] = core::array::from_fn(|k| transposed[inverse[k]]);
        for k in 0..LANES {
            for l in 0..LANES {
                assert_eq!(
                    out[k][l], tile[order[l]][inverse[k]],
                    "{LANES} lanes at ({k},{l})"
                );
            }
        }
    }

    #[test]
    fn relabeled_transpose_holds_at_every_width() {
        relabeled_transpose_holds::<2>(2);
        relabeled_transpose_holds::<4>(2);
        relabeled_transpose_holds::<4>(4);
        relabeled_transpose_holds::<8>(4);
        relabeled_transpose_holds::<8>(2);
        relabeled_transpose_holds::<16>(4);
    }

    #[test]
    fn identity_is_the_identity() {
        for c in 0..40 {
            assert_eq!(LaneOrder::IDENTITY.column(c), c);
        }
    }

    #[test]
    fn documented_orders_hold() {
        let four = LaneOrder::from_geometry(4, 2);
        assert_eq!(
            (0..4).map(|c| four.column(c)).collect::<Vec<_>>(),
            [0, 2, 1, 3]
        );
        let eight = LaneOrder::from_geometry(8, 4);
        assert_eq!(
            (0..8).map(|c| eight.column(c)).collect::<Vec<_>>(),
            [0, 1, 4, 5, 2, 3, 6, 7]
        );
        assert_eq!(eight.column(13), 8 + 3);
        let flat = LaneOrder::from_geometry(4, 4);
        assert_eq!(flat.lanes(), 4);
        assert_eq!(
            (0..8).map(|c| flat.column(c)).collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        let sixteen = LaneOrder::from_geometry(16, 4);
        assert_eq!(
            (0..16).map(|c| sixteen.column(c)).collect::<Vec<_>>(),
            [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
        );
        // Neither AVX-512 order is an involution: `plane` is its own map.
        let wide = LaneOrder::from_geometry(8, 2);
        assert_eq!(
            (0..8).map(|c| wide.column(c)).collect::<Vec<_>>(),
            [0, 4, 1, 5, 2, 6, 3, 7]
        );
        assert_eq!(
            (0..8).map(|c| wide.plane(c)).collect::<Vec<_>>(),
            [0, 2, 4, 6, 1, 3, 5, 7]
        );
        assert_eq!(wide.column(wide.column(1)), 2);
        assert_eq!(sixteen.plane(8), 2);
    }
}
