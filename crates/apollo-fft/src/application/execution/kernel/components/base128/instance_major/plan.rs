//! Immutable table construction and directional state for the base kernel.

use super::table_lanes;
use crate::application::execution::kernel::components::lane_capability::native_lanes_supported;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use core::mem::size_of;
use std::sync::OnceLock;

/// Native register layout selected when the plan is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BaseLaneWidth {
    /// Two interleaved complex samples per register.
    Four,
    /// Four interleaved complex samples per register.
    Eight,
}

const fn select_lane_width(
    scalar_bytes: usize,
    eight_lanes_supported: bool,
    four_lanes_supported: bool,
) -> Option<BaseLaneWidth> {
    if scalar_bytes == 4 && eight_lanes_supported {
        Some(BaseLaneWidth::Eight)
    } else if four_lanes_supported {
        Some(BaseLaneWidth::Four)
    } else {
        None
    }
}

pub(crate) struct BasePlan<T, const ROWS: usize, const TABLE_LANES: usize> {
    /// Dup-split twiddles. The fixed-size type is load-bearing, not
    /// decoration: the checked view's `offset + LANE_COUNT <= len` assert
    /// folds only when the length is a compile-time constant, and this
    /// kernel reads the table from inside its hot loops. The sample-major
    /// kernel learned the same lesson first (gap_audit.md#base128-bounds);
    /// this module was shelved before that fix landed and never received it.
    pub(super) table: Box<[T; TABLE_LANES]>,
    /// Register layout used to build `table` and execute this plan.
    pub(super) lane_width: BaseLaneWidth,
    /// `W_8^1` and `W_8^3` as complex values for the eight-row column pass's
    /// splats; unread by the four-row pass, whose radix-4 column transform
    /// needs no multiply beyond its rotation.
    pub(super) col: [[T; 2]; 2],
}

impl<T: MixedRadixScalar, const ROWS: usize, const TABLE_LANES: usize>
    BasePlan<T, ROWS, TABLE_LANES>
{
    /// Builds the immutable plan for the widest native layout this kernel
    /// implements. Scalar fallback is not a base-kernel capability.
    pub(crate) fn new_if_supported<const INVERSE: bool>() -> Option<Self> {
        let eight_lanes_supported = size_of::<T>() == 4 && native_lanes_supported::<8, T>();
        let four_lanes_supported = !eight_lanes_supported && native_lanes_supported::<4, T>();
        let lane_width =
            select_lane_width(size_of::<T>(), eight_lanes_supported, four_lanes_supported)?;
        Some(Self::new::<INVERSE>(lane_width))
    }

    /// Whether this plan selected the eight-lane register layout, so
    /// width-dispatched companions (the split gather) run at the same
    /// native width as the base kernel.
    pub(crate) const fn native_eight_lanes(&self) -> bool {
        matches!(self.lane_width, BaseLaneWidth::Eight)
    }

    fn new<const INVERSE: bool>(lane_width: BaseLaneWidth) -> Self {
        let dir = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        let w = |j: usize, n: usize| -> [f64; 2] {
            let (s, c) = (dir * core::f64::consts::TAU * j as f64 / n as f64).sin_cos();
            [c, s]
        };
        debug_assert_eq!(TABLE_LANES, table_lanes(ROWS));
        let n = 16 * ROWS;
        let mut table = Vec::with_capacity(TABLE_LANES);
        for a in 1..ROWS {
            let groups = match lane_width {
                BaseLaneWidth::Four => 8,
                BaseLaneWidth::Eight => 4,
            };
            let samples_per_group = match lane_width {
                BaseLaneWidth::Four => 2,
                BaseLaneWidth::Eight => 4,
            };
            for group in 0..groups {
                for component in 0..2 {
                    for sample in 0..samples_per_group {
                        let twiddle = w((a * (samples_per_group * group + sample)) % n, n);
                        table.extend([T::from_precise(twiddle[component]); 2]);
                    }
                }
            }
        }
        // Broadcast row twiddles: each dup-split chunk pair repeats one
        // scalar across both samples (chunks 124..131).
        let mut push_broadcast = |v: [f64; 2]| {
            for c in [v[0], v[1]] {
                table.extend([c; 4].map(T::from_precise));
            }
        };
        let row1 = w(1, 16);
        let row3 = w(3, 16);
        push_broadcast(row1);
        push_broadcast(row3);
        let neg1 = row1;
        push_broadcast([-neg1[0], -neg1[1]]);
        table.extend([core::f64::consts::FRAC_1_SQRT_2; 4].map(T::from_precise));

        let col = [w(1, 8), w(3, 8)].map(|v| [T::from_precise(v[0]), T::from_precise(v[1])]);
        let table: Box<[T; TABLE_LANES]> = table
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("the pushes above emit exactly TABLE_LANES lanes"));
        Self {
            table,
            lane_width,
            col,
        }
    }
}

/// Plan-owned directional state for a selected base route.
pub(crate) struct BasePlanState<T, const ROWS: usize, const TABLE_LANES: usize> {
    forward: BasePlan<T, ROWS, TABLE_LANES>,
    inverse: OnceLock<BasePlan<T, ROWS, TABLE_LANES>>,
}

impl<T: MixedRadixScalar, const ROWS: usize, const TABLE_LANES: usize>
    BasePlanState<T, ROWS, TABLE_LANES>
{
    /// Builds the forward plan when the exact-width route is available.
    ///
    /// Both row counts serve every scalar. The four-row route briefly carried
    /// a per-scalar switch, because on 2026-08-29 a four-byte scalar measured
    /// 252 ns against 126 ns without it at n = 64. Two things have changed
    /// since: this construction replaced the sample-major kernel that measured
    /// it, and hermes now enters its scalar fallback inside the AVX2+FMA frame
    /// (`HS-SCALAR-FALLBACK-FRAME`). Re-measured against both, the route is
    /// 76 ns against 126 ns — the reverse — so the switch is gone rather than
    /// flipped.
    pub(crate) fn new_if_supported() -> Option<Self> {
        BasePlan::new_if_supported::<false>().map(|forward| Self {
            forward,
            inverse: OnceLock::new(),
        })
    }

    /// Borrows the immutable forward plan.
    pub(crate) fn forward(&self) -> &BasePlan<T, ROWS, TABLE_LANES> {
        &self.forward
    }

    /// Borrows the immutable inverse plan, initializing it once across clones.
    pub(crate) fn inverse(&self) -> &BasePlan<T, ROWS, TABLE_LANES> {
        self.inverse
            .get_or_init(|| BasePlan::new::<true>(self.forward.lane_width))
    }

    #[cfg(test)]
    pub(crate) fn inverse_is_initialized(&self) -> bool {
        self.inverse.get().is_some()
    }
}

#[cfg(test)]
mod lane_width_tests {
    use super::{select_lane_width, BaseLaneWidth};

    #[test]
    fn selector_preserves_the_four_lane_eight_byte_route() {
        assert_eq!(select_lane_width(8, true, true), Some(BaseLaneWidth::Four));
        assert_eq!(select_lane_width(4, true, true), Some(BaseLaneWidth::Eight));
        assert_eq!(select_lane_width(4, false, true), Some(BaseLaneWidth::Four));
        assert_eq!(select_lane_width(4, false, false), None);
    }
}
