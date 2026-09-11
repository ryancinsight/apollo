//! Immutable table construction and directional state for the base kernel.

use super::store::SplitSinks;
use super::table_lanes;
use crate::application::execution::kernel::components::aligned::CacheLineAligned;
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

/// The widest native layout the base kernel runs for `T` on this host, or
/// none where neither width is native.
pub(super) fn native_width<T: MixedRadixScalar>() -> Option<BaseLaneWidth> {
    let eight_lanes_supported = size_of::<T>() == 4 && native_lanes_supported::<8, T>();
    let four_lanes_supported = !eight_lanes_supported && native_lanes_supported::<4, T>();
    select_lane_width(size_of::<T>(), eight_lanes_supported, four_lanes_supported)
}

pub(crate) struct BasePlan<T, const ROWS: usize, const ROW_LEN: usize, const TABLE_LANES: usize> {
    /// Dup-split twiddles. The fixed-size type is load-bearing, not
    /// decoration: the checked view's `offset + LANE_COUNT <= len` assert
    /// folds only when the length is a compile-time constant, and this
    /// kernel reads the table from inside its hot loops. The sample-major
    /// kernel learned the same lesson first (gap_audit.md#base128-bounds);
    /// this module was shelved before that fix landed and never received it.
    /// The cache-line alignment is load-bearing likewise ([`CacheLineAligned`]).
    pub(super) table: Box<CacheLineAligned<[T; TABLE_LANES]>>,
    /// Register layout used to build `table` and execute this plan.
    pub(super) lane_width: BaseLaneWidth,
    /// The column network's twiddles as complex values for its splats:
    /// `W_8^{1,3}` (the distance-4 stage) then `W_16^{1,3,5,7}` (the
    /// sixteen-row form's distance-8 stage). The eight-row pass reads the
    /// first two, the four-row pass none — its radix-4 column transform
    /// needs no multiply beyond its rotation.
    pub(super) col: [[T; 2]; 6],
}

impl<T: MixedRadixScalar, const ROWS: usize, const ROW_LEN: usize, const TABLE_LANES: usize>
    BasePlan<T, ROWS, ROW_LEN, TABLE_LANES>
{
    /// Builds the immutable plan for the widest native layout this kernel
    /// implements. Scalar fallback is not a base-kernel capability.
    pub(crate) fn new_if_supported<const INVERSE: bool>() -> Option<Self> {
        Some(Self::new::<INVERSE>(native_width::<T>()?))
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
        debug_assert_eq!(TABLE_LANES, table_lanes(ROWS, ROW_LEN));
        let n = ROW_LEN * ROWS;
        let mut table = Vec::with_capacity(TABLE_LANES);
        for a in 1..ROWS {
            let groups = match lane_width {
                BaseLaneWidth::Four => ROW_LEN / 2,
                BaseLaneWidth::Eight => ROW_LEN / 4,
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
        if ROW_LEN == 32 {
            // Every general `W_32^{b0 m}` of the `8 x 4` layer, pre-rotated:
            // `W_32^{1,3,5,7}` and `W_16^{1,3}` for the first group, then
            // the odd multiples the later groups reach under a rotation or a
            // sign (`W_32^{9,15,21}`, `W_16^{5,7,9}`), so no operation
            // follows a layer multiply; then the eighths `W_8^{1,3}` for
            // the radix-8 and the layer alike.
            for (j, n) in [
                (1, 32),
                (3, 32),
                (5, 32),
                (7, 32),
                (1, 16),
                (3, 16),
                (9, 32),
                (15, 32),
                (21, 32),
                (5, 16),
                (7, 16),
                (9, 16),
                (1, 8),
                (3, 8),
            ] {
                push_broadcast(w(j, n));
            }
        } else {
            let row1 = w(1, 16);
            let row3 = w(3, 16);
            push_broadcast(row1);
            push_broadcast(row3);
            let neg1 = row1;
            push_broadcast([-neg1[0], -neg1[1]]);
            push_broadcast(w(1, 8));
            push_broadcast(w(3, 8));
        }

        let col = [w(1, 8), w(3, 8), w(1, 16), w(3, 16), w(5, 16), w(7, 16)]
            .map(|v| [T::from_precise(v[0]), T::from_precise(v[1])]);
        let table: [T; TABLE_LANES] = table
            .try_into()
            .unwrap_or_else(|_| unreachable!("the pushes above emit exactly TABLE_LANES lanes"));
        Self {
            table: Box::new(CacheLineAligned(table)),
            lane_width,
            col,
        }
    }
}

/// Plan-owned directional state for a selected base route of `n` samples:
/// the forward plan and, initialized on first use, the inverse; and the
/// split's sink twiddles ([`SplitSinks`]) for `n`, likewise per direction.
pub(crate) struct BasePlanState<
    T,
    const ROWS: usize,
    const ROW_LEN: usize,
    const TABLE_LANES: usize,
> {
    n: usize,
    forward: BasePlan<T, ROWS, ROW_LEN, TABLE_LANES>,
    inverse: OnceLock<BasePlan<T, ROWS, ROW_LEN, TABLE_LANES>>,
    sinks: SplitSinks<T>,
    inverse_sinks: OnceLock<SplitSinks<T>>,
}

impl<T, const ROWS: usize, const ROW_LEN: usize, const TABLE_LANES: usize>
    BasePlanState<T, ROWS, ROW_LEN, TABLE_LANES>
where
    T: MixedRadixScalar<Complex = eunomia::Complex<T>>,
{
    /// Builds the forward plan and sink tables for a route of `n` samples
    /// when the exact-width route is available.
    ///
    /// Both row counts serve every scalar. The four-row route briefly carried
    /// a per-scalar switch, because on 2026-08-29 a four-byte scalar measured
    /// 252 ns against 126 ns without it at n = 64. Two things have changed
    /// since: this construction replaced the sample-major kernel that measured
    /// it, and hermes now enters its scalar fallback inside the AVX2+FMA frame
    /// (`HS-SCALAR-FALLBACK-FRAME`). Re-measured against both, the route is
    /// 76 ns against 126 ns — the reverse — so the switch is gone rather than
    /// flipped.
    pub(crate) fn new_if_supported(n: usize) -> Option<Self> {
        let forward = BasePlan::new_if_supported::<false>()?;
        let sinks = Self::sinks_for::<false>(&forward, n);
        Some(Self {
            n,
            forward,
            inverse: OnceLock::new(),
            sinks,
            inverse_sinks: OnceLock::new(),
        })
    }

    /// The sink tables for `n` at the plan's register width: empty when
    /// the route is one block, computed for the radix-3 step over three
    /// blocks, and relaid from the stage-major table for the radix-4 and
    /// radix-8 steps — so a single-block plan never touches the twiddle
    /// cache, and a three-block one never asks it for a length it does not
    /// serve.
    fn sinks_for<const INVERSE: bool>(
        plan: &BasePlan<T, ROWS, ROW_LEN, TABLE_LANES>,
        n: usize,
    ) -> SplitSinks<T> {
        let base = ROWS * ROW_LEN;
        let samples = match plan.lane_width {
            BaseLaneWidth::Four => 2,
            BaseLaneWidth::Eight => 4,
        };
        if n == 3 * base {
            SplitSinks::build_radix3::<INVERSE>(samples, base)
        } else if n > base {
            let twiddles = if INVERSE {
                T::cached_twiddle_inv(n)
            } else {
                T::cached_twiddle_fwd(n)
            };
            if n == 8 * base {
                SplitSinks::build_radix8(samples, &twiddles, base)
            } else {
                SplitSinks::build(samples, &twiddles, base, n)
            }
        } else {
            SplitSinks::empty()
        }
    }

    /// Borrows the immutable forward plan.
    pub(crate) fn forward(&self) -> &BasePlan<T, ROWS, ROW_LEN, TABLE_LANES> {
        &self.forward
    }

    /// Borrows the immutable inverse plan, initializing it once across clones.
    pub(crate) fn inverse(&self) -> &BasePlan<T, ROWS, ROW_LEN, TABLE_LANES> {
        self.inverse
            .get_or_init(|| BasePlan::new::<true>(self.forward.lane_width))
    }

    /// The forward route's sink tables.
    pub(crate) fn sinks(&self) -> &SplitSinks<T> {
        &self.sinks
    }

    /// The inverse route's sink tables, initialized once across clones.
    pub(crate) fn inverse_sinks(&self) -> &SplitSinks<T> {
        self.inverse_sinks
            .get_or_init(|| Self::sinks_for::<true>(&self.forward, self.n))
    }

    #[cfg(test)]
    pub(crate) fn inverse_is_initialized(&self) -> bool {
        self.inverse.get().is_some()
    }

    #[cfg(test)]
    pub(crate) fn inverse_sinks_initialized(&self) -> bool {
        self.inverse_sinks.get().is_some()
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
