//! Operation-owned tables for one planar four-step length.

use super::fold::FourStepFold;
use super::lane_order::LaneOrder;
use super::plan::BatchedPlan;
use super::plane::{planar_applies, plane_geometry};
use super::BatchedPlanCache;
use std::sync::{Arc, OnceLock};

struct DirectionPlans<T> {
    first: Arc<BatchedPlan<T>>,
    second: Arc<BatchedPlan<T>>,
}

/// Lazily initialized tables retained by a reusable four-step operation.
pub(crate) struct PlanarState<T> {
    len: usize,
    directions: [OnceLock<DirectionPlans<T>>; 2],
    fold: OnceLock<Arc<FourStepFold<T>>>,
}

impl<T: BatchedPlanCache> PlanarState<T> {
    /// Creates empty ownership slots for an admitted planar length.
    pub(crate) fn new(len: usize) -> Self {
        assert!(planar_applies(len), "requires a planar power of two");
        Self {
            len,
            directions: [OnceLock::new(), OnceLock::new()],
            fold: OnceLock::new(),
        }
    }

    /// Executes one direction while retaining every initialized table.
    pub(crate) fn execute<const INVERSE: bool>(
        &self,
        data: &mut [eunomia::Complex<T>],
        scratch: &mut [eunomia::Complex<T>],
    ) {
        super::driver::execute_planar::<T, INVERSE>(self, data, scratch);
    }

    pub(super) fn transform_len(&self) -> usize {
        self.len
    }

    pub(super) fn plans<const INVERSE: bool>(&self) -> (&BatchedPlan<T>, &BatchedPlan<T>) {
        let plans = self.directions[usize::from(INVERSE)].get_or_init(|| {
            let (first_len, second_len) = plane_geometry(self.len);
            let first = T::cached_plan::<INVERSE>(first_len);
            let second = if first_len == second_len {
                Arc::clone(&first)
            } else {
                T::cached_plan::<INVERSE>(second_len)
            };
            DirectionPlans { first, second }
        });
        (&plans.first, &plans.second)
    }

    pub(super) fn fold(&self) -> &FourStepFold<T> {
        self.fold.get_or_init(|| {
            let (rows, cols) = plane_geometry(self.len);
            T::cached_four_step_fold(self.len, cols, rows)
        })
    }
}
