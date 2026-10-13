//! Strong ownership boundary for reusable one-dimensional plans.

use super::FftPlan1D;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::domain::metadata::shape::Shape1D;
use std::sync::Arc;

/// Acquires the bounded strong owner for a reusable one-dimensional plan.
///
/// The plan layer owns this contract so execution can borrow prepared state
/// without depending on the orchestration cache that implements the policy.
pub trait PlanOwner: Sized {
    /// Returns the strong owner for `shape` from the bounded plan policy.
    fn acquire_plan(shape: Shape1D) -> Arc<FftPlan1D<Self>>
    where
        Self: MixedRadixScalar;
}
