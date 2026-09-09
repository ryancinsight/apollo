//! Concrete-scalar bridge to the shared borrowed planar transform.

use crate::application::execution::kernel::components::batched;
use eunomia::Complex;

pub(super) fn try_four_step<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) -> bool
where
    T: batched::BatchedPlanCache<Complex = Complex<T>>,
{
    let n = data.len();
    if batched::planar_applies(n) {
        batched::four_step_batched::<T, INVERSE>(data, scratch);
        return true;
    }
    false
}
