//! Zero-copy real ↔ complex array fill helpers shared by every [`RealFftData`]
//! storage implementation, generic over the storage scalar and dimension.
//!
//! [`RealFftData`]: super::RealFftData

use super::RealFftData;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use ndarray::{Array, Dimension, Zip};
use eunomia::Complex;

/// Fill a caller-owned spectrum array from real storage values.
#[inline]
pub(super) fn fill_spectrum<T, D>(
    input: &Array<T, D>,
    output: &mut Array<Complex<T::PlanScalar>, D>,
) where
    T: RealFftData,
    T::PlanScalar: MixedRadixScalar<Complex = Complex<T::PlanScalar>>,
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "real-to-complex shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = src.to_spectrum());
}

/// Fill caller-owned real storage from the real parts of a spectrum array.
#[inline]
pub(super) fn fill_real<T, D>(input: &Array<Complex<T::PlanScalar>, D>, output: &mut Array<T, D>)
where
    T: RealFftData,
    T::PlanScalar: MixedRadixScalar<Complex = Complex<T::PlanScalar>>,
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "complex-to-real shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = T::from_spectrum(src));
}
