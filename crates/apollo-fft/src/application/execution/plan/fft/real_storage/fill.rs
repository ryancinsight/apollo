//! Zero-copy real ↔ complex array fill helpers shared by every [`RealFftData`]
//! storage implementation, generic over the storage scalar and rank.
//!
//! [`RealFftData`]: super::RealFftData

use super::RealFftData;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use leto::{Array, Storage, StorageMut};

/// Fill a caller-owned spectrum array from real storage values.
///
/// The output is written in logical row-major order; the source is read in the
/// same order (correct for any input strides). Both arrays share their shape.
#[inline]
pub(super) fn fill_spectrum<T, S1, S2, const N: usize>(
    input: &Array<T, S1, N>,
    output: &mut Array<Complex<T::PlanScalar>, S2, N>,
) where
    T: RealFftData,
    T::PlanScalar: MixedRadixScalar<Complex = Complex<T::PlanScalar>>,
    S1: Storage<T>,
    S2: StorageMut<Complex<T::PlanScalar>>,
{
    debug_assert_eq!(input.shape(), output.shape(), "real-to-complex shape mismatch");
    let dst = output
        .as_slice_mut()
        .expect("invariant: spectrum output is C-contiguous");
    for (slot, &src) in dst.iter_mut().zip(input.iter()) {
        *slot = src.to_spectrum();
    }
}

/// Fill caller-owned real storage from the real parts of a spectrum array.
#[inline]
pub(super) fn fill_real<T, S1, S2, const N: usize>(
    input: &Array<Complex<T::PlanScalar>, S1, N>,
    output: &mut Array<T, S2, N>,
) where
    T: RealFftData,
    T::PlanScalar: MixedRadixScalar<Complex = Complex<T::PlanScalar>>,
    S1: Storage<Complex<T::PlanScalar>>,
    S2: StorageMut<T>,
{
    debug_assert_eq!(input.shape(), output.shape(), "complex-to-real shape mismatch");
    let dst = output
        .as_slice_mut()
        .expect("invariant: real output is C-contiguous");
    for (slot, &src) in dst.iter_mut().zip(input.iter()) {
        *slot = T::from_spectrum(src);
    }
}
