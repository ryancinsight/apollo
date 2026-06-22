//! Precision storage contract for FrFT execution.

use crate::application::execution::plan::frft::dimension_1d::FrftPlan;
use crate::domain::contracts::error::FrftError;
use apollo_fft::{f16, PrecisionProfile};
use mnemosyne::scratch::ScratchPool;
use ndarray::Array1;
use num_complex::{Complex32, Complex64};

thread_local! {
    static TYPED_INPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    static TYPED_OUTPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Complex storage accepted by typed FrFT paths.
pub trait FrftStorage: Copy + Send + Sync + 'static {
    /// Required precision profile for this storage type.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `Complex64` arithmetic path.
    fn to_complex64(self) -> Complex64;

    /// Convert owner arithmetic result back to storage.
    fn from_complex64(value: Complex64) -> Self;

    /// View slice as `Complex32` if layout is identical.
    #[inline]
    fn as_c32_slice(slice: &[Self]) -> Option<&[Complex32]> {
        let _ = slice;
        None
    }

    /// View mutable slice as `Complex32` if layout is identical.
    #[inline]
    fn as_c32_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        let _ = slice;
        None
    }

    /// Execute forward transform into caller-owned contiguous storage.
    fn forward_slice_into(
        plan: &FrftPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        validate_lengths(plan, input.len(), output.len())?;
        with_complex64_workspaces(plan.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(input.iter().copied()) {
                *slot = Self::to_complex64(value);
            }
            plan.forward_complex64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_complex64(value);
            }
            Ok(())
        })
    }

    /// Execute forward transform into caller-owned ndarray storage.
    fn forward_into(
        plan: &FrftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        Self::forward_slice_into(
            plan,
            input.as_slice().expect("Array must be contiguous"),
            output.as_slice_mut().expect("Array must be contiguous"),
            profile,
        )
    }

    /// Execute inverse transform into caller-owned contiguous storage.
    fn inverse_slice_into(
        plan: &FrftPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        validate_lengths(plan, input.len(), output.len())?;
        with_complex64_workspaces(plan.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(input.iter().copied()) {
                *slot = Self::to_complex64(value);
            }
            plan.inverse_complex64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_complex64(value);
            }
            Ok(())
        })
    }

    /// Execute inverse transform into caller-owned ndarray storage.
    fn inverse_into(
        plan: &FrftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        Self::inverse_slice_into(
            plan,
            input.as_slice().expect("Array must be contiguous"),
            output.as_slice_mut().expect("Array must be contiguous"),
            profile,
        )
    }
}

impl FrftStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn forward_slice_into(
        plan: &FrftPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_complex64_slice_into(input, output)
    }

    fn forward_into(
        plan: &FrftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(input, output)
    }

    fn inverse_slice_into(
        plan: &FrftPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_complex64_slice_into(input, output)
    }

    fn inverse_into(
        plan: &FrftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(input, output)
    }
}

impl FrftStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }

    fn from_complex64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }

    #[inline]
    fn as_c32_slice(slice: &[Self]) -> Option<&[Complex32]> {
        Some(slice)
    }

    #[inline]
    fn as_c32_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        Some(slice)
    }
}

impl FrftStorage for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self[0].to_f32()), f64::from(self[1].to_f32()))
    }

    fn from_complex64(value: Complex64) -> Self {
        [
            f16::from_f32(value.re as f32),
            f16::from_f32(value.im as f32),
        ]
    }
}

fn validate_lengths(plan: &FrftPlan, input: usize, output: usize) -> Result<(), FrftError> {
    if input != plan.len() || output != plan.len() {
        Err(FrftError::LengthMismatch {
            input: input.max(output),
            plan: plan.len(),
        })
    } else {
        Ok(())
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> Result<(), FrftError> {
    if apollo_fft::application::utilities::leto_interop::profile_matches(actual, expected) {
        Ok(())
    } else {
        Err(FrftError::PrecisionMismatch)
    }
}

fn with_complex64_workspaces<R>(
    n: usize,
    f: impl FnOnce(&mut [Complex64], &mut [Complex64]) -> R,
) -> R {
    TYPED_INPUT64_SCRATCH.with(|in_pool| {
        in_pool.with_scratch(n, |input64| {
            TYPED_OUTPUT64_SCRATCH
                .with(|out_pool| out_pool.with_scratch(n, |output64| f(input64, output64)))
        })
    })
}

#[cfg(test)]
pub(crate) fn typed_scratch_capacities() -> (usize, usize) {
    TYPED_INPUT64_SCRATCH.with(|in_pool| {
        TYPED_OUTPUT64_SCRATCH.with(|out_pool| (in_pool.capacity(), out_pool.capacity()))
    })
}
