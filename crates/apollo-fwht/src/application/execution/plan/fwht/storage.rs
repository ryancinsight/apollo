//! Precision storage contract for FWHT execution.

use crate::application::execution::kernel::direct::wht_inplace;
use crate::application::execution::plan::fwht::dimension_1d::FwhtPlan;
use crate::domain::contracts::error::FwhtError;
use apollo_fft::{f16, PrecisionProfile};
use ndarray::Array1;

thread_local! {
    static TYPED_INPUT64_SCRATCH: mnemosyne::scratch::ScratchPool<f64> = const { mnemosyne::scratch::ScratchPool::new() };
    static TYPED_OUTPUT64_SCRATCH: mnemosyne::scratch::ScratchPool<f64> = const { mnemosyne::scratch::ScratchPool::new() };
    static TYPED_F32_SCRATCH: mnemosyne::scratch::ScratchPool<f32> = const { mnemosyne::scratch::ScratchPool::new() };
}

/// Real storage accepted by typed FWHT paths.
pub trait FwhtStorage: Copy + Send + Sync + 'static {
    /// Required precision profile for this storage type.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `f64` arithmetic path.
    fn to_f64(self) -> f64;

    /// Convert an owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute forward transform into caller-owned contiguous storage.
    fn forward_slice_into(
        plan: &FwhtPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        validate_lengths(plan, input.len(), output.len())?;
        with_f64_workspaces(plan.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(input.iter().copied()) {
                *slot = Self::to_f64(value);
            }
            plan.forward_f64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_f64(value);
            }
            Ok(())
        })
    }

    /// Execute forward transform into caller-owned ndarray storage.
    fn forward_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        Self::forward_slice_into(
            plan,
            input.as_slice().expect("Array must be contiguous"),
            output.as_slice_mut().expect("Array must be contiguous"),
            profile,
        )
    }

    /// Execute inverse transform into caller-owned contiguous storage.
    fn inverse_slice_into(
        plan: &FwhtPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        validate_lengths(plan, input.len(), output.len())?;
        with_f64_workspaces(plan.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(input.iter().copied()) {
                *slot = Self::to_f64(value);
            }
            plan.inverse_f64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_f64(value);
            }
            Ok(())
        })
    }

    /// Execute inverse transform into caller-owned ndarray storage.
    fn inverse_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        Self::inverse_slice_into(
            plan,
            input.as_slice().expect("Array must be contiguous"),
            output.as_slice_mut().expect("Array must be contiguous"),
            profile,
        )
    }
}

impl FwhtStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_slice_into(
        plan: &FwhtPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_f64_slice_into(input, output)
    }

    fn inverse_slice_into(
        plan: &FwhtPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_f64_slice_into(input, output)
    }
}

impl FwhtStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn forward_slice_into(
        plan: &FwhtPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        validate_lengths(plan, input.len(), output.len())?;
        output.copy_from_slice(input);
        wht_inplace(output);
        Ok(())
    }

    fn inverse_slice_into(
        plan: &FwhtPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        Self::forward_slice_into(plan, input, output, profile)?;
        let scale = 1.0_f32 / plan.len() as f32;
        for value in output.iter_mut() {
            *value *= scale;
        }
        Ok(())
    }
}

impl FwhtStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }

    fn forward_slice_into(
        plan: &FwhtPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        validate_lengths(plan, input.len(), output.len())?;
        with_f32_workspace(plan.len(), |compute| {
            for (slot, value) in compute.iter_mut().zip(input.iter()) {
                *slot = value.to_f32();
            }
            wht_inplace(compute);
            for (slot, value) in output.iter_mut().zip(compute.iter().copied()) {
                *slot = f16::from_f32(value);
            }
            Ok(())
        })
    }

    fn inverse_slice_into(
        plan: &FwhtPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        validate_lengths(plan, input.len(), output.len())?;
        with_f32_workspace(plan.len(), |compute| {
            for (slot, value) in compute.iter_mut().zip(input.iter()) {
                *slot = value.to_f32();
            }
            wht_inplace(compute);
            let scale = 1.0_f32 / plan.len() as f32;
            for (slot, value) in output.iter_mut().zip(compute.iter().copied()) {
                *slot = f16::from_f32(value * scale);
            }
            Ok(())
        })
    }
}

fn validate_lengths(plan: &FwhtPlan, input: usize, output: usize) -> Result<(), FwhtError> {
    if input == plan.len() && output == plan.len() {
        Ok(())
    } else {
        Err(FwhtError::LengthMismatch)
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> Result<(), FwhtError> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(FwhtError::PrecisionMismatch)
    }
}

fn with_f64_workspaces<R>(n: usize, f: impl FnOnce(&mut [f64], &mut [f64]) -> R) -> R {
    TYPED_INPUT64_SCRATCH.with(|in_pool| {
        in_pool.with_scratch(n, |input64| {
            TYPED_OUTPUT64_SCRATCH
                .with(|out_pool| out_pool.with_scratch(n, |output64| f(input64, output64)))
        })
    })
}

fn with_f32_workspace<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    TYPED_F32_SCRATCH.with(|pool| pool.with_scratch(n, f))
}

#[cfg(test)]
pub(crate) fn typed_scratch_capacities() -> (usize, usize, usize) {
    TYPED_INPUT64_SCRATCH.with(|input_scratch| {
        TYPED_OUTPUT64_SCRATCH.with(|output_scratch| {
            TYPED_F32_SCRATCH.with(|f32_scratch| {
                (
                    input_scratch.capacity(),
                    output_scratch.capacity(),
                    f32_scratch.capacity(),
                )
            })
        })
    })
}
