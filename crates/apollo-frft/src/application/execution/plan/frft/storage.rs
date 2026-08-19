//! Precision storage contract for FrFT execution.

use crate::application::execution::plan::frft::dimension_1d::FrftPlan;
use crate::domain::contracts::error::FrftError;
use apollo_fft::{f16, CpuStorage, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::Array1;
use mnemosyne::scratch::ScratchPool;

thread_local! {
    static TYPED_INPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    static TYPED_OUTPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Complex storage accepted by typed FrFT paths.
pub trait FrftStorage: CpuStorage<Complex64> {
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
                *slot = value.to_cpu();
            }
            plan.forward_complex64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_cpu(value);
            }
            Ok(())
        })
    }

    /// Execute forward transform into caller-owned Leto storage.
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
                *slot = value.to_cpu();
            }
            plan.inverse_complex64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_cpu(value);
            }
            Ok(())
        })
    }

    /// Execute inverse transform into caller-owned Leto storage.
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

impl FrftStorage for Complex32 {}

impl FrftStorage for [f16; 2] {}

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
    if actual.matches_storage_and_compute(expected) {
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
