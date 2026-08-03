//! Typed storage implementations for Discrete Hartley Transform.

use super::plan::DhtPlan;
use crate::domain::contracts::error::{DhtError, DhtResult};
use apollo_fft::{f16, CpuElement, CpuStorage, PrecisionProfile};

/// Real storage accepted by typed DHT paths.
///
/// The conversion ladder and precision profile come from the shared
/// [`CpuStorage`] vocabulary; this trait adds only the plan-coupled
/// dispatch.
pub trait HartleyStorage: CpuStorage {
    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &DhtPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DhtError::LengthMismatch);
        }
        let n = plan.len();
        f64::with_scratch(n, n, |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(signal.iter()) {
                *slot = value.to_cpu();
            }
            plan.forward_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter()) {
                *slot = Self::from_cpu(*value);
            }
            Ok(())
        })
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &DhtPlan,
        spectrum: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if spectrum.len() != plan.len() || output.len() != plan.len() {
            return Err(DhtError::LengthMismatch);
        }
        let n = plan.len();
        f64::with_scratch(n, n, |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(spectrum.iter()) {
                *slot = value.to_cpu();
            }
            plan.inverse_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter()) {
                *slot = Self::from_cpu(*value);
            }
            Ok(())
        })
    }
}

impl HartleyStorage for f64 {
    fn forward_into(
        plan: &DhtPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(signal, output)
    }

    fn inverse_into(
        plan: &DhtPlan,
        spectrum: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(spectrum, output)
    }
}

impl HartleyStorage for f32 {}

impl HartleyStorage for f16 {}
#[inline]
fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> DhtResult<()> {
    if actual.matches_storage_and_compute(expected) {
        Ok(())
    } else {
        Err(DhtError::PrecisionMismatch)
    }
}
