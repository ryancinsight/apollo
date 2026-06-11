//! Typed storage implementations for Discrete Hartley Transform.

use super::helpers::{validate_profile, TYPED_INPUT64_SCRATCH, TYPED_OUTPUT64_SCRATCH};
use super::plan::DhtPlan;
use crate::domain::contracts::error::{DhtError, DhtResult};
use apollo_fft::{f16, PrecisionProfile};

/// Real storage accepted by typed DHT paths.
pub trait HartleyStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage value to the owner `f64` arithmetic path.
    fn to_f64(self) -> f64;
    /// Convert owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

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
        TYPED_INPUT64_SCRATCH.with(|in_pool| {
            in_pool.with_scratch(n, |input64| {
                TYPED_OUTPUT64_SCRATCH.with(|out_pool| {
                    out_pool.with_scratch(n, |output64| {
                        for (slot, value) in input64.iter_mut().zip(signal.iter()) {
                            *slot = value.to_f64();
                        }
                        plan.forward_into(input64, output64)?;
                        for (slot, value) in output.iter_mut().zip(output64.iter()) {
                            *slot = Self::from_f64(*value);
                        }
                        Ok(())
                    })
                })
            })
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
        TYPED_INPUT64_SCRATCH.with(|in_pool| {
            in_pool.with_scratch(n, |input64| {
                TYPED_OUTPUT64_SCRATCH.with(|out_pool| {
                    out_pool.with_scratch(n, |output64| {
                        for (slot, value) in input64.iter_mut().zip(spectrum.iter()) {
                            *slot = value.to_f64();
                        }
                        plan.inverse_into(input64, output64)?;
                        for (slot, value) in output.iter_mut().zip(output64.iter()) {
                            *slot = Self::from_f64(*value);
                        }
                        Ok(())
                    })
                })
            })
        })
    }
}

impl HartleyStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

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

impl HartleyStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl HartleyStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}
