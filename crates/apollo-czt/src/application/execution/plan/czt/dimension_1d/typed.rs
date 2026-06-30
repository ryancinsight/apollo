//! Typed storage implementations for 1D Chirp Z-Transform.

use super::helpers::{validate_profile, with_complex64_workspaces};
use super::plan::CztPlan;
use crate::domain::contracts::error::CztError;
use apollo_fft::{f16, PrecisionProfile};
use leto::Array1;
use eunomia::{Complex32, Complex64};

/// Complex storage accepted by typed CZT paths.
pub trait CztStorage: Copy + Send + Sync + 'static {
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

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &CztPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        Self::forward_slice_into(
            plan,
            input
                .as_slice()
                .expect("CZT typed input must be contiguous"),
            output
                .as_slice_mut()
                .expect("CZT typed output must be contiguous"),
            profile,
        )
    }

    /// Execute forward transform into caller-owned contiguous storage.
    fn forward_slice_into(
        plan: &CztPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.input_len() || output.len() != plan.output_len() {
            return Err(CztError::LengthMismatch);
        }
        with_complex64_workspaces(plan.input_len(), plan.output_len(), |input64, output64| {
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

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &CztPlan,
        spectrum: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        Self::inverse_slice_into(
            plan,
            spectrum
                .as_slice()
                .expect("CZT typed spectrum must be contiguous"),
            output
                .as_slice_mut()
                .expect("CZT typed inverse output must be contiguous"),
            profile,
        )
    }

    /// Execute inverse transform into caller-owned contiguous storage.
    fn inverse_slice_into(
        plan: &CztPlan,
        spectrum: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        if spectrum.len() != plan.output_len() || output.len() != plan.input_len() {
            return Err(CztError::LengthMismatch);
        }
        with_complex64_workspaces(
            plan.output_len(),
            plan.input_len(),
            |spectrum64, output64| {
                for (slot, value) in spectrum64.iter_mut().zip(spectrum.iter().copied()) {
                    *slot = Self::to_complex64(value);
                }
                plan.inverse_complex64_slice_into(spectrum64, output64)?;
                for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                    *slot = Self::from_complex64(value);
                }
                Ok(())
            },
        )
    }
}

impl CztStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn forward_into(
        plan: &CztPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(input, output)
    }

    fn forward_slice_into(
        plan: &CztPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_complex64_slice_into(input, output)
    }

    fn inverse_into(
        plan: &CztPlan,
        spectrum: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        if spectrum.size() != plan.output_len() || output.size() != plan.input_len() {
            return Err(CztError::LengthMismatch);
        }
        plan.inverse_complex64_slice_into(
            spectrum
                .as_slice()
                .expect("CZT spectrum must be contiguous"),
            output
                .as_slice_mut()
                .expect("CZT inverse output must be contiguous"),
        )
    }

    fn inverse_slice_into(
        plan: &CztPlan,
        spectrum: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_complex64_slice_into(spectrum, output)
    }
}

impl CztStorage for Complex32 {
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

impl CztStorage for [f16; 2] {
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
