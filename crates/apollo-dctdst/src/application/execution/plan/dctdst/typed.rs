use crate::domain::contracts::error::{DctDstError, DctDstResult};
use apollo_fft::{f16, PrecisionProfile};
use super::DctDstPlan;
use super::helpers::{leto_array1_from_slice, leto_view1_cow, validate_profile};

impl DctDstPlan {
    /// Execute the forward transform for `f64`, `f32`, or mixed `f16` storage.
    ///
    /// Lower storage profiles reuse the crate's authoritative `f64` transform
    /// and quantize once into the caller-owned output slice. This avoids
    /// precision-specific algorithm forks and preserves the DCT/DST theorem
    /// surface.
    pub fn forward_typed_into<T: RealTransformStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        T::forward_into(self, signal, output, profile)
    }

    /// Execute the forward transform over a typed Leto real-valued 1D view.
    pub fn forward_leto_typed<T: RealTransformStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> DctDstResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = leto_view1_cow(&signal);
        let mut output = vec![T::from_f64(0.0); self.len()];
        T::forward_into(self, &signal, &mut output, profile)?;
        Ok(leto_array1_from_slice(&output))
    }

    /// Execute the inverse transform for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: RealTransformStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        T::inverse_into(self, signal, output, profile)
    }

    /// Execute the inverse transform over a typed Leto real-valued 1D view.
    pub fn inverse_leto_typed<T: RealTransformStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> DctDstResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = leto_view1_cow(&signal);
        let mut output = vec![T::from_f64(0.0); self.len()];
        T::inverse_into(self, &signal, &mut output, profile)?;
        Ok(leto_array1_from_slice(&output))
    }
}

/// Real storage accepted by typed DCT/DST paths.
pub trait RealTransformStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage to owner arithmetic.
    fn to_f64(self) -> f64;
    /// Convert owner arithmetic to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let input64: Vec<f64> = signal.iter().map(|value| value.to_f64()).collect();
        let mut output64 = vec![0.0_f64; plan.len()];
        plan.forward_into(&input64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.into_iter()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let input64: Vec<f64> = signal.iter().map(|value| value.to_f64()).collect();
        let mut output64 = vec![0.0_f64; plan.len()];
        plan.inverse_into(&input64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.into_iter()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }
}

impl RealTransformStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(signal, output)
    }

    fn inverse_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(signal, output)
    }
}

impl RealTransformStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl RealTransformStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}
