use super::helpers::{
    dwt_typed_coefficients_to_leto, leto_array1_from_slice, leto_view1_cow,
    validate_cwt_output_shape, validate_dwt_output_shapes, validate_profile,
};
use super::{DwtLetoCoefficients, DwtPlan};
use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::spectrum::coefficients::DwtCoefficients;
use crate::CwtPlan;
use apollo_fft::{f16, PrecisionProfile};
use leto::Array2;

impl DwtPlan {
    /// Execute a multilevel forward DWT for `f64`, `f32`, or mixed `f16` storage.
    ///
    /// The owner kernel remains the `f64` orthogonal filter bank. Typed paths
    /// convert represented input into owner arithmetic and quantize once when
    /// writing caller-owned approximation/detail buffers.
    pub fn forward_typed_into<T: WaveletStorage>(
        &self,
        signal: &[T],
        approximation: &mut [T],
        details: &mut [Vec<T>],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        T::forward_dwt_into(self, signal, approximation, details, profile)
    }

    /// Execute a multilevel forward DWT from typed Leto signal storage.
    pub fn forward_leto_typed<T: WaveletStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> WaveletResult<DwtLetoCoefficients<T>> {
        validate_profile(profile, T::PROFILE)?;
        let signal = leto_view1_cow(signal)?;
        if signal.len() != self.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let mut approximation = vec![T::from_f64(0.0); self.len() >> self.levels()];
        let mut details = self
            .coefficient_shapes()
            .map(|len| vec![T::from_f64(0.0); len])
            .collect::<Vec<_>>();
        self.forward_typed_into(signal.as_ref(), &mut approximation, &mut details, profile)?;
        dwt_typed_coefficients_to_leto(self.len(), self.levels(), &approximation, &details)
    }

    /// Execute inverse multilevel DWT for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: WaveletStorage>(
        &self,
        approximation: &[T],
        details: &[Vec<T>],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        T::inverse_dwt_into(self, approximation, details, output, profile)
    }

    /// Execute inverse multilevel DWT from typed Leto-backed coefficients.
    pub fn inverse_leto_typed<T: WaveletStorage>(
        &self,
        coefficients: &DwtLetoCoefficients<T>,
        profile: PrecisionProfile,
    ) -> WaveletResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        validate_profile(profile, T::PROFILE)?;
        if coefficients.len() != self.len() || coefficients.levels() != self.levels() {
            return Err(WaveletError::CoefficientShapeMismatch);
        }
        let approximation = leto_view1_cow(coefficients.approximation().view())?;
        let details = coefficients
            .details()
            .iter()
            .map(|detail| Ok(leto_view1_cow(detail.view())?.into_owned()))
            .collect::<WaveletResult<Vec<_>>>()?;
        let mut output = vec![T::from_f64(0.0); self.len()];
        self.inverse_typed_into(approximation.as_ref(), &details, &mut output, profile)?;
        leto_array1_from_slice(&output)
    }
}

/// Real storage accepted by typed wavelet paths.
pub trait WaveletStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage value to owner `f64` arithmetic.
    fn to_f64(self) -> f64;

    /// Convert owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute typed forward DWT into caller-owned buffers.
    fn forward_dwt_into(
        plan: &DwtPlan,
        signal: &[Self],
        approximation: &mut [Self],
        details: &mut [Vec<Self>],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_dwt_output_shapes(plan, approximation.len(), details)?;
        if signal.len() != plan.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let signal64: Vec<f64> = signal.iter().copied().map(Self::to_f64).collect();
        let coefficients = plan.forward(&signal64)?;
        for (slot, value) in approximation
            .iter_mut()
            .zip(coefficients.approximation().iter().copied())
        {
            *slot = Self::from_f64(value);
        }
        for (detail_out, detail_in) in details.iter_mut().zip(coefficients.details()) {
            for (slot, value) in detail_out.iter_mut().zip(detail_in.iter().copied()) {
                *slot = Self::from_f64(value);
            }
        }
        Ok(())
    }

    /// Execute typed inverse DWT into a caller-owned signal buffer.
    fn inverse_dwt_into(
        plan: &DwtPlan,
        approximation: &[Self],
        details: &[Vec<Self>],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_dwt_output_shapes(plan, approximation.len(), details)?;
        if output.len() != plan.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let approximation64: Vec<f64> = approximation.iter().copied().map(Self::to_f64).collect();
        let details64: Vec<Vec<f64>> = details
            .iter()
            .map(|detail| detail.iter().copied().map(Self::to_f64).collect())
            .collect();
        let coefficients =
            DwtCoefficients::new(plan.len(), plan.levels(), approximation64, details64);
        let signal = plan.inverse(&coefficients)?;
        for (slot, value) in output.iter_mut().zip(signal.into_iter()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }

    /// Execute typed CWT into caller-owned storage.
    fn transform_cwt_into(
        plan: &CwtPlan,
        signal: &[Self],
        output: &mut Array2<Self>,
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_cwt_output_shape(plan, output)?;
        if signal.len() != plan.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let signal64: Vec<f64> = signal.iter().copied().map(Self::to_f64).collect();
        let coefficients = plan.transform(&signal64)?;
        for (slot, value) in output
            .as_slice_mut()
            .expect("contiguous output")
            .iter_mut()
            .zip(coefficients.values().iter().copied())
        {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }
}

impl WaveletStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_dwt_into(
        plan: &DwtPlan,
        signal: &[Self],
        approximation: &mut [Self],
        details: &mut [Vec<Self>],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_dwt_output_shapes(plan, approximation.len(), details)?;
        if signal.len() != plan.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let coefficients = plan.forward(signal)?;
        approximation.copy_from_slice(coefficients.approximation());
        for (detail_out, detail_in) in details.iter_mut().zip(coefficients.details()) {
            detail_out.copy_from_slice(detail_in);
        }
        Ok(())
    }

    fn inverse_dwt_into(
        plan: &DwtPlan,
        approximation: &[Self],
        details: &[Vec<Self>],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_dwt_output_shapes(plan, approximation.len(), details)?;
        if output.len() != plan.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let coefficients = DwtCoefficients::new(
            plan.len(),
            plan.levels(),
            approximation.to_vec(),
            details.to_vec(),
        );
        let signal = plan.inverse(&coefficients)?;
        output.copy_from_slice(&signal);
        Ok(())
    }

    fn transform_cwt_into(
        plan: &CwtPlan,
        signal: &[Self],
        output: &mut Array2<Self>,
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_cwt_output_shape(plan, output)?;
        if signal.len() != plan.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let coefficients = plan.transform(signal)?;
        output.assign(&coefficients.values().view());
        Ok(())
    }
}

impl WaveletStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl WaveletStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}
