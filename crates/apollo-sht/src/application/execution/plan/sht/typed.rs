//! Typed storage traits and implementations for Spherical Harmonic Transforms.

use super::helpers::{
    validate_coefficient_array_shape, validate_profile, validate_sample_array_shape,
    write_complex_array,
};
use super::ShtPlan;
use crate::domain::contracts::error::ShtResult;
use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
use apollo_fft::{f16, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::Array2;

/// Real sample storage accepted by typed SHT paths.
pub trait ShtRealStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into owner `f64` arithmetic.
    fn to_f64(self) -> f64;

    /// Convert owner arithmetic back to storage.
    fn from_f64(value: f64) -> Self;

    /// View slice as `f32` if layout is identical.
    #[inline]
    fn as_f32_slice(slice: &[Self]) -> Option<&[f32]> {
        let _ = slice;
        None
    }

    /// View mutable slice as `f32` if layout is identical.
    #[inline]
    fn as_f32_slice_mut(slice: &mut [Self]) -> Option<&mut [f32]> {
        let _ = slice;
        None
    }

    /// Execute typed forward real SHT.
    fn forward_real_into<O: ShtComplexStorage>(
        plan: &ShtPlan,
        samples: &Array2<Self>,
        output: &mut Array2<O>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(sample_profile, Self::PROFILE)?;
        validate_profile(coefficient_profile, O::PROFILE)?;
        validate_sample_array_shape(plan, samples)?;
        validate_coefficient_array_shape(plan, output)?;
        let samples64 = samples.mapv(Self::to_f64);
        let coefficients = plan.forward_real(&samples64)?;
        write_complex_array(coefficients.values(), output);
        Ok(())
    }
}

impl ShtRealStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_real_into<O: ShtComplexStorage>(
        plan: &ShtPlan,
        samples: &Array2<Self>,
        output: &mut Array2<O>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(sample_profile, Self::PROFILE)?;
        validate_profile(coefficient_profile, O::PROFILE)?;
        validate_sample_array_shape(plan, samples)?;
        validate_coefficient_array_shape(plan, output)?;
        let coefficients = plan.forward_real(samples)?;
        write_complex_array(coefficients.values(), output);
        Ok(())
    }
}

impl ShtRealStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    #[inline]
    fn as_f32_slice(slice: &[Self]) -> Option<&[f32]> {
        Some(slice)
    }

    #[inline]
    fn as_f32_slice_mut(slice: &mut [Self]) -> Option<&mut [f32]> {
        Some(slice)
    }
}

impl ShtRealStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

/// Complex sample/coefficient storage accepted by typed SHT paths.
pub trait ShtComplexStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into owner `Complex64` arithmetic.
    fn to_complex64(self) -> Complex64;

    /// Convert owner arithmetic back to storage.
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

    /// Execute typed forward complex SHT.
    fn forward_complex_into<O: ShtComplexStorage>(
        plan: &ShtPlan,
        samples: &Array2<Self>,
        output: &mut Array2<O>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(sample_profile, Self::PROFILE)?;
        validate_profile(coefficient_profile, O::PROFILE)?;
        validate_sample_array_shape(plan, samples)?;
        validate_coefficient_array_shape(plan, output)?;
        let samples64 = samples.mapv(Self::to_complex64);
        let coefficients = plan.forward_complex(&samples64)?;
        write_complex_array(coefficients.values(), output);
        Ok(())
    }

    /// Execute typed inverse SHT into complex samples.
    fn inverse_complex_into<O: ShtComplexStorage>(
        plan: &ShtPlan,
        coefficients: &Array2<Self>,
        output: &mut Array2<O>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(coefficient_profile, Self::PROFILE)?;
        validate_profile(sample_profile, O::PROFILE)?;
        validate_coefficient_array_shape(plan, coefficients)?;
        validate_sample_array_shape(plan, output)?;
        let coefficients64 = coefficients.mapv(Self::to_complex64);
        let owner_coefficients =
            SphericalHarmonicCoefficients::from_values(plan.grid().max_degree(), coefficients64);
        let samples = plan.inverse_complex(&owner_coefficients)?;
        write_complex_array(&samples, output);
        Ok(())
    }

    /// Execute typed inverse SHT into real samples.
    fn inverse_real_into<O: ShtRealStorage>(
        plan: &ShtPlan,
        coefficients: &Array2<Self>,
        output: &mut Array2<O>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(coefficient_profile, Self::PROFILE)?;
        validate_profile(sample_profile, O::PROFILE)?;
        validate_coefficient_array_shape(plan, coefficients)?;
        validate_sample_array_shape(plan, output)?;
        let coefficients64 = coefficients.mapv(Self::to_complex64);
        let owner_coefficients =
            SphericalHarmonicCoefficients::from_values(plan.grid().max_degree(), coefficients64);
        let samples = plan.inverse_real(&owner_coefficients)?;
        for (slot, value) in output
            .as_slice_mut()
            .expect("contiguous output")
            .iter_mut()
            .zip(samples.iter().copied())
        {
            *slot = O::from_f64(value);
        }
        Ok(())
    }
}

impl ShtComplexStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn forward_complex_into<O: ShtComplexStorage>(
        plan: &ShtPlan,
        samples: &Array2<Self>,
        output: &mut Array2<O>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(sample_profile, Self::PROFILE)?;
        validate_profile(coefficient_profile, O::PROFILE)?;
        validate_sample_array_shape(plan, samples)?;
        validate_coefficient_array_shape(plan, output)?;
        let coefficients = plan.forward_complex(samples)?;
        write_complex_array(coefficients.values(), output);
        Ok(())
    }

    fn inverse_complex_into<O: ShtComplexStorage>(
        plan: &ShtPlan,
        coefficients: &Array2<Self>,
        output: &mut Array2<O>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(coefficient_profile, Self::PROFILE)?;
        validate_profile(sample_profile, O::PROFILE)?;
        validate_coefficient_array_shape(plan, coefficients)?;
        validate_sample_array_shape(plan, output)?;
        let owner_coefficients = SphericalHarmonicCoefficients::from_values(
            plan.grid().max_degree(),
            coefficients.clone(),
        );
        let samples = plan.inverse_complex(&owner_coefficients)?;
        write_complex_array(&samples, output);
        Ok(())
    }

    fn inverse_real_into<O: ShtRealStorage>(
        plan: &ShtPlan,
        coefficients: &Array2<Self>,
        output: &mut Array2<O>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(coefficient_profile, Self::PROFILE)?;
        validate_profile(sample_profile, O::PROFILE)?;
        validate_coefficient_array_shape(plan, coefficients)?;
        validate_sample_array_shape(plan, output)?;
        let owner_coefficients = SphericalHarmonicCoefficients::from_values(
            plan.grid().max_degree(),
            coefficients.clone(),
        );
        let samples = plan.inverse_real(&owner_coefficients)?;
        for (slot, value) in output
            .as_slice_mut()
            .expect("contiguous output")
            .iter_mut()
            .zip(samples.iter().copied())
        {
            *slot = O::from_f64(value);
        }
        Ok(())
    }
}

impl ShtComplexStorage for Complex32 {
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

impl ShtComplexStorage for [f16; 2] {
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
