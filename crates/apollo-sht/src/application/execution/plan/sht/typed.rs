//! Typed storage traits and implementations for Spherical Harmonic Transforms.

use super::quadrature::{
    validate_coefficient_array_shape, validate_profile, validate_sample_array_shape,
    write_complex_array,
};
use super::ShtPlan;
use crate::domain::contracts::error::ShtResult;
use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
use apollo_fft::{f16, CpuStorage, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::Array2;

/// Real sample storage accepted by typed SHT paths.
pub trait ShtRealStorage: CpuStorage {
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
        let samples64 = samples.mapv(CpuStorage::to_cpu);
        let coefficients = plan.forward_real(&samples64)?;
        write_complex_array(coefficients.values(), output);
        Ok(())
    }
}

impl ShtRealStorage for f64 {
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

impl ShtRealStorage for f32 {}

impl ShtRealStorage for f16 {}

/// Complex sample/coefficient storage accepted by typed SHT paths.
pub trait ShtComplexStorage: CpuStorage<Complex64> {
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
        let samples64 = samples.mapv(CpuStorage::to_cpu);
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
        let coefficients64 = coefficients.mapv(CpuStorage::to_cpu);
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
        let coefficients64 = coefficients.mapv(CpuStorage::to_cpu);
        let owner_coefficients =
            SphericalHarmonicCoefficients::from_values(plan.grid().max_degree(), coefficients64);
        let samples = plan.inverse_real(&owner_coefficients)?;
        for (slot, value) in output
            .as_slice_mut()
            .expect("contiguous output")
            .iter_mut()
            .zip(samples.iter().copied())
        {
            *slot = O::from_cpu(value);
        }
        Ok(())
    }
}

impl ShtComplexStorage for Complex64 {
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
            *slot = O::from_cpu(value);
        }
        Ok(())
    }
}

impl ShtComplexStorage for Complex32 {}

impl ShtComplexStorage for [f16; 2] {}
