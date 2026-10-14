//! Typed complex storage views and conversions for SFT paths.

use super::{validate_support_len, validate_transform_len, SparseFftPlan};
use crate::domain::spectrum::sparse::SparseSpectrum;
use apollo_fft::{ApolloError, ApolloResult, CpuStorage, PrecisionProfile, F16};
use eunomia::{Complex32, Complex64};
use moirai::ParallelSliceMut;

/// Below this independent element count, serial conversion avoids scheduling overhead.
pub(crate) const STORAGE_PAR_LEN_THRESHOLD: usize = 16_384;

/// Complex storage accepted by typed SFT paths.
pub trait SparseComplexStorage: CpuStorage<Complex64> {
    /// Execute typed forward sparse transform.
    fn forward_into(
        plan: &SparseFftPlan,
        signal: &[Self],
        frequencies: &mut Vec<usize>,
        values: &mut Vec<Self>,
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        profile.ensure_matches(Self::PROFILE, || {
            ApolloError::validation(
                "precision_profile",
                format!("{profile:?}"),
                format!(
                    "storage {:?} with compute {:?}",
                    Self::PROFILE.storage,
                    Self::PROFILE.compute
                ),
            )
        })?;
        validate_transform_len(plan.len(), signal.len())?;
        let owner_values = owner_values_from_storage(signal);
        let spectrum = plan.forward(&owner_values)?;
        frequencies.clear();
        values.clear();
        frequencies.reserve(spectrum.frequencies.len());
        values.reserve(spectrum.values.len());
        frequencies.extend_from_slice(&spectrum.frequencies);
        extend_storage_from_owner_values(values, &spectrum.values);
        Ok(())
    }

    /// Execute typed inverse sparse transform.
    fn inverse_into(
        plan: &SparseFftPlan,
        frequencies: &[usize],
        values: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        profile.ensure_matches(Self::PROFILE, || {
            ApolloError::validation(
                "precision_profile",
                format!("{profile:?}"),
                format!(
                    "storage {:?} with compute {:?}",
                    Self::PROFILE.storage,
                    Self::PROFILE.compute
                ),
            )
        })?;
        validate_transform_len(plan.len(), output.len())?;
        let spectrum = sparse_spectrum_from_storage(plan.len(), frequencies, values)?;
        let owner_values = plan.inverse(&spectrum)?;
        write_storage_from_owner_values(output, &owner_values);
        Ok(())
    }
}

impl SparseComplexStorage for Complex64 {
    fn forward_into(
        plan: &SparseFftPlan,
        signal: &[Self],
        frequencies: &mut Vec<usize>,
        values: &mut Vec<Self>,
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        profile.ensure_matches(Self::PROFILE, || {
            ApolloError::validation(
                "precision_profile",
                format!("{profile:?}"),
                format!(
                    "storage {:?} with compute {:?}",
                    Self::PROFILE.storage,
                    Self::PROFILE.compute
                ),
            )
        })?;
        let spectrum = plan.forward(signal)?;
        frequencies.clear();
        values.clear();
        frequencies.extend_from_slice(&spectrum.frequencies);
        values.extend_from_slice(&spectrum.values);
        Ok(())
    }

    fn inverse_into(
        plan: &SparseFftPlan,
        frequencies: &[usize],
        values: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        profile.ensure_matches(Self::PROFILE, || {
            ApolloError::validation(
                "precision_profile",
                format!("{profile:?}"),
                format!(
                    "storage {:?} with compute {:?}",
                    Self::PROFILE.storage,
                    Self::PROFILE.compute
                ),
            )
        })?;
        validate_transform_len(plan.len(), output.len())?;
        let spectrum = sparse_spectrum_from_storage(plan.len(), frequencies, values)?;
        let signal = plan.inverse(&spectrum)?;
        output.copy_from_slice(&signal);
        Ok(())
    }
}

impl SparseComplexStorage for Complex32 {}

impl SparseComplexStorage for [F16; 2] {}

fn sparse_spectrum_from_storage<T: SparseComplexStorage>(
    len: usize,
    frequencies: &[usize],
    values: &[T],
) -> ApolloResult<SparseSpectrum> {
    validate_support_len(frequencies.len(), values.len())?;
    let mut spectrum = SparseSpectrum::new(len);
    for (&frequency, &value) in frequencies.iter().zip(values.iter()) {
        spectrum.insert(frequency, value.to_cpu())?;
    }
    Ok(spectrum)
}

pub(crate) fn try_mnemosyne_array1_from_slice<T: Copy>(
    values: &[T],
) -> ApolloResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    apollo_leto_interop::try_array1_from_slice(values).ok_or_else(|| {
        ApolloError::validation(
            "leto_array",
            "rejected length",
            "Mnemosyne-backed 1D array construction",
        )
    })
}

pub(crate) fn owner_values_from_storage<T: SparseComplexStorage>(values: &[T]) -> Vec<Complex64> {
    if values.len() >= STORAGE_PAR_LEN_THRESHOLD {
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(values.len(), |index| {
            values[index].to_cpu()
        })
    } else {
        values.iter().copied().map(CpuStorage::to_cpu).collect()
    }
}

pub(crate) fn extend_storage_from_owner_values<T: SparseComplexStorage>(
    output: &mut Vec<T>,
    values: &[Complex64],
) {
    if values.len() >= STORAGE_PAR_LEN_THRESHOLD {
        let converted =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(values.len(), |index| {
                T::from_cpu(values[index])
            });
        output.extend(converted);
    } else {
        output.extend(values.iter().copied().map(T::from_cpu));
    }
}

pub(crate) fn write_storage_from_owner_values<T: SparseComplexStorage>(
    output: &mut [T],
    values: &[Complex64],
) {
    debug_assert_eq!(output.len(), values.len());
    if output.len() >= STORAGE_PAR_LEN_THRESHOLD {
        output.par_mut().enumerate(|index, slot| {
            *slot = T::from_cpu(values[index]);
        });
    } else {
        for (slot, value) in output.iter_mut().zip(values.iter().copied()) {
            *slot = T::from_cpu(value);
        }
    }
}
