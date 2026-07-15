use super::{DwtLetoCoefficients, DwtPlan};
use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::spectrum::coefficients::DwtCoefficients;
use crate::CwtPlan;
use apollo_fft::PrecisionProfile;
use leto::Array2;

pub(crate) fn dwt_coefficients_to_leto(
    coefficients: &DwtCoefficients,
) -> WaveletResult<DwtLetoCoefficients<f64>> {
    let approximation = apollo_leto_interop::try_array1_from_slice(coefficients.approximation())
        .ok_or(WaveletError::CoefficientShapeMismatch)?;
    let details = coefficients
        .details()
        .iter()
        .map(|detail| {
            apollo_leto_interop::try_array1_from_slice(detail)
                .ok_or(WaveletError::CoefficientShapeMismatch)
        })
        .collect::<WaveletResult<Vec<_>>>()?;
    Ok(DwtLetoCoefficients::new(
        coefficients.len(),
        coefficients.levels(),
        approximation,
        details,
    ))
}

pub(crate) fn dwt_typed_coefficients_to_leto<T: Copy>(
    len: usize,
    levels: usize,
    approximation: &[T],
    details: &[Vec<T>],
) -> WaveletResult<DwtLetoCoefficients<T>> {
    let approximation = apollo_leto_interop::try_array1_from_slice(approximation)
        .ok_or(WaveletError::CoefficientShapeMismatch)?;
    let details = details
        .iter()
        .map(|detail| {
            apollo_leto_interop::try_array1_from_slice(detail)
                .ok_or(WaveletError::CoefficientShapeMismatch)
        })
        .collect::<WaveletResult<Vec<_>>>()?;
    Ok(DwtLetoCoefficients::new(
        len,
        levels,
        approximation,
        details,
    ))
}

pub(crate) fn dwt_coefficients_from_leto(
    coefficients: &DwtLetoCoefficients<f64>,
) -> WaveletResult<DwtCoefficients> {
    let approximation_view = coefficients.approximation().view();
    if approximation_view.shape()[0] == 0 {
        return Err(WaveletError::EmptySignal);
    }
    let approximation = apollo_leto_interop::view_cow(&approximation_view).into_owned();
    let details = coefficients
        .details()
        .iter()
        .map(|detail| {
            let detail_view = detail.view();
            if detail_view.shape()[0] == 0 {
                return Err(WaveletError::EmptySignal);
            }
            Ok(apollo_leto_interop::view_cow(&detail_view).into_owned())
        })
        .collect::<WaveletResult<Vec<_>>>()?;
    Ok(DwtCoefficients::new(
        coefficients.len(),
        coefficients.levels(),
        approximation,
        details,
    ))
}

pub(crate) fn validate_profile(
    actual: PrecisionProfile,
    expected: PrecisionProfile,
) -> WaveletResult<()> {
    if actual.matches_storage_and_compute(expected) {
        Ok(())
    } else {
        Err(WaveletError::PrecisionMismatch)
    }
}

pub(crate) fn validate_dwt_output_shapes<T>(
    plan: &DwtPlan,
    approximation_len: usize,
    details: &[Vec<T>],
) -> WaveletResult<()> {
    let expected_approximation_len = plan.len() >> plan.levels();
    if approximation_len != expected_approximation_len || details.len() != plan.levels() {
        return Err(WaveletError::CoefficientShapeMismatch);
    }
    if details
        .iter()
        .map(Vec::len)
        .zip(plan.coefficient_shapes())
        .any(|(actual, expected)| actual != expected)
    {
        return Err(WaveletError::CoefficientShapeMismatch);
    }
    Ok(())
}

pub(crate) fn validate_cwt_output_shape<T>(
    plan: &CwtPlan,
    output: &Array2<T>,
) -> WaveletResult<()> {
    if output.shape()[0] == plan.scales().len() && output.shape()[1] == plan.len() {
        Ok(())
    } else {
        Err(WaveletError::CoefficientShapeMismatch)
    }
}
