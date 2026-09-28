use super::{DwtLetoCoefficients, DwtPlan};
use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::spectrum::coefficients::{detail_level_bounds, DwtCoefficients};
use crate::CwtPlan;
use apollo_fft::PrecisionProfile;
use leto::Array2;

pub(crate) fn dwt_coefficients_to_leto(
    coefficients: &DwtCoefficients,
) -> WaveletResult<DwtLetoCoefficients<f64>> {
    let approximation = apollo_leto_interop::try_array1_from_slice(coefficients.approximation())
        .ok_or(WaveletError::CoefficientShapeMismatch)?;
    let details = coefficients
        .detail_levels()
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
    details: &[T],
) -> WaveletResult<DwtLetoCoefficients<T>> {
    let approximation = apollo_leto_interop::try_array1_from_slice(approximation)
        .ok_or(WaveletError::CoefficientShapeMismatch)?;
    let details = (0..levels)
        .map(|level| {
            let (start, end) = detail_level_bounds(len, level);
            apollo_leto_interop::try_array1_from_slice(&details[start..end])
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
    let mut details = Vec::new();
    for detail in coefficients.details() {
        let detail_view = detail.view();
        if detail_view.shape()[0] == 0 {
            return Err(WaveletError::EmptySignal);
        }
        details.extend_from_slice(&apollo_leto_interop::view_cow(&detail_view));
    }
    DwtCoefficients::new(
        coefficients.len(),
        coefficients.levels(),
        approximation,
        details,
    )
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

pub(crate) fn validate_dwt_output_shapes(
    plan: &DwtPlan,
    approximation_len: usize,
    details_len: usize,
) -> WaveletResult<()> {
    let expected_approximation_len = plan.len() >> plan.levels();
    let expected_details_len = plan.len() - expected_approximation_len;
    if approximation_len != expected_approximation_len || details_len != expected_details_len {
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
