use super::{DwtLetoCoefficients, DwtPlan};
use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::spectrum::coefficients::DwtCoefficients;
use crate::CwtPlan;
use apollo_fft::PrecisionProfile;
use ndarray::Array2;
use std::borrow::Cow;

pub(crate) fn leto_view1_cow<T: Copy>(
    view: leto::ArrayView1<'_, T>,
) -> WaveletResult<Cow<'_, [T]>> {
    if view.shape()[0] == 0 {
        return Err(WaveletError::EmptySignal);
    }
    Ok(apollo_fft::application::utilities::leto_interop::view1_cow(
        &view,
    ))
}

pub(crate) fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WaveletResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    apollo_fft::application::utilities::leto_interop::try_array1_from_slice(values)
        .ok_or(WaveletError::CoefficientShapeMismatch)
}

pub(crate) fn dwt_coefficients_to_leto(
    coefficients: &DwtCoefficients,
) -> WaveletResult<DwtLetoCoefficients<f64>> {
    let approximation = leto_array1_from_slice(coefficients.approximation())?;
    let details = coefficients
        .details()
        .iter()
        .map(|detail| leto_array1_from_slice(detail))
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
    let approximation = leto_array1_from_slice(approximation)?;
    let details = details
        .iter()
        .map(|detail| leto_array1_from_slice(detail))
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
    let approximation = leto_view1_cow(coefficients.approximation().view())?.into_owned();
    let details = coefficients
        .details()
        .iter()
        .map(|detail| Ok(leto_view1_cow(detail.view())?.into_owned()))
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
    if apollo_fft::application::utilities::leto_interop::profile_matches(actual, expected) {
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
    if output.nrows() == plan.scales().len() && output.ncols() == plan.len() {
        Ok(())
    } else {
        Err(WaveletError::CoefficientShapeMismatch)
    }
}
