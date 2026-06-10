use super::helpers::{dwt_coefficients_from_leto, leto_array1_from_slice};
use super::{DwtLetoCoefficients, DwtPlan};
use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::spectrum::coefficients::DwtCoefficients;
use crate::infrastructure::kernel::discrete::synthesis_stage_into;

impl DwtPlan {
    /// Execute inverse multilevel DWT.
    pub fn inverse(&self, coefficients: &DwtCoefficients) -> WaveletResult<Vec<f64>> {
        if coefficients.len() != self.len() || coefficients.levels() != self.levels() {
            return Err(WaveletError::CoefficientShapeMismatch);
        }
        let mut current = coefficients.approximation().to_vec();
        for detail in coefficients.details().iter().rev() {
            let n = current.len() * 2;
            let mut output = vec![0.0; n];
            synthesis_stage_into(&current, detail, self.wavelet(), &mut output);
            current = output;
        }
        Ok(current)
    }

    /// Execute inverse multilevel DWT from Leto-backed coefficients.
    pub fn inverse_leto(
        &self,
        coefficients: &DwtLetoCoefficients<f64>,
    ) -> WaveletResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
        let coefficients = dwt_coefficients_from_leto(coefficients)?;
        let signal = self.inverse(&coefficients)?;
        leto_array1_from_slice(&signal)
    }
}
