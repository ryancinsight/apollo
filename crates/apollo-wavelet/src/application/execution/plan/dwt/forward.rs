use super::helpers::{dwt_coefficients_to_leto, leto_view1_cow};
use super::{DwtLetoCoefficients, DwtPlan};
use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::spectrum::coefficients::DwtCoefficients;
use crate::infrastructure::kernel::discrete::analysis_stage_into;

impl DwtPlan {
    /// Execute a multilevel forward DWT.
    pub fn forward(&self, signal: &[f64]) -> WaveletResult<DwtCoefficients> {
        if signal.len() != self.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let mut current = signal.to_vec();
        let mut details = Vec::with_capacity(self.levels());
        for _ in 0..self.levels() {
            let half = current.len() / 2;
            let mut approximation = vec![0.0; half];
            let mut detail = vec![0.0; half];
            analysis_stage_into(&current, self.wavelet(), &mut approximation, &mut detail);
            details.push(detail);
            current = approximation;
        }
        Ok(DwtCoefficients::new(
            self.len(),
            self.levels(),
            current,
            details,
        ))
    }

    /// Execute a multilevel forward DWT from a Leto 1D signal view.
    ///
    /// Contiguous Leto views are borrowed directly; strided views are copied once
    /// into logical order before reusing the canonical slice DWT kernel.
    pub fn forward_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
    ) -> WaveletResult<DwtLetoCoefficients<f64>> {
        let signal = leto_view1_cow(signal)?;
        let coefficients = self.forward(signal.as_ref())?;
        dwt_coefficients_to_leto(&coefficients)
    }
}
