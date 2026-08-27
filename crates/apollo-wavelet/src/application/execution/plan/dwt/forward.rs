use super::boundary::dwt_coefficients_to_leto;
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
        // One scratch buffer ping-pongs with `current` across the shrinking
        // levels instead of allocating a fresh approximation per level: the
        // analysis kernel fully overwrites its outputs, so stale contents
        // surviving `truncate` never reach the coefficients. Only `detail`,
        // which the result keeps, is allocated per level.
        let mut scratch = vec![0.0_f64; current.len() / 2];
        let mut details = Vec::with_capacity(self.levels());
        for _ in 0..self.levels() {
            let half = current.len() / 2;
            let mut detail = vec![0.0_f64; half];
            scratch.truncate(half);
            analysis_stage_into(&current, self.wavelet(), &mut scratch, &mut detail);
            details.push(detail);
            std::mem::swap(&mut current, &mut scratch);
        }
        // The surviving buffer carries ping-pong capacity; return the
        // approximation at its exact size, as the per-level allocation did.
        current.shrink_to_fit();
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
        if signal.shape()[0] == 0 {
            return Err(WaveletError::EmptySignal);
        }
        let signal = apollo_leto_interop::view_cow(&signal);
        let coefficients = self.forward(signal.as_ref())?;
        dwt_coefficients_to_leto(&coefficients)
    }
}
