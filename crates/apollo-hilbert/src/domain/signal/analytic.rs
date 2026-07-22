//! Analytic signal storage and derived envelope/phase observables.

use crate::domain::contracts::error::{HilbertError, HilbertResult};
use eunomia::Complex64;

/// Dense analytic signal `z[n] = x[n] + i H{x}[n]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalyticSignal {
    values: Vec<Complex64>,
}

impl AnalyticSignal {
    /// Create analytic-signal storage from computed values.
    #[must_use]
    pub fn new(values: Vec<Complex64>) -> Self {
        Self { values }
    }

    /// Return sample count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return true when no samples are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Borrow complex analytic samples.
    #[must_use]
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Return the real original signal component.
    #[must_use]
    pub fn real(&self) -> Vec<f64> {
        let mut output = vec![0.0; self.len()];
        real_values_into(&self.values, &mut output)
            .expect("output length initialized from analytic signal length");
        output
    }

    /// Write the real original signal component into caller-owned storage.
    pub fn real_into(&self, output: &mut [f64]) -> HilbertResult<()> {
        real_values_into(&self.values, output)
    }

    /// Return the Hilbert quadrature component.
    #[must_use]
    pub fn quadrature(&self) -> Vec<f64> {
        let mut output = vec![0.0; self.len()];
        quadrature_values_into(&self.values, &mut output)
            .expect("output length initialized from analytic signal length");
        output
    }

    /// Write the Hilbert quadrature component into caller-owned storage.
    pub fn quadrature_into(&self, output: &mut [f64]) -> HilbertResult<()> {
        quadrature_values_into(&self.values, output)
    }

    /// Return the instantaneous envelope `|z[n]|`.
    #[must_use]
    pub fn envelope(&self) -> Vec<f64> {
        let mut output = vec![0.0; self.len()];
        envelope_values_into(&self.values, &mut output)
            .expect("output length initialized from analytic signal length");
        output
    }

    /// Write the instantaneous envelope `|z[n]|` into caller-owned storage.
    pub fn envelope_into(&self, output: &mut [f64]) -> HilbertResult<()> {
        envelope_values_into(&self.values, output)
    }

    /// Return the wrapped instantaneous phase `atan2(imag, real)`.
    #[must_use]
    pub fn phase(&self) -> Vec<f64> {
        let mut output = vec![0.0; self.len()];
        phase_values_into(&self.values, &mut output)
            .expect("output length initialized from analytic signal length");
        output
    }

    /// Write the wrapped instantaneous phase `atan2(imag, real)` into caller-owned storage.
    pub fn phase_into(&self, output: &mut [f64]) -> HilbertResult<()> {
        phase_values_into(&self.values, output)
    }

    /// Return the instantaneous frequency in cycles per sample.
    ///
    /// Uses the complex-derivative formula
    /// `f[n] = arg(conj(z[n]) · z[n+1]) / (2π)`,
    /// which avoids explicit phase unwrapping and remains well-defined
    /// whenever `|z[n]| > 0`. Returns a vector of length `len() - 1`.
    /// The value `f[n]` is in `(−0.5, +0.5]` cycles per sample.
    #[must_use]
    pub fn instantaneous_frequency(&self) -> Vec<f64> {
        let mut output = vec![0.0; self.len().saturating_sub(1)];
        self.instantaneous_frequency_into(&mut output)
            .expect("output length initialized from analytic signal length");
        output
    }

    /// Write the instantaneous frequency in cycles per sample into caller-owned storage.
    pub fn instantaneous_frequency_into(&self, output: &mut [f64]) -> HilbertResult<()> {
        instantaneous_frequency_values_into(&self.values, output)
    }

    /// Consume storage and return complex samples.
    #[must_use]
    pub fn into_values(self) -> Vec<Complex64> {
        self.values
    }
}

pub(crate) fn real_values_into(values: &[Complex64], output: &mut [f64]) -> HilbertResult<()> {
    if output.len() != values.len() {
        return Err(HilbertError::LengthMismatch);
    }
    for (slot, value) in output.iter_mut().zip(values.iter()) {
        *slot = value.re;
    }
    Ok(())
}

pub(crate) fn quadrature_values_into(
    values: &[Complex64],
    output: &mut [f64],
) -> HilbertResult<()> {
    if output.len() != values.len() {
        return Err(HilbertError::LengthMismatch);
    }
    for (slot, value) in output.iter_mut().zip(values.iter()) {
        *slot = value.im;
    }
    Ok(())
}

pub(crate) fn envelope_values_into(values: &[Complex64], output: &mut [f64]) -> HilbertResult<()> {
    if output.len() != values.len() {
        return Err(HilbertError::LengthMismatch);
    }
    for (slot, value) in output.iter_mut().zip(values.iter()) {
        *slot = value.norm();
    }
    Ok(())
}

pub(crate) fn phase_values_into(values: &[Complex64], output: &mut [f64]) -> HilbertResult<()> {
    if output.len() != values.len() {
        return Err(HilbertError::LengthMismatch);
    }
    for (slot, value) in output.iter_mut().zip(values.iter()) {
        *slot = value.arg();
    }
    Ok(())
}

pub(crate) fn instantaneous_frequency_values_into(
    values: &[Complex64],
    output: &mut [f64],
) -> HilbertResult<()> {
    use std::f64::consts::TAU;
    if output.len() != values.len().saturating_sub(1) {
        return Err(HilbertError::LengthMismatch);
    }
    for (slot, window) in output.iter_mut().zip(values.windows(2)) {
        *slot = (window[0].conj() * window[1]).arg() / TAU;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_abs_diff_eq;

    fn analytic_fixture() -> AnalyticSignal {
        AnalyticSignal::new(vec![
            Complex64::new(3.0, 4.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(-5.0, 12.0),
        ])
    }

    #[test]
    fn caller_owned_observables_match_allocating_observables() {
        let analytic = analytic_fixture();
        let mut real = vec![0.0; analytic.len()];
        let mut quadrature = vec![0.0; analytic.len()];
        let mut envelope = vec![0.0; analytic.len()];
        let mut phase = vec![0.0; analytic.len()];
        let mut frequency = vec![0.0; analytic.len() - 1];

        analytic.real_into(&mut real).expect("real projection");
        analytic
            .quadrature_into(&mut quadrature)
            .expect("quadrature projection");
        analytic
            .envelope_into(&mut envelope)
            .expect("envelope projection");
        analytic.phase_into(&mut phase).expect("phase projection");
        analytic
            .instantaneous_frequency_into(&mut frequency)
            .expect("frequency projection");

        assert_eq!(real, analytic.real());
        assert_eq!(quadrature, analytic.quadrature());
        for (actual, expected) in envelope.iter().zip(analytic.envelope().iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
        for (actual, expected) in phase.iter().zip(analytic.phase().iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
        for (actual, expected) in frequency
            .iter()
            .zip(analytic.instantaneous_frequency().iter())
        {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn caller_owned_observables_reject_length_mismatch() {
        let analytic = analytic_fixture();
        let mut output = [0.0; 2];

        assert!(matches!(
            analytic.envelope_into(&mut output),
            Err(HilbertError::LengthMismatch)
        ));
        assert!(matches!(
            analytic.instantaneous_frequency_into(&mut []),
            Err(HilbertError::LengthMismatch)
        ));
    }
}
