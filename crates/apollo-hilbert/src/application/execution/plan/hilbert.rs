//! Reusable Hilbert transform plan.

use crate::domain::contracts::error::{HilbertError, HilbertResult};
use crate::domain::metadata::length::SignalLength;
use crate::domain::signal::analytic::{envelope_values_into, phase_values_into, AnalyticSignal};
use crate::infrastructure::kernel::direct::{
    analytic_signal, analytic_signal_into, hilbert_transform, hilbert_transform_into,
};
use apollo_fft::{f16, PrecisionProfile};
use num_complex::Complex64;
use mnemosyne::scratch::ScratchPool;

thread_local! {
    static TYPED_INPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static TYPED_OUTPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static OBSERVABLE_ANALYTIC_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Reusable 1D Hilbert transform plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HilbertPlan {
    length: SignalLength,
}

impl HilbertPlan {
    /// Create a Hilbert transform plan for a non-empty signal length.
    pub fn new(len: usize) -> HilbertResult<Self> {
        Ok(Self {
            length: SignalLength::new(len)?,
        })
    }

    /// Return validated signal length.
    #[must_use]
    pub const fn length(self) -> SignalLength {
        self.length
    }

    /// Return signal length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.length.get()
    }

    /// Return true when signal length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.length.is_empty()
    }

    /// Compute the analytic signal `x + i H{x}`.
    pub fn analytic_signal(&self, signal: &[f64]) -> HilbertResult<AnalyticSignal> {
        if signal.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        Ok(AnalyticSignal::new(analytic_signal(signal)?))
    }

    /// Compute the analytic signal `x + i H{x}` into caller-owned storage.
    pub fn analytic_signal_into(
        &self,
        signal: &[f64],
        output: &mut [Complex64],
    ) -> HilbertResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        analytic_signal_into(signal, output)
    }

    /// Compute the analytic signal for `f64`, `f32`, or mixed `f16` input storage.
    ///
    /// The owner computation remains the `f64` analytic-mask path. Lower
    /// storage profiles are converted once at input, so the output preserves
    /// the exact represented real values and `f64` quadrature observables.
    pub fn analytic_signal_typed<T: HilbertStorage>(
        &self,
        signal: &[T],
        profile: PrecisionProfile,
    ) -> HilbertResult<AnalyticSignal> {
        T::analytic_signal(self, signal, profile)
    }

    /// Compute only the Hilbert quadrature component.
    pub fn transform(&self, signal: &[f64]) -> HilbertResult<Vec<f64>> {
        if signal.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        hilbert_transform(signal)
    }

    /// Compute only the Hilbert quadrature component into caller-owned storage.
    pub fn transform_into(&self, signal: &[f64], output: &mut [f64]) -> HilbertResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        hilbert_transform_into(signal, output)
    }

    /// Compute Hilbert quadrature for `f64`, `f32`, or mixed `f16` storage.
    pub fn transform_typed_into<T: HilbertStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> HilbertResult<()> {
        T::transform_into(self, signal, output, profile)
    }

    /// Compute the instantaneous envelope from the analytic signal.
    pub fn envelope(&self, signal: &[f64]) -> HilbertResult<Vec<f64>> {
        if signal.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        let mut output = vec![0.0; self.len()];
        self.envelope_into(signal, &mut output)?;
        Ok(output)
    }

    /// Compute the instantaneous envelope into caller-owned storage.
    pub fn envelope_into(&self, signal: &[f64], output: &mut [f64]) -> HilbertResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        with_observable_analytic_workspace(self.len(), |analytic| {
            analytic_signal_into(signal, analytic)?;
            envelope_values_into(analytic, output)
        })
    }

    /// Compute the wrapped instantaneous phase from the analytic signal.
    pub fn phase(&self, signal: &[f64]) -> HilbertResult<Vec<f64>> {
        if signal.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        let mut output = vec![0.0; self.len()];
        self.phase_into(signal, &mut output)?;
        Ok(output)
    }

    /// Compute the wrapped instantaneous phase into caller-owned storage.
    pub fn phase_into(&self, signal: &[f64], output: &mut [f64]) -> HilbertResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(HilbertError::LengthMismatch);
        }
        with_observable_analytic_workspace(self.len(), |analytic| {
            analytic_signal_into(signal, analytic)?;
            phase_values_into(analytic, output)
        })
    }
}

fn with_typed_signal_workspace<R>(
    len: usize,
    f: impl FnOnce(&mut [f64]) -> HilbertResult<R>,
) -> HilbertResult<R> {
    TYPED_INPUT64_SCRATCH.with(|scratch| scratch.with_scratch(len, f))
}

fn with_typed_transform_workspaces<R>(
    len: usize,
    f: impl FnOnce(&mut [f64], &mut [f64]) -> HilbertResult<R>,
) -> HilbertResult<R> {
    TYPED_INPUT64_SCRATCH.with(|input| {
        input.with_scratch(len, |input64| {
            TYPED_OUTPUT64_SCRATCH.with(|output| {
                output.with_scratch(len, |output64| {
                    f(input64, output64)
                })
            })
        })
    })
}

fn with_observable_analytic_workspace<R>(
    len: usize,
    f: impl FnOnce(&mut [Complex64]) -> HilbertResult<R>,
) -> HilbertResult<R> {
    OBSERVABLE_ANALYTIC_SCRATCH.with(|scratch| scratch.with_scratch(len, f))
}

#[cfg(test)]
fn typed_workspace_capacities() -> (usize, usize) {
    TYPED_INPUT64_SCRATCH.with(|input| {
        TYPED_OUTPUT64_SCRATCH
            .with(|output| (input.capacity(), output.capacity()))
    })
}

#[cfg(test)]
fn observable_workspace_capacity() -> usize {
    OBSERVABLE_ANALYTIC_SCRATCH.with(|scratch| scratch.capacity())
}

/// Real storage accepted by typed Hilbert input and quadrature paths.
pub trait HilbertStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `f64` arithmetic path.
    fn to_f64(self) -> f64;

    /// Convert owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

    /// Compute the analytic signal from typed input storage.
    fn analytic_signal(
        plan: &HilbertPlan,
        signal: &[Self],
        profile: PrecisionProfile,
    ) -> HilbertResult<AnalyticSignal> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() {
            return Err(HilbertError::LengthMismatch);
        }
        with_typed_signal_workspace(plan.len(), |input64| {
            for (slot, sample) in input64.iter_mut().zip(signal.iter().copied()) {
                *slot = Self::to_f64(sample);
            }
            plan.analytic_signal(input64)
        })
    }

    /// Compute the Hilbert quadrature component into caller-owned storage.
    fn transform_into(
        plan: &HilbertPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> HilbertResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(HilbertError::LengthMismatch);
        }
        with_typed_transform_workspaces(plan.len(), |input64, output64| {
            for (slot, sample) in input64.iter_mut().zip(signal.iter().copied()) {
                *slot = Self::to_f64(sample);
            }
            plan.transform_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_f64(value);
            }
            Ok(())
        })
    }
}

impl HilbertStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn analytic_signal(
        plan: &HilbertPlan,
        signal: &[Self],
        profile: PrecisionProfile,
    ) -> HilbertResult<AnalyticSignal> {
        validate_profile(profile, Self::PROFILE)?;
        plan.analytic_signal(signal)
    }

    fn transform_into(
        plan: &HilbertPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> HilbertResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.transform_into(signal, output)
    }
}

impl HilbertStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl HilbertStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> HilbertResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(HilbertError::PrecisionMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn caller_owned_transform_matches_allocating_path() {
        let len = 16;
        let signal: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).cos())
            .collect();
        let plan = HilbertPlan::new(len).expect("plan");
        let expected = plan.transform(&signal).expect("quadrature");
        let mut output = vec![0.0; len];
        plan.transform_into(&signal, &mut output)
            .expect("caller-owned transform");
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn caller_owned_analytic_signal_matches_allocating_path() {
        let len = 16;
        let signal: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).cos())
            .collect();
        let plan = HilbertPlan::new(len).expect("plan");
        let expected = plan.analytic_signal(&signal).expect("analytic");
        let mut output = vec![Complex64::new(f64::NAN, f64::NAN); len];

        plan.analytic_signal_into(&signal, &mut output)
            .expect("caller-owned analytic");

        for (actual, expected) in output.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn caller_owned_envelope_and_phase_match_allocating_paths() {
        let len = 16;
        let signal: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).cos())
            .collect();
        let plan = HilbertPlan::new(len).expect("plan");
        let expected_envelope = plan.envelope(&signal).expect("envelope");
        let expected_phase = plan.phase(&signal).expect("phase");
        let mut envelope = vec![0.0; len];
        let mut phase = vec![0.0; len];

        plan.envelope_into(&signal, &mut envelope)
            .expect("caller-owned envelope");
        let after_envelope = observable_workspace_capacity();
        plan.phase_into(&signal, &mut phase)
            .expect("caller-owned phase");
        let after_phase = observable_workspace_capacity();

        assert!(after_envelope >= len);
        assert_eq!(after_phase, after_envelope);
        for (actual, expected) in envelope.iter().zip(expected_envelope.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
        for (actual, expected) in phase.iter().zip(expected_phase.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
        let len = 16;
        let signal64: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).cos())
            .collect();
        let plan = HilbertPlan::new(len).expect("plan");
        let expected = plan.transform(&signal64).expect("quadrature");

        let analytic64 = plan
            .analytic_signal_typed(&signal64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 analytic");
        for (sample, original) in analytic64.values().iter().zip(signal64.iter()) {
            assert_abs_diff_eq!(sample.re, *original, epsilon = 1.0e-12);
        }

        let mut out64 = vec![0.0_f64; len];
        plan.transform_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 transform");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let signal32: Vec<f32> = signal64.iter().map(|value| *value as f32).collect();
        let expected32_input: Vec<f64> = signal32.iter().map(|value| f64::from(*value)).collect();
        let expected32 = plan.transform(&expected32_input).expect("f32 represented");
        let mut out32 = vec![0.0_f32; len];
        plan.transform_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 transform");
        for (actual, expected) in out32.iter().zip(expected32.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }

        let signal16: Vec<f16> = signal64
            .iter()
            .map(|value| f16::from_f32(*value as f32))
            .collect();
        let expected16_input: Vec<f64> = signal16
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .collect();
        let expected16 = plan.transform(&expected16_input).expect("f16 represented");
        let mut out16 = vec![f16::from_f32(0.0); len];
        plan.transform_typed_into(
            &signal16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 transform");
        for (actual, expected) in out16.iter().zip(expected16.iter()) {
            let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
        }
    }

    #[test]
    fn typed_paths_reuse_workspace_capacity() {
        let len = 16;
        let signal: Vec<f32> = (0..len)
            .map(|n| (std::f32::consts::TAU * n as f32 / len as f32).cos())
            .collect();
        let plan = HilbertPlan::new(len).expect("plan");
        let mut first = vec![0.0_f32; len];
        let mut second = vec![0.0_f32; len];

        plan.transform_typed_into(&signal, &mut first, PrecisionProfile::LOW_PRECISION_F32)
            .expect("first typed transform");
        let after_first = typed_workspace_capacities();
        assert!(after_first.0 >= len);
        assert!(after_first.1 >= len);

        plan.transform_typed_into(&signal, &mut second, PrecisionProfile::LOW_PRECISION_F32)
            .expect("second typed transform");
        let after_second = typed_workspace_capacities();
        assert_eq!(after_second, after_first);

        for (actual, expected) in second.iter().zip(first.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = HilbertPlan::new(4).expect("plan");
        let signal = [1.0_f32, -2.0, 0.5, 0.25];
        let mut output = [0.0_f32; 4];
        assert!(matches!(
            plan.transform_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(HilbertError::PrecisionMismatch)
        ));
    }
}
