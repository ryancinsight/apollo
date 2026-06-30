//! Reusable Mellin plan metadata surface.

use crate::domain::contracts::error::{MellinError, MellinResult};
use crate::domain::metadata::scale::MellinScaleConfig;
use crate::infrastructure::kernel::resample::{
    calculate_log_resample, log_frequency_spectrum, mellin_moment,
};
use apollo_fft::{f16, PrecisionProfile};
use eunomia::Complex64;
use std::borrow::Cow;

/// Dense Mellin log-frequency spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct MellinSpectrum {
    values: Vec<Complex64>,
}

impl MellinSpectrum {
    /// Create spectrum storage from computed values.
    #[must_use]
    pub fn new(values: Vec<Complex64>) -> Self {
        Self { values }
    }

    /// Borrow spectrum coefficients.
    #[must_use]
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Return coefficient count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return true when no coefficients are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Reusable Mellin transform plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MellinPlan {
    config: MellinScaleConfig,
}

impl MellinPlan {
    /// Create a validated Mellin transform plan.
    pub fn new(samples: usize, min_scale: f64, max_scale: f64) -> MellinResult<Self> {
        Ok(Self {
            config: MellinScaleConfig::new(samples, min_scale, max_scale)?,
        })
    }

    /// Return the validated scale configuration.
    #[must_use]
    pub const fn config(self) -> MellinScaleConfig {
        self.config
    }

    /// Resample an input signal onto this plan's logarithmic scale grid.
    pub fn forward_resample(
        &self,
        signal: &[f64],
        signal_min: f64,
        signal_max: f64,
        output: &mut [f64],
    ) -> MellinResult<()> {
        validate_signal_domain(signal, signal_min, signal_max)?;
        validate_output_len(output.len(), self.config.samples())?;

        calculate_log_resample(
            signal,
            signal_min,
            signal_max,
            output,
            self.config.min_scale(),
            self.config.max_scale(),
        );

        Ok(())
    }

    /// Resample a Leto 1D view onto this plan's logarithmic scale grid.
    ///
    /// C-contiguous views borrow directly; strided views copy once into
    /// logical order before reusing the canonical slice resampling path.
    pub fn forward_resample_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
        signal_min: f64,
        signal_max: f64,
    ) -> MellinResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
        let signal = leto_view1_cow(signal);
        let mut output = vec![0.0; self.config.samples()];
        self.forward_resample(&signal, signal_min, signal_max, &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Resample typed input onto the logarithmic scale grid into caller-owned storage.
    ///
    /// The owner implementation remains the `f64` log-resampling kernel.
    /// Lower storage profiles convert input once into `f64` and quantize once
    /// when writing the caller-owned output.
    pub fn forward_resample_typed_into<T: MellinStorage>(
        &self,
        signal: &[T],
        signal_min: f64,
        signal_max: f64,
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> MellinResult<()> {
        T::forward_resample_into(self, signal, signal_min, signal_max, output, profile)
    }

    /// Resample typed Leto input onto this plan's logarithmic scale grid.
    pub fn forward_resample_leto_typed<T: MellinStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        signal_min: f64,
        signal_max: f64,
        profile: PrecisionProfile,
    ) -> MellinResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = leto_view1_cow(signal);
        let mut output = vec![T::from_f64(0.0); self.config.samples()];
        self.forward_resample_typed_into(&signal, signal_min, signal_max, &mut output, profile)?;
        leto_array1_from_slice(&output)
    }

    /// Evaluate the real Mellin moment `M(s) = int f(r) r^(s-1) dr`.
    pub fn moment(
        &self,
        signal: &[f64],
        signal_min: f64,
        signal_max: f64,
        exponent: f64,
    ) -> MellinResult<f64> {
        validate_signal_domain(signal, signal_min, signal_max)?;
        if !exponent.is_finite() {
            return Err(MellinError::InvalidExponent);
        }
        Ok(mellin_moment(signal, signal_min, signal_max, exponent))
    }

    /// Evaluate a real Mellin moment from a Leto 1D view.
    pub fn moment_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
        signal_min: f64,
        signal_max: f64,
        exponent: f64,
    ) -> MellinResult<f64> {
        let signal = leto_view1_cow(signal);
        self.moment(&signal, signal_min, signal_max, exponent)
    }

    /// Evaluate the real Mellin moment for typed input storage.
    pub fn moment_typed<T: MellinStorage>(
        &self,
        signal: &[T],
        signal_min: f64,
        signal_max: f64,
        exponent: f64,
        profile: PrecisionProfile,
    ) -> MellinResult<f64> {
        T::moment(self, signal, signal_min, signal_max, exponent, profile)
    }

    /// Evaluate a real Mellin moment from typed Leto input storage.
    pub fn moment_leto_typed<T: MellinStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        signal_min: f64,
        signal_max: f64,
        exponent: f64,
        profile: PrecisionProfile,
    ) -> MellinResult<f64> {
        let signal = leto_view1_cow(signal);
        self.moment_typed(&signal, signal_min, signal_max, exponent, profile)
    }

    /// Compute the direct log-frequency Mellin spectrum over this plan's scale grid.
    pub fn forward_spectrum(
        &self,
        signal: &[f64],
        signal_min: f64,
        signal_max: f64,
    ) -> MellinResult<MellinSpectrum> {
        let mut log_samples = vec![0.0; self.config.samples()];
        self.forward_resample(signal, signal_min, signal_max, &mut log_samples)?;
        Ok(MellinSpectrum::new(log_frequency_spectrum(
            &log_samples,
            self.config.min_scale().ln(),
            self.config.max_scale().ln(),
        )))
    }

    /// Compute the direct log-frequency Mellin spectrum from a Leto 1D view.
    pub fn forward_spectrum_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
        signal_min: f64,
        signal_max: f64,
    ) -> MellinResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let signal = leto_view1_cow(signal);
        let spectrum = self.forward_spectrum(&signal, signal_min, signal_max)?;
        leto_array1_from_slice(spectrum.values())
    }

    /// Compute the direct log-frequency Mellin spectrum for typed input storage.
    pub fn forward_spectrum_typed<T: MellinStorage>(
        &self,
        signal: &[T],
        signal_min: f64,
        signal_max: f64,
        profile: PrecisionProfile,
    ) -> MellinResult<MellinSpectrum> {
        T::forward_spectrum(self, signal, signal_min, signal_max, profile)
    }

    /// Compute the direct log-frequency Mellin spectrum from typed Leto input storage.
    pub fn forward_spectrum_leto_typed<T: MellinStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        signal_min: f64,
        signal_max: f64,
        profile: PrecisionProfile,
    ) -> MellinResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let signal = leto_view1_cow(signal);
        let spectrum = self.forward_spectrum_typed(&signal, signal_min, signal_max, profile)?;
        leto_array1_from_slice(spectrum.values())
    }

    /// Recover the time-domain signal from a log-frequency Mellin spectrum.
    ///
    /// # Mathematical contract
    ///
    /// The forward spectrum computes `F[k] = du · DFT{g}[k]` where
    /// `g[n] = f(exp(u_n))` and `u_n = ln(min_scale) + n·du`.
    /// The inverse applies `g[n] = (1/(N·du)) · Σ_k F[k] exp(2πi·kn/N)` then
    /// linearly resamples the log-domain result back onto
    /// `[output_min, output_max]` at `output.len()` equally spaced points.
    ///
    /// # Errors
    ///
    /// Returns `MellinError::SpectrumLengthMismatch` when `spectrum.len()`
    /// differs from `self.config.samples()`.  Returns
    /// `MellinError::InvalidSignalBound` when `output_min` or `output_max`
    /// are not finite and positive, or `MellinError::InvalidSignalOrder` when
    /// `output_min >= output_max`.
    pub fn inverse_spectrum(
        &self,
        spectrum: &MellinSpectrum,
        output_min: f64,
        output_max: f64,
        output: &mut [f64],
    ) -> MellinResult<()> {
        if spectrum.len() != self.config.samples() {
            return Err(MellinError::SpectrumLengthMismatch);
        }
        if output.is_empty() {
            return Err(MellinError::EmptySignal);
        }
        if !output_min.is_finite()
            || !output_max.is_finite()
            || output_min <= 0.0
            || output_max <= 0.0
        {
            return Err(MellinError::InvalidSignalBound);
        }
        if output_min >= output_max {
            return Err(MellinError::InvalidSignalOrder);
        }

        let log_min = self.config.min_scale().ln();
        let log_max = self.config.max_scale().ln();

        let log_samples = crate::infrastructure::kernel::resample::inverse_log_frequency_spectrum(
            spectrum.values(),
            log_min,
            log_max,
        );

        crate::infrastructure::kernel::resample::exp_resample(
            &log_samples,
            log_min,
            log_max,
            output,
            output_min,
            output_max,
        );

        Ok(())
    }

    /// Recover an output signal from a Mellin spectrum into a Mnemosyne-backed Leto array.
    pub fn inverse_spectrum_leto(
        &self,
        spectrum: &MellinSpectrum,
        output_min: f64,
        output_max: f64,
        output_len: usize,
    ) -> MellinResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
        let mut output = vec![0.0; output_len];
        self.inverse_spectrum(spectrum, output_min, output_max, &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Recover an output signal from a Leto spectrum view.
    pub fn inverse_spectrum_from_leto(
        &self,
        spectrum: leto::ArrayView1<'_, Complex64>,
        output_min: f64,
        output_max: f64,
        output_len: usize,
    ) -> MellinResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
        let spectrum = leto_view1_cow(spectrum);
        self.inverse_spectrum_leto(
            &MellinSpectrum::new(spectrum.into_owned()),
            output_min,
            output_max,
            output_len,
        )
    }
}

/// Real storage accepted by typed Mellin input and log-resample output paths.
pub trait MellinStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `f64` arithmetic path.
    fn to_f64(self) -> f64;

    /// Convert owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

    /// Resample typed input into caller-owned typed output.
    fn forward_resample_into(
        plan: &MellinPlan,
        signal: &[Self],
        signal_min: f64,
        signal_max: f64,
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> MellinResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_signal_domain_typed(signal, signal_min, signal_max)?;
        validate_output_len(output.len(), plan.config.samples())?;
        let input64: Vec<f64> = signal.iter().copied().map(Self::to_f64).collect();
        let mut output64 = vec![0.0_f64; plan.config.samples()];
        plan.forward_resample(&input64, signal_min, signal_max, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.into_iter()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }

    /// Evaluate a real Mellin moment from typed input storage.
    fn moment(
        plan: &MellinPlan,
        signal: &[Self],
        signal_min: f64,
        signal_max: f64,
        exponent: f64,
        profile: PrecisionProfile,
    ) -> MellinResult<f64> {
        validate_profile(profile, Self::PROFILE)?;
        validate_signal_domain_typed(signal, signal_min, signal_max)?;
        let input64: Vec<f64> = signal.iter().copied().map(Self::to_f64).collect();
        plan.moment(&input64, signal_min, signal_max, exponent)
    }

    /// Compute a log-frequency Mellin spectrum from typed input storage.
    fn forward_spectrum(
        plan: &MellinPlan,
        signal: &[Self],
        signal_min: f64,
        signal_max: f64,
        profile: PrecisionProfile,
    ) -> MellinResult<MellinSpectrum> {
        validate_profile(profile, Self::PROFILE)?;
        validate_signal_domain_typed(signal, signal_min, signal_max)?;
        let input64: Vec<f64> = signal.iter().copied().map(Self::to_f64).collect();
        plan.forward_spectrum(&input64, signal_min, signal_max)
    }
}

impl MellinStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_resample_into(
        plan: &MellinPlan,
        signal: &[Self],
        signal_min: f64,
        signal_max: f64,
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> MellinResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_resample(signal, signal_min, signal_max, output)
    }

    fn moment(
        plan: &MellinPlan,
        signal: &[Self],
        signal_min: f64,
        signal_max: f64,
        exponent: f64,
        profile: PrecisionProfile,
    ) -> MellinResult<f64> {
        validate_profile(profile, Self::PROFILE)?;
        plan.moment(signal, signal_min, signal_max, exponent)
    }

    fn forward_spectrum(
        plan: &MellinPlan,
        signal: &[Self],
        signal_min: f64,
        signal_max: f64,
        profile: PrecisionProfile,
    ) -> MellinResult<MellinSpectrum> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_spectrum(signal, signal_min, signal_max)
    }
}

impl MellinStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl MellinStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

fn validate_output_len(actual: usize, expected: usize) -> MellinResult<()> {
    if actual != expected {
        return Err(MellinError::LengthMismatch);
    }
    Ok(())
}

fn validate_signal_domain(signal: &[f64], signal_min: f64, signal_max: f64) -> MellinResult<()> {
    if signal.is_empty() {
        return Err(MellinError::EmptySignal);
    }
    if !signal_min.is_finite() || !signal_max.is_finite() || signal_min <= 0.0 || signal_max <= 0.0
    {
        return Err(MellinError::InvalidSignalBound);
    }
    if signal_min >= signal_max {
        return Err(MellinError::InvalidSignalOrder);
    }
    Ok(())
}

fn validate_signal_domain_typed<T>(
    signal: &[T],
    signal_min: f64,
    signal_max: f64,
) -> MellinResult<()> {
    if signal.is_empty() {
        return Err(MellinError::EmptySignal);
    }
    if !signal_min.is_finite() || !signal_max.is_finite() || signal_min <= 0.0 || signal_max <= 0.0
    {
        return Err(MellinError::InvalidSignalBound);
    }
    if signal_min >= signal_max {
        return Err(MellinError::InvalidSignalOrder);
    }
    Ok(())
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> MellinResult<()> {
    if apollo_fft::application::utilities::leto_interop::profile_matches(actual, expected) {
        Ok(())
    } else {
        Err(MellinError::PrecisionMismatch)
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> Cow<'_, [T]> {
    apollo_fft::application::utilities::leto_interop::view1_cow(&view)
}
fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> MellinResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    apollo_fft::application::utilities::leto_interop::try_array1_from_slice(values)
        .ok_or(MellinError::LengthMismatch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
        let plan = MellinPlan::new(5, 1.0, 4.0).expect("plan");
        let signal64 = [1.0_f64, 1.5, 2.25, 3.0, 4.0];
        let mut expected = [0.0_f64; 5];
        plan.forward_resample(&signal64, 1.0, 4.0, &mut expected)
            .expect("resample");

        let mut out64 = [0.0_f64; 5];
        plan.forward_resample_typed_into(
            &signal64,
            1.0,
            4.0,
            &mut out64,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect("typed f64 resample");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let signal32 = signal64.map(|value| value as f32);
        let represented32: Vec<f64> = signal32.iter().map(|value| f64::from(*value)).collect();
        let mut expected32 = [0.0_f64; 5];
        plan.forward_resample(&represented32, 1.0, 4.0, &mut expected32)
            .expect("represented f32 resample");
        let mut out32 = [0.0_f32; 5];
        plan.forward_resample_typed_into(
            &signal32,
            1.0,
            4.0,
            &mut out32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed f32 resample");
        for (actual, expected) in out32.iter().zip(expected32.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }

        let signal16 = signal64.map(|value| f16::from_f32(value as f32));
        let represented16: Vec<f64> = signal16
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .collect();
        let mut expected16 = [0.0_f64; 5];
        plan.forward_resample(&represented16, 1.0, 4.0, &mut expected16)
            .expect("represented f16 resample");
        let mut out16 = [f16::from_f32(0.0); 5];
        plan.forward_resample_typed_into(
            &signal16,
            1.0,
            4.0,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 resample");
        for (actual, expected) in out16.iter().zip(expected16.iter()) {
            let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
        }
    }

    #[test]
    fn typed_moment_and_spectrum_match_represented_owner_paths() {
        let plan = MellinPlan::new(8, 1.0, 4.0).expect("plan");
        let signal32 = [2.0_f32; 65];
        let represented: Vec<f64> = signal32.iter().map(|value| f64::from(*value)).collect();
        let expected_moment = plan.moment(&represented, 1.0, 4.0, 1.0).expect("moment");
        let actual_moment = plan
            .moment_typed(
                &signal32,
                1.0,
                4.0,
                1.0,
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect("typed moment");
        assert_abs_diff_eq!(actual_moment, expected_moment, epsilon = 1.0e-12);

        let expected_spectrum = plan
            .forward_spectrum(&represented, 1.0, 4.0)
            .expect("spectrum");
        let actual_spectrum = plan
            .forward_spectrum_typed(&signal32, 1.0, 4.0, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed spectrum");
        for (actual, expected) in actual_spectrum
            .values()
            .iter()
            .zip(expected_spectrum.values().iter())
        {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = MellinPlan::new(4, 1.0, 2.0).expect("plan");
        let signal = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        assert!(matches!(
            plan.forward_resample_typed_into(
                &signal,
                1.0,
                2.0,
                &mut output,
                PrecisionProfile::HIGH_ACCURACY_F64
            ),
            Err(MellinError::PrecisionMismatch)
        ));
    }

    #[test]
    fn leto_resample_matches_slice_reference() {
        let plan = MellinPlan::new(5, 1.0, 4.0).expect("plan");
        let signal = vec![1.0_f64, 1.5, 2.25, 3.0, 4.0];
        let input = leto::Array1::from_shape_vec([5], signal.clone()).expect("leto input");
        let mut expected = vec![0.0; 5];
        plan.forward_resample(&signal, 1.0, 4.0, &mut expected)
            .expect("slice resample");

        let actual = plan
            .forward_resample_leto(input.view(), 1.0, 4.0)
            .expect("leto resample");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_strided_resample_matches_slice_reference() {
        let plan = MellinPlan::new(5, 1.0, 4.0).expect("plan");
        let backing = vec![1.0_f64, 99.0, 1.5, 99.0, 2.25, 99.0, 3.0, 99.0, 4.0, 99.0];
        let input = leto::Array1::from_shape_vec([10], backing).expect("leto input");
        let strided = input.view().slice(&[(0, 10, 2)]).expect("strided view");
        let signal = [1.0_f64, 1.5, 2.25, 3.0, 4.0];
        let mut expected = vec![0.0; 5];
        plan.forward_resample(&signal, 1.0, 4.0, &mut expected)
            .expect("slice resample");

        let actual = plan
            .forward_resample_leto(strided, 1.0, 4.0)
            .expect("leto resample");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_typed_resample_matches_typed_slice_reference() {
        let plan = MellinPlan::new(5, 1.0, 4.0).expect("plan");
        let signal = vec![1.0_f32, 1.5, 2.25, 3.0, 4.0];
        let input = leto::Array1::from_shape_vec([5], signal.clone()).expect("leto input");
        let mut expected = vec![0.0_f32; 5];
        plan.forward_resample_typed_into(
            &signal,
            1.0,
            4.0,
            &mut expected,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed slice resample");

        let actual = plan
            .forward_resample_leto_typed(
                input.view(),
                1.0,
                4.0,
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect("typed leto resample");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn leto_moment_and_spectrum_match_slice_reference() {
        let plan = MellinPlan::new(8, 1.0, 4.0).expect("plan");
        let signal = vec![2.0_f64; 65];
        let input = leto::Array1::from_shape_vec([65], signal.clone()).expect("leto input");
        let expected_moment = plan.moment(&signal, 1.0, 4.0, 1.0).expect("moment");
        let actual_moment = plan
            .moment_leto(input.view(), 1.0, 4.0, 1.0)
            .expect("leto moment");
        assert_abs_diff_eq!(actual_moment, expected_moment, epsilon = 1.0e-12);

        let expected_spectrum = plan.forward_spectrum(&signal, 1.0, 4.0).expect("spectrum");
        let actual_spectrum = plan
            .forward_spectrum_leto(input.view(), 1.0, 4.0)
            .expect("leto spectrum");
        let actual_spectrum = actual_spectrum.view();
        let actual_spectrum = actual_spectrum.as_slice().expect("contiguous spectrum");

        for (actual, expected) in actual_spectrum
            .iter()
            .zip(expected_spectrum.values().iter())
        {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_inverse_spectrum_matches_slice_reference() {
        let plan = MellinPlan::new(8, 1.0, 4.0).expect("plan");
        let signal = vec![2.0_f64; 65];
        let spectrum = plan.forward_spectrum(&signal, 1.0, 4.0).expect("spectrum");
        let mut expected = vec![0.0; 65];
        plan.inverse_spectrum(&spectrum, 1.0, 4.0, &mut expected)
            .expect("slice inverse");

        let actual = plan
            .inverse_spectrum_leto(&spectrum, 1.0, 4.0, 65)
            .expect("leto inverse");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous inverse");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let spectrum_array =
            leto::Array1::from_shape_vec([spectrum.len()], spectrum.values().to_vec())
                .expect("leto spectrum");
        let from_leto = plan
            .inverse_spectrum_from_leto(spectrum_array.view(), 1.0, 4.0, 65)
            .expect("leto spectrum inverse");
        let from_leto = from_leto.view();
        let from_leto = from_leto.as_slice().expect("contiguous inverse");
        for (actual, expected) in from_leto.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }
}
