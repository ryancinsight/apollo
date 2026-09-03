//! Reusable continuous wavelet transform plan.

use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::metadata::wavelet::ContinuousWavelet;
use crate::domain::spectrum::coefficients::CwtCoefficients;
use crate::infrastructure::kernel::continuous::coefficient;
use crate::infrastructure::kernel::continuous::convolution::{CwtSpectrum, FFT_CWT_LEN_THRESHOLD};
use crate::WaveletStorage;
use apollo_fft::PrecisionProfile;
use leto::Array2;

/// Reusable real-valued 1D CWT plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CwtPlan {
    len: usize,
    scales: Vec<f64>,
    wavelet: ContinuousWavelet,
}

impl CwtPlan {
    /// Create a CWT plan for a real-valued signal length and scale list.
    pub fn new(len: usize, scales: Vec<f64>, wavelet: ContinuousWavelet) -> WaveletResult<Self> {
        if len == 0 {
            return Err(WaveletError::EmptySignal);
        }
        if scales.is_empty() {
            return Err(WaveletError::EmptyScales);
        }
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(WaveletError::InvalidScale);
        }
        Ok(Self {
            len,
            scales,
            wavelet,
        })
    }

    /// Return signal length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return true when signal length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return scales.
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Return mother wavelet descriptor.
    #[must_use]
    pub const fn wavelet(&self) -> ContinuousWavelet {
        self.wavelet
    }

    /// Execute the CWT. Output shape is `(scales, signal_len)`.
    ///
    /// Above [`FFT_CWT_LEN_THRESHOLD`] each scale row is evaluated as one
    /// circular cross-correlation through `apollo-fft`, costing
    /// O(scales · n log n) with `2n - 1` mother-wavelet evaluations per scale.
    /// Below it the direct O(scales · n²) kernel runs, which is also the
    /// differential oracle for the fast path
    /// (see [`crate::infrastructure::kernel::continuous::convolution`]).
    pub fn transform(&self, signal: &[f64]) -> WaveletResult<CwtCoefficients> {
        if signal.len() != self.len {
            return Err(WaveletError::LengthMismatch);
        }
        let output_len = self
            .scales
            .len()
            .checked_mul(self.len)
            .ok_or(WaveletError::CoefficientShapeMismatch)?;
        let values = if self.len >= FFT_CWT_LEN_THRESHOLD {
            self.transform_via_convolution(signal, output_len)
        } else {
            self.transform_direct(output_len, signal)
        };
        let values = Array2::from_shape_vec([self.scales.len(), self.len], values)
            .expect("CWT output shape");

        Ok(CwtCoefficients::new(self.scales.clone(), values))
    }

    /// Row-major coefficients from one shared signal spectrum, one scale row
    /// per parallel task.
    fn transform_via_convolution(&self, signal: &[f64], output_len: usize) -> Vec<f64> {
        let spectrum = CwtSpectrum::new(signal);
        let rows = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(
            self.scales.len(),
            |scale_index| {
                let mut row = vec![0.0; self.len];
                spectrum.row_into(self.wavelet, self.scales[scale_index], &mut row);
                row
            },
        );
        let mut values = Vec::with_capacity(output_len);
        for row in rows {
            values.extend_from_slice(&row);
        }
        values
    }

    /// Direct per-coefficient evaluation: the transform's specification, and
    /// the oracle the convolution path is verified against.
    ///
    /// Collects directly into the row-major output buffer; the indexed map
    /// preserves logical order while avoiding one allocation per scale row.
    fn transform_direct(&self, output_len: usize, signal: &[f64]) -> Vec<f64> {
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(output_len, |index| {
            let scale_index = index / self.len;
            let shift = index % self.len;
            coefficient(signal, self.wavelet, self.scales[scale_index], shift)
        })
    }

    /// Execute the CWT from a Leto 1D signal view.
    ///
    /// Contiguous Leto views are borrowed directly; strided views are copied once
    /// into logical order before reusing the canonical slice CWT kernel.
    pub fn transform_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
    ) -> WaveletResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        if signal.shape()[0] == 0 {
            return Err(WaveletError::EmptySignal);
        }
        let signal = apollo_leto_interop::view_cow(&signal);
        let coefficients = self.transform(signal.as_ref())?;
        apollo_leto_interop::try_dense_from_array(coefficients.values())
            .ok_or(WaveletError::CoefficientShapeMismatch)
    }

    /// Execute the CWT for `f64`, `f32`, or mixed `F16` storage into a
    /// caller-owned matrix with shape `(scales, signal_len)`.
    pub fn transform_typed_into<T: WaveletStorage>(
        &self,
        signal: &[T],
        output: &mut Array2<T>,
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        T::transform_cwt_into(self, signal, output, profile)
    }

    /// Execute the CWT from typed Leto signal storage.
    pub fn transform_leto_typed<T: WaveletStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> WaveletResult<leto::Array<T, leto::MnemosyneStorage<T>, 2>> {
        if signal.shape()[0] == 0 {
            return Err(WaveletError::EmptySignal);
        }
        let signal = apollo_leto_interop::view_cow(&signal);
        let mut output = Array2::<T>::from_elem([self.scales.len(), self.len], T::from_cpu(0.0));
        self.transform_typed_into(signal.as_ref(), &mut output, profile)?;
        apollo_leto_interop::try_dense_from_array(&output)
            .ok_or(WaveletError::CoefficientShapeMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_fft::F16;
    use eunomia::assert_abs_diff_eq;

    #[test]
    fn typed_cwt_paths_support_f64_f32_and_mixed_f16_storage() {
        let plan = CwtPlan::new(4, vec![1.0, 2.0], ContinuousWavelet::Morlet { omega0: 5.0 })
            .expect("valid CWT plan");
        let signal64 = [1.0_f64, -0.5, 0.25, 2.0];
        let expected = plan.transform(&signal64).expect("CWT");

        let mut out64 = Array2::<f64>::zeros([2, 4]);
        plan.transform_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 CWT");
        for (actual, expected) in out64.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let signal32 = signal64.map(|value| value as f32);
        let represented32 = signal32.map(f64::from);
        let expected32 = plan.transform(&represented32).expect("represented f32 CWT");
        let mut out32 = Array2::<f32>::zeros([2, 4]);
        plan.transform_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 CWT");
        for (actual, expected) in out32.iter().zip(expected32.values().iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }

        let signal16 = signal64.map(|value| F16::from_f32(value as f32));
        let represented16 = signal16.map(|value| f64::from(value.to_f32()));
        let expected16 = plan.transform(&represented16).expect("represented F16 CWT");
        let mut out16 = Array2::from_elem([2, 4], F16::from_f32(0.0));
        plan.transform_typed_into(
            &signal16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed F16 CWT");
        for (actual, expected) in out16.iter().zip(expected16.values().iter()) {
            let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
        }
    }

    #[test]
    fn leto_transform_matches_slice_reference() {
        let plan = CwtPlan::new(4, vec![1.0, 2.0], ContinuousWavelet::Morlet { omega0: 5.0 })
            .expect("valid CWT plan");
        let signal = [1.0_f64, -0.5, 0.25, 2.0];
        let leto_signal =
            leto::Array1::from_shape_vec([signal.len()], signal.to_vec()).expect("leto signal");
        let expected = plan.transform(&signal).expect("slice CWT");

        let actual = plan.transform_leto(leto_signal.view()).expect("leto CWT");
        let actual_view = actual.view();
        let actual = actual_view.as_slice().expect("contiguous leto output");
        for (actual, expected) in actual.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_strided_transform_matches_slice_reference() {
        let plan =
            CwtPlan::new(4, vec![1.0, 2.0], ContinuousWavelet::Ricker).expect("valid CWT plan");
        let signal = [1.0_f64, -0.5, 0.25, 2.0];
        let mut interleaved = Vec::with_capacity(signal.len() * 2);
        for value in signal {
            interleaved.push(value);
            interleaved.push(99.0);
        }
        let leto_signal =
            leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto signal");
        let strided = leto_signal
            .view()
            .slice(&[(0, signal.len() * 2, 2)])
            .expect("strided signal");
        let expected = plan.transform(&signal).expect("slice CWT");

        let actual = plan.transform_leto(strided).expect("leto CWT");
        let actual_view = actual.view();
        let actual = actual_view.as_slice().expect("contiguous leto output");
        for (actual, expected) in actual.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_leto_transform_matches_slice_reference() {
        let plan = CwtPlan::new(4, vec![1.0, 2.0], ContinuousWavelet::Morlet { omega0: 5.0 })
            .expect("valid CWT plan");
        let signal = [1.0_f32, -0.5, 0.25, 2.0];
        let leto_signal =
            leto::Array1::from_shape_vec([signal.len()], signal.to_vec()).expect("leto signal");
        let mut expected = Array2::<f32>::zeros([2, 4]);
        plan.transform_typed_into(&signal, &mut expected, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed slice CWT");

        let actual = plan
            .transform_leto_typed(leto_signal.view(), PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed leto CWT");
        let actual_view = actual.view();
        let actual = actual_view.as_slice().expect("contiguous leto output");
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    /// Difference bound between the convolution row and the direct sum.
    ///
    /// Direct naive summation over `n` terms carries `|error| ≤ n · ε · ‖x‖₂ ‖w‖₂`;
    /// the three-transform FFT convolution carries the classical
    /// `c · log₂ L · ε · ‖x‖₂ ‖w‖₂` growth with `c = 8` a conservative radix-2
    /// constant. The two computed results differ by at most the sum. The norms
    /// are measured from the inputs; only the `(n + 8 log₂ L) · ε` shape is
    /// assumed.
    fn plan_difference_bound(signal: &[f64], wavelet: ContinuousWavelet, scale: f64) -> f64 {
        use crate::infrastructure::kernel::continuous::convolution::transform_len;
        use crate::infrastructure::kernel::continuous::mother_wavelet;

        let n = signal.len();
        let signal_norm = signal.iter().map(|value| value * value).sum::<f64>().sqrt();
        let inv_sqrt_scale = 1.0 / scale.sqrt();
        let kernel_norm = (-(n as isize - 1)..n as isize)
            .map(|lag| {
                let weight = inv_sqrt_scale * mother_wavelet(wavelet, lag as f64 / scale);
                weight * weight
            })
            .sum::<f64>()
            .sqrt();
        (n as f64 + 8.0 * (transform_len(n) as f64).log2())
            * f64::EPSILON
            * signal_norm
            * kernel_norm
    }

    #[test]
    fn transform_matches_the_direct_kernel_above_and_below_the_threshold() {
        use crate::infrastructure::kernel::continuous::coefficient;
        use crate::infrastructure::kernel::continuous::convolution::FFT_CWT_LEN_THRESHOLD;

        let scales = vec![0.75_f64, 2.0, 6.5, 25.0];
        for wavelet in [
            ContinuousWavelet::Ricker,
            ContinuousWavelet::Morlet { omega0: 5.0 },
        ] {
            // Straddles the threshold in both directions, including the
            // exact boundary and a non-power-of-two length above it.
            for len in [
                1_usize,
                5,
                FFT_CWT_LEN_THRESHOLD - 1,
                FFT_CWT_LEN_THRESHOLD,
                129,
                256,
            ] {
                let signal: Vec<f64> = (0..len)
                    .map(|index| (index as f64 * 0.29).sin() - (index % 3) as f64 * 0.4)
                    .collect();
                let plan = CwtPlan::new(len, scales.clone(), wavelet).expect("valid CWT plan");
                let actual = plan.transform(&signal).expect("CWT");
                for (scale_index, &scale) in scales.iter().enumerate() {
                    let bound = plan_difference_bound(&signal, wavelet, scale);
                    for shift in 0..len {
                        let expected = coefficient(&signal, wavelet, scale, shift);
                        let actual = actual.values()[[scale_index, shift]];
                        assert!(
                            (actual - expected).abs() <= bound,
                            "len {len} scale {scale} shift {shift}: \
                             |{actual} - {expected}| exceeds derived bound {bound}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn typed_cwt_rejects_profile_and_shape_mismatch() {
        let plan = CwtPlan::new(4, vec![1.0], ContinuousWavelet::Ricker).expect("valid CWT plan");
        let signal = [1.0_f32, -1.0, 0.5, -0.25];
        let mut output = Array2::<f32>::zeros([1, 4]);
        assert!(matches!(
            plan.transform_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(WaveletError::PrecisionMismatch)
        ));

        let mut wrong_shape = Array2::<f32>::zeros([1, 3]);
        assert!(matches!(
            plan.transform_typed_into(
                &signal,
                &mut wrong_shape,
                PrecisionProfile::LOW_PRECISION_F32
            ),
            Err(WaveletError::CoefficientShapeMismatch)
        ));
    }
}
