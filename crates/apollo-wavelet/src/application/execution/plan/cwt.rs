//! Reusable continuous wavelet transform plan.

use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::metadata::wavelet::ContinuousWavelet;
use crate::domain::spectrum::coefficients::CwtCoefficients;
use crate::infrastructure::kernel::continuous::coefficient;
use crate::WaveletStorage;
use apollo_fft::PrecisionProfile;
use moirai::ParallelSlice;
use leto::Array2;
use std::borrow::Cow;

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
    pub fn transform(&self, signal: &[f64]) -> WaveletResult<CwtCoefficients> {
        if signal.len() != self.len {
            return Err(WaveletError::LengthMismatch);
        }
        // Parallelize over the scale dimension; each scale row is independent.
        let rows: Vec<Vec<f64>> = self.scales.par().map_collect(|&scale| {
            (0..self.len)
                .map(|shift| coefficient(signal, self.wavelet, scale, shift))
                .collect()
        });
        let values = Array2::from_shape_fn([self.scales.len(), self.len], |[s, b]| rows[s][b]);
        Ok(CwtCoefficients::new(self.scales.clone(), values))
    }

    /// Execute the CWT from a Leto 1D signal view.
    ///
    /// Contiguous Leto views are borrowed directly; strided views are copied once
    /// into logical order before reusing the canonical slice CWT kernel.
    pub fn transform_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
    ) -> WaveletResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let signal = leto_view1_cow(signal)?;
        let coefficients = self.transform(signal.as_ref())?;
        leto_array2_from_dense(coefficients.values())
    }

    /// Execute the CWT for `f64`, `f32`, or mixed `f16` storage into a
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
        let signal = leto_view1_cow(signal)?;
        let mut output = Array2::<T>::from_elem([self.scales.len(), self.len], T::from_f64(0.0));
        self.transform_typed_into(signal.as_ref(), &mut output, profile)?;
        leto_array2_from_dense(&output)
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> WaveletResult<Cow<'_, [T]>> {
    if view.shape()[0] == 0 {
        return Err(WaveletError::EmptySignal);
    }
    Ok(apollo_fft::application::utilities::leto_interop::view1_cow(
        &view,
    ))
}

fn leto_array2_from_dense<T: Copy>(
    array: &Array2<T>,
) -> WaveletResult<leto::Array<T, leto::MnemosyneStorage<T>, 2>> {
    apollo_fft::application::utilities::leto_interop::try_dense_from_contiguous(array)
        .ok_or(WaveletError::CoefficientShapeMismatch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_fft::f16;
    use approx::assert_abs_diff_eq;

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

        let signal16 = signal64.map(|value| f16::from_f32(value as f32));
        let represented16 = signal16.map(|value| f64::from(value.to_f32()));
        let expected16 = plan.transform(&represented16).expect("represented f16 CWT");
        let mut out16 = Array2::from_elem([2, 4], f16::from_f32(0.0));
        plan.transform_typed_into(
            &signal16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 CWT");
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
