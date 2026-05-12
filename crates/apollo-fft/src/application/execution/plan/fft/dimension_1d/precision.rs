//! Precision-specific 1D FFT plan methods.

use super::FftPlan1D;
use crate::application::execution::kernel::mixed_radix::{
    forward_inplace_32_with_twiddles, inverse_inplace_32_with_twiddles,
};
use crate::application::execution::kernel::{
    fft_forward, fft_forward_32, fft_inverse, fft_inverse_32,
};
use crate::domain::metadata::precision::PrecisionProfile;
use half::f16;
use ndarray::Array1;
use num_complex::{Complex, Complex32, Complex64};

impl FftPlan1D {
    /// Forward transform of a real signal stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array1<f32>) -> Array1<Complex32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut output = input.mapv(|value| Complex32::new(value, 0.0));
            let slice = output.as_slice_mut().expect("Array must be contiguous");
            if let Some(twiddles) = &self.twiddle_fwd_32 {
                forward_inplace_32_with_twiddles(slice, Some(twiddles.as_ref()));
            } else {
                fft_forward_32(slice);
            }
            output
        } else {
            let promoted = input.mapv(f64::from);
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    /// Inverse transform of an `f32`-storage complex spectrum.
    #[must_use]
    pub(crate) fn inverse_f32(&self, input: &Array1<Complex32>) -> Array1<f32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            let mut output = input.clone();
            let slice = output.as_slice_mut().expect("Array must be contiguous");
            if let Some(twiddles) = &self.twiddle_inv_32 {
                inverse_inplace_32_with_twiddles(slice, Some(twiddles.as_ref()));
            } else {
                fft_inverse_32(slice);
            }
            output.mapv(|value| value.re)
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| value as f32)
        }
    }

    /// Forward transform of a real signal stored as `f16`.
    ///
    /// For `MIXED_PRECISION_F16_F32`:
    /// - power-of-two lengths use a `Complex<f16>` working buffer (N × 4 bytes)
    ///   through the generic storage bridge,
    /// - non-power-of-two lengths use the f32 auto-kernel path to preserve
    ///   unified runtime kernel selection (including Bluestein).
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array1<f16>) -> Array1<Complex32> {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            if input.len().is_power_of_two() {
                // Pack real f16 input as compact complex storage (imaginary = 0).
                let mut buf: Vec<Complex<f16>> =
                    input.iter().map(|&v| Complex::new(v, f16::ZERO)).collect();
                fft_forward(&mut buf);
                // Convert compact spectrum to Complex32 at the output boundary.
                Array1::from_vec(
                    buf.into_iter()
                        .map(|cf| Complex32::new(cf.re.to_f32(), cf.im.to_f32()))
                        .collect(),
                )
            } else {
                let mut data = input.mapv(|value| Complex32::new(value.to_f32(), 0.0));
                fft_forward_32(data.as_slice_mut().expect("Array must be contiguous"));
                data
            }
        } else {
            let promoted = input.mapv(|value| f64::from(value.to_f32()));
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    /// Inverse transform of a complex spectrum to `f16` storage.
    ///
    /// For `MIXED_PRECISION_F16_F32`:
    /// - power-of-two lengths use a `Complex<f16>` working buffer,
    /// - non-power-of-two lengths use f32 auto-kernel inverse then quantize once
    ///   at the real-output boundary.
    #[must_use]
    pub(crate) fn inverse_f16(&self, input: &Array1<Complex32>) -> Array1<f16> {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            if input.len().is_power_of_two() {
                // Convert Complex32 spectrum to compact f16 working buffer.
                let mut buf: Vec<Complex<f16>> = input
                    .iter()
                    .map(|&v| Complex::new(f16::from_f32(v.re), f16::from_f32(v.im)))
                    .collect();
                fft_inverse(&mut buf);
                // Extract real parts as f16 (imaginary parts are ~0 by Hermitian symmetry).
                Array1::from_vec(buf.into_iter().map(|cf| cf.re).collect())
            } else {
                let mut data = input.clone();
                fft_inverse_32(data.as_slice_mut().expect("Array must be contiguous"));
                data.mapv(|value| f16::from_f32(value.re))
            }
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| f16::from_f32(value as f32))
        }
    }
}
