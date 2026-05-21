//! Precision-specific 1D FFT plan methods.

use super::FftPlan1D;
use crate::application::execution::kernel::mixed_radix::{
    forward_inplace_32_with_twiddles, inverse_inplace_32_with_twiddles,
};
use crate::application::execution::kernel::{fft_forward, fft_inverse};
use crate::application::execution::plan::fft::workspace::uninit_copy_vec;
use crate::domain::metadata::precision::PrecisionProfile;
use half::f16;
use ndarray::Array1;
use num_complex::{Complex, Complex32, Complex64};

trait Plan1dReal32:
    Copy + crate::application::execution::plan::fft::workspace::UninitWorkspaceElement
{
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl Plan1dReal32 for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl Plan1dReal32 for f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl FftPlan1D {
    /// Forward transform of a real signal stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array1<f32>) -> Array1<Complex32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            self.forward_real32_native(input)
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
            self.inverse_real32_native(input)
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
                Self::forward_f16_compact_power_of_two(input)
            } else {
                self.forward_real32_native(input)
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
                Self::inverse_f16_compact_power_of_two(input)
            } else {
                self.inverse_real32_native(input)
            }
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| f16::from_f32(value as f32))
        }
    }

    fn forward_real32_native<T: Plan1dReal32>(&self, input: &Array1<T>) -> Array1<Complex32> {
        let mut output =
            Array1::<Complex32>::from_shape_vec(input.len(), uninit_copy_vec(input.len()))
                .expect("uninit Complex32 1D buffer length must match input len");
        ndarray::Zip::from(&mut output)
            .and(input)
            .for_each(|out, value| {
                *out = Complex32::new(value.to_f32(), 0.0);
            });
        let slice = output.as_slice_mut().expect("Array must be contiguous");
        if let Some(twiddles) = &self.twiddle_fwd_32 {
            forward_inplace_32_with_twiddles(slice, Some(twiddles.as_ref()));
        } else {
            fft_forward(slice);
        }
        output
    }

    fn inverse_real32_native<T: Plan1dReal32>(&self, input: &Array1<Complex32>) -> Array1<T> {
        let mut output = input.clone();
        let slice = output.as_slice_mut().expect("Array must be contiguous");
        if let Some(twiddles) = &self.twiddle_inv_32 {
            inverse_inplace_32_with_twiddles(slice, Some(twiddles.as_ref()));
        } else {
            fft_inverse(slice);
        }
        let mut result = Array1::<T>::from_shape_vec(output.len(), uninit_copy_vec(output.len()))
            .expect("uninit real32 1D buffer length must match output len");
        ndarray::Zip::from(&mut result)
            .and(&output)
            .for_each(|out, value| {
                *out = T::from_f32(value.re);
            });
        result
    }

    fn forward_f16_compact_power_of_two(input: &Array1<f16>) -> Array1<Complex32> {
        let input_slice = input.as_slice().expect("Array must be contiguous");
        let mut buf: Vec<Complex<f16>> = uninit_copy_vec(input_slice.len());
        for (slot, &value) in buf.iter_mut().zip(input_slice.iter()) {
            *slot = Complex::new(value, f16::ZERO);
        }
        fft_forward(&mut buf);

        let mut output = uninit_copy_vec(buf.len());
        for (slot, value) in output.iter_mut().zip(buf) {
            *slot = Complex32::new(value.re.to_f32(), value.im.to_f32());
        }
        Array1::from_vec(output)
    }

    fn inverse_f16_compact_power_of_two(input: &Array1<Complex32>) -> Array1<f16> {
        let input_slice = input.as_slice().expect("Array must be contiguous");
        let mut buf: Vec<Complex<f16>> = uninit_copy_vec(input_slice.len());
        for (slot, &value) in buf.iter_mut().zip(input_slice.iter()) {
            *slot = Complex::new(f16::from_f32(value.re), f16::from_f32(value.im));
        }
        fft_inverse(&mut buf);

        let mut output = uninit_copy_vec(buf.len());
        for (slot, value) in output.iter_mut().zip(buf) {
            *slot = value.re;
        }
        Array1::from_vec(output)
    }
}
