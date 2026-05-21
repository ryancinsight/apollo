//! Precision-specific 1D FFT plan methods.

use super::FftPlan1D;
use crate::application::execution::kernel::mixed_radix::dispatch_inplace;
use crate::application::execution::kernel::{fft_forward, fft_inverse};
use crate::application::execution::plan::fft::workspace::uninit_copy_vec;
use crate::domain::metadata::precision::PrecisionProfile;
use half::f16;
use ndarray::Array1;
use num_complex::{Complex, Complex32, Complex64};

trait Plan1dReal32:
    Copy + crate::application::execution::plan::fft::workspace::UninitWorkspaceElement
{
    fn as_real(self) -> f32;
    fn from_real(value: f32) -> Self;
}

impl Plan1dReal32 for f32 {
    #[inline]
    fn as_real(self) -> f32 {
        self
    }

    #[inline]
    fn from_real(value: f32) -> Self {
        value
    }
}

impl Plan1dReal32 for f16 {
    #[inline]
    fn as_real(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_real(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl FftPlan1D {
    /// Forward transform of a real signal stored as `f32`.
    #[must_use]
    pub(crate) fn forward_real(&self, input: &Array1<f32>) -> Array1<Complex32> {
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
    pub(crate) fn inverse_real(&self, input: &Array1<Complex32>) -> Array1<f32> {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            self.inverse_real32_native(input)
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted)
                .mapv(|value| value as f32)
        }
    }

    pub(crate) fn forward_real_into(&self, input: &Array1<f32>, output: &mut Array1<Complex32>) {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            self.forward_real32_native_into(input, output);
        } else {
            output.assign(
                &self
                    .forward_real_to_complex(&input.mapv(f64::from))
                    .mapv(|value| Complex32::new(value.re as f32, value.im as f32)),
            );
        }
    }

    pub(crate) fn inverse_real_into(
        &self,
        input: &Array1<Complex32>,
        output: &mut Array1<f32>,
        scratch: &mut Array1<Complex32>,
    ) {
        if self.precision == PrecisionProfile::LOW_PRECISION_F32 {
            self.inverse_real32_native_into(input, output, scratch);
        } else {
            output.assign(
                &self
                    .inverse_complex_to_real(
                        &input
                            .mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im))),
                    )
                    .mapv(|value| value as f32),
            );
        }
    }

    /// Forward transform of a real signal stored as `f16`.
    ///
    /// For `MIXED_PRECISION_F16_F32`:
    /// - power-of-two lengths use a `Complex<f16>` working buffer (N × 4 bytes)
    ///   through the generic storage bridge,
    /// - non-power-of-two lengths use the f32 auto-kernel path to preserve
    ///   unified runtime kernel selection (including Rader).
    #[must_use]
    pub(crate) fn forward_half(&self, input: &Array1<f16>) -> Array1<Complex32> {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            if input.len().is_power_of_two() {
                Self::forward_half_compact_power_of_two(input)
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
    pub(crate) fn inverse_half(&self, input: &Array1<Complex32>) -> Array1<f16> {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            if input.len().is_power_of_two() {
                Self::inverse_half_compact_power_of_two(input)
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

    pub(crate) fn forward_half_into(&self, input: &Array1<f16>, output: &mut Array1<Complex32>) {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            if input.len().is_power_of_two() {
                output.assign(&Self::forward_half_compact_power_of_two(input));
            } else {
                self.forward_real32_native_into(input, output);
            }
        } else {
            output.assign(
                &self
                    .forward_real_to_complex(&input.mapv(|value| f64::from(value.to_f32())))
                    .mapv(|value| Complex32::new(value.re as f32, value.im as f32)),
            );
        }
    }

    pub(crate) fn inverse_half_into(
        &self,
        input: &Array1<Complex32>,
        output: &mut Array1<f16>,
        scratch: &mut Array1<Complex32>,
    ) {
        if self.precision == PrecisionProfile::MIXED_PRECISION_F16_F32 {
            if input.len().is_power_of_two() {
                output.assign(&Self::inverse_half_compact_power_of_two(input));
            } else {
                self.inverse_real32_native_into(input, output, scratch);
            }
        } else {
            output.assign(
                &self
                    .inverse_complex_to_real(
                        &input
                            .mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im))),
                    )
                    .mapv(|value| f16::from_f32(value as f32)),
            );
        }
    }

    fn forward_real32_native<T: Plan1dReal32>(&self, input: &Array1<T>) -> Array1<Complex32> {
        let mut output =
            Array1::<Complex32>::from_shape_vec(input.len(), uninit_copy_vec(input.len()))
                .expect("uninit Complex32 1D buffer length must match input len");
        self.forward_real32_native_into(input, &mut output);
        output
    }

    fn forward_real32_native_into<T: Plan1dReal32>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<Complex32>,
    ) {
        assert_eq!(input.len(), self.n, "forward input length mismatch");
        assert_eq!(output.len(), self.n, "forward output length mismatch");
        ndarray::Zip::from(&mut *output)
            .and(input)
            .for_each(|out, value| {
                *out = Complex32::new(value.as_real(), 0.0);
            });
        let slice = output.as_slice_mut().expect("Array must be contiguous");
        if let Some(twiddles) = &self.twiddle_fwd_32 {
            dispatch_inplace::<f32, false, false>(slice, Some(twiddles.as_ref()));
        } else {
            fft_forward(slice);
        }
    }

    fn inverse_real32_native<T: Plan1dReal32>(&self, input: &Array1<Complex32>) -> Array1<T> {
        let mut output = input.clone();
        let slice = output.as_slice_mut().expect("Array must be contiguous");
        if let Some(twiddles) = &self.twiddle_inv_32 {
            dispatch_inplace::<f32, true, true>(slice, Some(twiddles.as_ref()));
        } else {
            fft_inverse(slice);
        }
        let mut result = Array1::<T>::from_shape_vec(output.len(), uninit_copy_vec(output.len()))
            .expect("uninit real32 1D buffer length must match output len");
        ndarray::Zip::from(&mut result)
            .and(&output)
            .for_each(|out, value| {
                *out = T::from_real(value.re);
            });
        result
    }

    fn inverse_real32_native_into<T: Plan1dReal32>(
        &self,
        input: &Array1<Complex32>,
        output: &mut Array1<T>,
        scratch: &mut Array1<Complex32>,
    ) {
        assert_eq!(input.len(), self.n, "inverse input length mismatch");
        assert_eq!(output.len(), self.n, "inverse output length mismatch");
        assert_eq!(scratch.len(), self.n, "inverse scratch length mismatch");
        scratch.assign(input);
        let slice = scratch.as_slice_mut().expect("Array must be contiguous");
        if let Some(twiddles) = &self.twiddle_inv_32 {
            dispatch_inplace::<f32, true, true>(slice, Some(twiddles.as_ref()));
        } else {
            fft_inverse(slice);
        }
        ndarray::Zip::from(output)
            .and(&*scratch)
            .for_each(|out, value| {
                *out = T::from_real(value.re);
            });
    }

    fn forward_half_compact_power_of_two(input: &Array1<f16>) -> Array1<Complex32> {
        let input_slice = input.as_slice().expect("Array must be contiguous");
        let mut buf: Vec<Complex<f16>> = uninit_copy_vec(input_slice.len());
        for (slot, &value) in buf.iter_mut().zip(input_slice.iter()) {
            *slot = Complex::new(value, f16::ZERO);
        }
        fft_forward(&mut buf);

        let mut output = uninit_copy_vec(buf.len());
        for (slot, value) in output.iter_mut().zip(buf.into_iter()) {
            *slot = Complex32::new(value.re.to_f32(), value.im.to_f32());
        }
        Array1::from_vec(output)
    }

    fn inverse_half_compact_power_of_two(input: &Array1<Complex32>) -> Array1<f16> {
        let input_slice = input.as_slice().expect("Array must be contiguous");
        let mut buf: Vec<Complex<f16>> = uninit_copy_vec(input_slice.len());
        for (slot, &value) in buf.iter_mut().zip(input_slice.iter()) {
            *slot = Complex::new(f16::from_f32(value.re), f16::from_f32(value.im));
        }
        fft_inverse(&mut buf);

        let mut output = uninit_copy_vec(buf.len());
        for (slot, value) in output.iter_mut().zip(buf.into_iter()) {
            *slot = value.re;
        }
        Array1::from_vec(output)
    }
}
