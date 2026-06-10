use apollo_fft::PrecisionProfile;
use apollo_stft::{StftRealOutputStorage, StftSpectrumInput};
use num_complex::Complex32;

use crate::application::plan::StftWgpuPlan;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::device::helpers::{leto_array1_from_slice, leto_view1_cow};
use crate::infrastructure::device::StftWgpuBackend;

impl StftWgpuBackend {
    /// Execute the inverse STFT (WOLA reconstruction) on the GPU.
    pub fn execute_inverse(
        &self,
        plan: &StftWgpuPlan,
        spectrum: &[Complex32],
        signal_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        if plan.frame_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: frame_len must be non-zero",
                    plan.frame_len(),
                    plan.hop_len()
                ),
            });
        }
        if plan.hop_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: hop_len must be non-zero",
                    plan.frame_len(),
                    plan.hop_len()
                ),
            });
        }
        if plan.hop_len() > plan.frame_len() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: hop_len must not exceed frame_len",
                    plan.frame_len(),
                    plan.hop_len()
                ),
            });
        }
        if signal_len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: signal_len must be non-zero",
                    plan.frame_len(),
                    plan.hop_len()
                ),
            });
        }
        let frame_count = 1 + signal_len.div_ceil(plan.hop_len());
        let expected = frame_count * plan.frame_len();
        if spectrum.len() != expected {
            return Err(WgpuError::LengthMismatch {
                expected,
                actual: spectrum.len(),
            });
        }
        self.kernel.execute_inverse(
            &self.device,
            spectrum,
            plan.frame_len(),
            plan.hop_len(),
            frame_count,
            signal_len,
        )
    }

    /// Execute the inverse STFT from a Leto complex `f32` spectrum view.
    pub fn execute_inverse_leto(
        &self,
        plan: &StftWgpuPlan,
        spectrum: leto::ArrayView1<'_, Complex32>,
        signal_len: usize,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let spectrum = leto_view1_cow(spectrum);
        let output = self.execute_inverse(plan, &spectrum, signal_len)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the inverse STFT with typed complex spectrum input and typed real output.
    pub fn execute_inverse_typed_into<I: StftSpectrumInput, O: StftRealOutputStorage>(
        &self,
        plan: &StftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        spectrum: &[I],
        signal_len: usize,
        output: &mut [O],
    ) -> WgpuResult<()> {
        let expected_in = I::PROFILE;
        if input_precision.storage != expected_in.storage
            || input_precision.compute != expected_in.compute
        {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let expected_out = O::PROFILE;
        if output_precision.storage != expected_out.storage
            || output_precision.compute != expected_out.compute
        {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        if output.len() != signal_len {
            return Err(WgpuError::LengthMismatch {
                expected: signal_len,
                actual: output.len(),
            });
        }
        let promoted = if std::any::TypeId::of::<I>() == std::any::TypeId::of::<Complex32>() {
            // Safety: I is Complex32, so &[I] is layout-compatible with &[Complex32].
            let slice_c32 = unsafe {
                std::slice::from_raw_parts(spectrum.as_ptr().cast::<Complex32>(), spectrum.len())
            };
            std::borrow::Cow::Borrowed(slice_c32)
        } else {
            let vec: Vec<Complex32> = spectrum
                .iter()
                .map(|v| {
                    let c = v.to_complex64();
                    Complex32::new(c.re as f32, c.im as f32)
                })
                .collect();
            std::borrow::Cow::Owned(vec)
        };
        let computed = self.execute_inverse(plan, &promoted, signal_len)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = O::from_f64(f64::from(value));
        }
        Ok(())
    }

    /// Execute typed inverse STFT from a Leto spectrum view.
    pub fn execute_inverse_leto_typed<I: StftSpectrumInput, O: StftRealOutputStorage>(
        &self,
        plan: &StftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        spectrum: leto::ArrayView1<'_, I>,
        signal_len: usize,
    ) -> WgpuResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>> {
        let spectrum = leto_view1_cow(spectrum);
        let mut output = vec![O::from_f64(0.0); signal_len];
        self.execute_inverse_typed_into(
            plan,
            input_precision,
            output_precision,
            &spectrum,
            signal_len,
            &mut output,
        )?;
        leto_array1_from_slice(&output)
    }
}
