use apollo_fft::PrecisionProfile;
use crate::{StftRealStorage, StftSpectrumStorage};
use num_complex::Complex32;

use crate::infrastructure::transport::gpu::application::plan::StftWgpuPlan;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::device::helpers::{leto_array1_from_slice, leto_view1_cow};
use crate::infrastructure::transport::gpu::infrastructure::device::StftWgpuBackend;

impl StftWgpuBackend {
    /// Execute the forward STFT on `signal` using the supplied plan.
    pub fn execute_forward(
        &self,
        plan: &StftWgpuPlan,
        signal: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
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
        if signal.len() < plan.frame_len() {
            return Err(WgpuError::InputTooShort {
                min: plan.frame_len(),
                actual: signal.len(),
            });
        }
        let frame_count = 1 + signal.len().div_ceil(plan.hop_len());
        self.kernel.execute_forward_fft(
            &self.device,
            signal,
            plan.frame_len(),
            plan.hop_len(),
            frame_count,
        )
    }

    /// Execute the forward STFT from a Leto real `f32` signal view.
    pub fn execute_forward_leto(
        &self,
        plan: &StftWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let signal = leto_view1_cow(signal);
        let output = self.execute_forward(plan, &signal)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the forward STFT with typed real input and typed complex spectrum output.
    pub fn execute_forward_typed_into<I: StftRealStorage, O: StftSpectrumStorage>(
        &self,
        plan: &StftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        signal: &[I],
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
        let represented = if std::any::TypeId::of::<I>() == std::any::TypeId::of::<f32>() {
            // Safety: I is f32, so &[I] is layout-compatible with &[f32].
            let slice_f32 =
                unsafe { std::slice::from_raw_parts(signal.as_ptr().cast::<f32>(), signal.len()) };
            std::borrow::Cow::Borrowed(slice_f32)
        } else {
            let vec: Vec<f32> = signal.iter().map(|v| v.to_f64() as f32).collect();
            std::borrow::Cow::Owned(vec)
        };
        let computed = self.execute_forward(plan, &represented)?;
        if output.len() != computed.len() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: output length does not match computed frame count * frame_len",
                    plan.frame_len(),
                    plan.hop_len()
                ),
            });
        }
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = O::from_complex64(num_complex::Complex64::new(
                f64::from(value.re),
                f64::from(value.im),
            ));
        }
        Ok(())
    }

    /// Execute typed forward STFT from a Leto real signal view.
    pub fn execute_forward_leto_typed<I: StftRealStorage, O: StftSpectrumStorage>(
        &self,
        plan: &StftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, I>,
    ) -> WgpuResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>> {
        let signal = leto_view1_cow(signal);
        let output_len = forward_output_len(plan, signal.len())?;
        let mut output = vec![O::from_complex64(num_complex::Complex64::new(0.0, 0.0)); output_len];
        self.execute_forward_typed_into(
            plan,
            input_precision,
            output_precision,
            &signal,
            &mut output,
        )?;
        leto_array1_from_slice(&output)
    }
}

pub(crate) fn forward_output_len(plan: &StftWgpuPlan, signal_len: usize) -> WgpuResult<usize> {
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
    if signal_len < plan.frame_len() {
        return Err(WgpuError::InputTooShort {
            min: plan.frame_len(),
            actual: signal_len,
        });
    }
    Ok((1 + signal_len.div_ceil(plan.hop_len())) * plan.frame_len())
}
