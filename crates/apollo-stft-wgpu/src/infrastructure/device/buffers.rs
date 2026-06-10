use num_complex::Complex32;

use crate::application::plan::StftWgpuPlan;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::buffers::StftGpuBuffers;
use crate::infrastructure::device::StftWgpuBackend;

impl StftWgpuBackend {
    /// Allocate pre-allocated GPU buffers for repeated STFT dispatches with the given plan.
    pub fn make_buffers(
        &self,
        plan: &StftWgpuPlan,
        signal_len: usize,
    ) -> WgpuResult<StftGpuBuffers> {
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
        let frame_count = 1 + signal_len.div_ceil(plan.hop_len());
        Ok(StftGpuBuffers::new(
            self.device.device(),
            &self.kernel,
            frame_count,
            plan.frame_len(),
            signal_len,
            plan.hop_len(),
        ))
    }

    /// Execute the forward STFT using pre-allocated GPU buffers.
    pub fn execute_forward_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        signal: &[f32],
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
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
        if !plan.frame_len().is_power_of_two() {
            let result = self.execute_forward(plan, signal)?;
            buffers.fwd_output_host.copy_from_slice(&result);
            return Ok(());
        }
        self.kernel.execute_forward_fft_with_buffers(
            self.device.device(),
            self.device.queue(),
            signal,
            buffers,
        )
    }

    /// Execute the inverse STFT using pre-allocated GPU buffers.
    pub fn execute_inverse_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        spectrum: &[Complex32],
        signal_len: usize,
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
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
        if !plan.frame_len().is_power_of_two() {
            let result = self.execute_inverse(plan, spectrum, signal_len)?;
            buffers.inv_output_host.copy_from_slice(&result);
            return Ok(());
        }
        self.kernel.execute_inverse_with_buffers(
            self.device.device(),
            self.device.queue(),
            spectrum,
            signal_len,
            buffers,
        )
    }
}
