use eunomia::Complex32;

use crate::infrastructure::transport::gpu::application::plan::StftWgpuPlan;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::buffers::StftGpuBuffers;
use crate::infrastructure::transport::gpu::infrastructure::{
    device::StftWgpuBackend, kernel::StftGpuKernel,
};

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
        if plan.hop_len() > plan.frame_len() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: hop_len must not exceed frame_len",
                    plan.frame_len(),
                    plan.hop_len()
                ),
            });
        }
        let frame_count = 1 + signal_len.div_ceil(plan.hop_len());
        StftGpuBuffers::new(
            &self.device,
            frame_count,
            plan.frame_len(),
            signal_len,
            plan.hop_len(),
        )
    }

    /// Execute the forward STFT using pre-allocated GPU buffers.
    pub fn execute_forward_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        signal: &[f32],
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
        let expected_output = super::forward::forward_output_len(plan, signal.len())?;
        if plan.frame_len() != buffers.frame_len()
            || plan.hop_len() != buffers.hop_len()
            || signal.len() != buffers.signal_len()
            || expected_output != buffers.forward_host.len()
        {
            return Err(WgpuError::InvalidPlan {
                message: "buffer geometry does not match the forward STFT plan and signal"
                    .to_owned(),
            });
        }
        if !plan.frame_len().is_power_of_two() {
            let result = self.execute_forward(plan, signal)?;
            for (dest, src) in buffers.forward_host.iter_mut().zip(result.iter()) {
                *dest = crate::infrastructure::transport::gpu::infrastructure::kernel::ComplexPod {
                    re: src.re,
                    im: src.im,
                };
            }
            return Ok(());
        }
        StftGpuKernel::execute_forward_fft_with_buffers(&self.device, signal, buffers)
    }

    /// Execute the inverse STFT using pre-allocated GPU buffers.
    pub fn execute_inverse_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        spectrum: &[Complex32],
        signal_len: usize,
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
        if plan.frame_len() == 0 || plan.hop_len() == 0 || plan.hop_len() > plan.frame_len() {
            return Err(WgpuError::InvalidPlan {
                message: "buffered inverse requires a non-empty valid STFT plan".to_owned(),
            });
        }
        let frame_count = 1 + signal_len.div_ceil(plan.hop_len());
        let expected =
            frame_count
                .checked_mul(plan.frame_len())
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: "frame_count * frame_len overflows host address space".to_owned(),
                })?;
        if plan.frame_len() != buffers.frame_len()
            || plan.hop_len() != buffers.hop_len()
            || signal_len != buffers.signal_len()
            || frame_count != buffers.frame_count()
            || spectrum.len() != expected
        {
            return Err(WgpuError::InvalidPlan {
                message: "buffer geometry does not match the inverse STFT plan and spectrum"
                    .to_owned(),
            });
        }
        if !plan.frame_len().is_power_of_two() {
            let result = self.execute_inverse(plan, spectrum, signal_len)?;
            buffers.inverse_host.copy_from_slice(&result);
            return Ok(());
        }
        StftGpuKernel::execute_inverse_with_buffers(&self.device, spectrum, signal_len, buffers)
    }
}
