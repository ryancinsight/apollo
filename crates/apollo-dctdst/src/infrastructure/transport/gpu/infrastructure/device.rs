//! WGPU device acquisition for this transform backend.

/// Forward execution implementations.
pub mod forward;
/// Inverse execution implementations.
pub mod inverse;

use crate::{RealTransformGpuStorage, RealTransformKind};
use apollo_fft::PrecisionProfile;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::DctDstWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{DctGpuKernel, DctMode};
use hephaestus_wgpu::WgpuDevice;

thread_local! {
    pub(crate) static GPU_INPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
    pub(crate) static GPU_OUTPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct DctDstWgpuBackend {
    pub(crate) device: WgpuDevice,
}

impl DctDstWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-dctdst-wgpu")?))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::full(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize, kind: RealTransformKind) -> DctDstWgpuPlan {
        DctDstWgpuPlan::new(len, kind)
    }

    pub(crate) fn validate_dct_typed_precision<T: RealTransformGpuStorage>(
        precision: PrecisionProfile,
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    pub(crate) fn validate_plan_input(plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_output(plan: &DctDstWgpuPlan, output: &[f32]) -> WgpuResult<()> {
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn cubic_element_count(len: usize, rank: u32) -> WgpuResult<usize> {
        len.checked_pow(rank).ok_or_else(|| WgpuError::InvalidPlan {
            message: format!("{rank}D element count overflows usize for length {len}"),
        })
    }

    pub(crate) fn validate_typed_plan_input<T: RealTransformGpuStorage>(
        plan: &DctDstWgpuPlan,
        input: &[T],
        output: &[T],
    ) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        if output.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: output.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn execute_typed_into<T: RealTransformGpuStorage>(
        &self,
        input: &[T],
        output: &mut [T],
        mode: DctMode,
        scale: f32,
    ) -> WgpuResult<()> {
        if let (Some(input), Some(output)) = (T::as_f32_slice(input), T::as_f32_slice_mut(output)) {
            return DctGpuKernel::execute_into(&self.device, input, output, mode, scale);
        }
        GPU_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |represented| {
                for (slot, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *slot = value.to_gpu();
                }
                GPU_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |computed| {
                        DctGpuKernel::execute_into(
                            &self.device,
                            represented,
                            computed,
                            mode,
                            scale,
                        )?;
                        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
                            *slot = T::from_gpu(value);
                        }
                        Ok(())
                    })
                })
            })
        })
    }
}
