//! WGPU device acquisition for this transform backend.

use std::borrow::Cow;

use crate::FwhtStorage;
use apollo_fft::PrecisionProfile;

use crate::infrastructure::transport::gpu::application::plan::FwhtWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::FwhtGpuKernel;
use hephaestus_wgpu::WgpuDevice;

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct FwhtWgpuBackend {
    device: WgpuDevice,
}

impl FwhtWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-fwht-wgpu")?))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::implemented(true)
    }

    /// Return the acquired Hephaestus device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> FwhtWgpuPlan {
        FwhtWgpuPlan::new(len)
    }

    /// Execute the unnormalized forward 1D FWHT for a real-valued `f32` signal.
    pub fn execute_forward(&self, plan: &FwhtWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        FwhtGpuKernel::execute(&self.device, input, false)
    }

    /// Execute the unnormalized forward 1D FWHT from a Leto `f32` view.
    ///
    /// Contiguous Leto views borrow host storage directly; strided views copy
    /// once into logical order before dispatching to the existing WGPU slice path.
    pub fn execute_forward_leto(
        &self,
        plan: &FwhtWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let output = self.execute_forward(plan, &input)?;
        apollo_leto_interop::try_array1_from_vec(output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to construct Mnemosyne-backed Leto output".to_owned(),
        })
    }

    /// Execute the unnormalized forward 1D FWHT with caller-owned typed storage.
    ///
    /// WGPU arithmetic remains `f32`; mixed `f16` storage is promoted once to
    /// represented `f32` before dispatch and quantized at the output boundary.
    pub fn execute_forward_typed_into<T: FwhtStorage>(
        &self,
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        let represented = typed_to_f32(input);
        let computed = self.execute_forward(plan, &represented)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    /// Execute the unnormalized forward 1D FWHT from typed Leto storage.
    ///
    /// Precision-profile validation and host quantization match
    /// [`Self::execute_forward_typed_into`].
    pub fn execute_forward_leto_typed<T: FwhtStorage>(
        &self,
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_forward_typed_into(plan, precision, &input, &mut output)?;
        apollo_leto_interop::try_array1_from_vec(output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to construct Mnemosyne-backed Leto output".to_owned(),
        })
    }

    /// Execute the normalized inverse 1D FWHT for a real-valued `f32` spectrum.
    pub fn execute_inverse(&self, plan: &FwhtWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        FwhtGpuKernel::execute(&self.device, input, true)
    }

    /// Execute the normalized inverse 1D FWHT from a Leto `f32` view.
    ///
    /// Output storage is Mnemosyne-backed Leto host memory.
    pub fn execute_inverse_leto(
        &self,
        plan: &FwhtWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let output = self.execute_inverse(plan, &input)?;
        apollo_leto_interop::try_array1_from_vec(output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to construct Mnemosyne-backed Leto output".to_owned(),
        })
    }

    /// Execute the normalized inverse 1D FWHT with caller-owned typed storage.
    pub fn execute_inverse_typed_into<T: FwhtStorage>(
        &self,
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        let represented = typed_to_f32(input);
        let computed = self.execute_inverse(plan, &represented)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    /// Execute the normalized inverse 1D FWHT from typed Leto storage.
    pub fn execute_inverse_leto_typed<T: FwhtStorage>(
        &self,
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_inverse_typed_into(plan, precision, &input, &mut output)?;
        apollo_leto_interop::try_array1_from_vec(output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to construct Mnemosyne-backed Leto output".to_owned(),
        })
    }

    fn validate_plan_input(plan: &FwhtWgpuPlan, input: &[f32]) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be a power of two"),
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

    fn validate_typed_plan_input<T: FwhtStorage>(
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &[T],
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be a power of two"),
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
}

fn typed_to_f32<T: FwhtStorage>(input: &[T]) -> Cow<'_, [f32]> {
    if let Some(slice_f32) = T::as_f32_slice(input) {
        Cow::Borrowed(slice_f32)
    } else {
        let vec: Vec<f32> = input.iter().map(|value| value.to_f64() as f32).collect();
        Cow::Owned(vec)
    }
}

fn write_typed_output<T: FwhtStorage>(source: &[f32], output: &mut [T]) {
    if let Some(slice_f32) = T::as_f32_slice_mut(output) {
        slice_f32.copy_from_slice(source);
    } else {
        for (slot, value) in output.iter_mut().zip(source.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
    }
}
