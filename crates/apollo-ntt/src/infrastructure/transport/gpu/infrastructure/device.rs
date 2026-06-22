//! WGPU device acquisition for this transform backend.

use std::{borrow::Cow, sync::Arc};

use crate::{DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};

use crate::infrastructure::transport::gpu::application::plan::NttWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{NttGpuBuffers, NttGpuKernel, NttMode};
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    NttWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct NttWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<NttGpuKernel>,
}

impl NttWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(NttGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-ntt-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::full(true)
    }

    /// Return the acquired WGPU device.
    #[must_use]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        self.device.device()
    }

    /// Return the acquired WGPU queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        self.device.queue()
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> NttWgpuPlan {
        NttWgpuPlan::new(len, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT)
    }

    /// Create a plan with an explicit modulus and primitive root.
    #[must_use]
    pub const fn plan_with_modulus(
        &self,
        len: usize,
        modulus: u64,
        primitive_root: u64,
    ) -> NttWgpuPlan {
        NttWgpuPlan::new(len, modulus, primitive_root)
    }

    /// Allocate reusable GPU and host buffers for repeated execution of one plan length.
    pub fn create_buffers(&self, plan: &NttWgpuPlan) -> WgpuResult<NttGpuBuffers> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid buffer length {len}"),
            });
        }
        let omega = Self::validate_plan_and_len(plan, len)?;
        self.kernel
            .create_buffers(&self.device, len, plan.modulus(), omega)
    }

    /// Execute the direct forward NTT over the configured residue field.
    pub fn execute_forward(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>> {
        let omega = Self::validate_plan_and_input(plan, input)?;
        self.kernel.execute(
            &self.device,
            input,
            plan.len(),
            plan.modulus(),
            omega,
            NttMode::Forward,
        )
    }

    /// Execute the direct forward NTT from a Leto residue view.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    pub fn execute_forward_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u64>,
    ) -> WgpuResult<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_forward(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute forward NTT from exact `u32` residue storage into caller-owned `u32` output.
    ///
    /// This is quantized integer storage, not floating mixed precision. It is
    /// exact when the plan modulus is bounded by `u32::MAX`, which is already
    /// required by the current WGPU shader surface.
    pub fn execute_forward_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
    ) -> WgpuResult<()> {
        self.execute_quantized_into(plan, input, output, NttMode::Forward)
    }

    /// Execute forward NTT from exact `u32` Leto residue storage.
    pub fn execute_forward_quantized_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u32>,
    ) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>> {
        let input = leto_view1_cow(input)?;
        let mut output = vec![0_u32; plan.len()];
        self.execute_forward_quantized_into(plan, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Execute forward NTT from exact `u32` residues with caller-owned reusable buffers.
    pub fn execute_forward_quantized_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        self.execute_quantized_with_buffers(plan, input, buffers, NttMode::Forward)
    }

    /// Execute the direct forward NTT with caller-owned reusable buffers.
    pub fn execute_forward_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        Self::validate_plan_input_and_buffers(plan, input, buffers)?;
        self.kernel.execute_with_buffers(
            &self.device,
            input,
            NttMode::Forward,
            buffers,
        )
    }

    /// Execute the direct inverse NTT over the configured residue field.
    pub fn execute_inverse(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>> {
        let omega = Self::validate_plan_and_input(plan, input)?;
        self.kernel.execute(
            &self.device,
            input,
            plan.len(),
            plan.modulus(),
            omega,
            NttMode::Inverse,
        )
    }

    /// Execute the direct inverse NTT from a Leto residue view.
    pub fn execute_inverse_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u64>,
    ) -> WgpuResult<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_inverse(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute inverse NTT from exact `u32` residue storage into caller-owned `u32` output.
    pub fn execute_inverse_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
    ) -> WgpuResult<()> {
        self.execute_quantized_into(plan, input, output, NttMode::Inverse)
    }

    /// Execute inverse NTT from exact `u32` Leto residue storage.
    pub fn execute_inverse_quantized_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u32>,
    ) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>> {
        let input = leto_view1_cow(input)?;
        let mut output = vec![0_u32; plan.len()];
        self.execute_inverse_quantized_into(plan, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Execute inverse NTT from exact `u32` residues with caller-owned reusable buffers.
    pub fn execute_inverse_quantized_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        self.execute_quantized_with_buffers(plan, input, buffers, NttMode::Inverse)
    }

    /// Execute the direct inverse NTT with caller-owned reusable buffers.
    pub fn execute_inverse_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        Self::validate_plan_input_and_buffers(plan, input, buffers)?;
        self.kernel.execute_with_buffers(
            &self.device,
            input,
            NttMode::Inverse,
            buffers,
        )
    }

    /// Return the last readback values written by a reusable-buffer execution.
    #[must_use]
    pub fn buffer_output<'a>(&self, buffers: &'a NttGpuBuffers) -> &'a [u64] {
        self.kernel.buffer_output(buffers)
    }

    fn execute_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
        mode: NttMode,
    ) -> WgpuResult<()> {
        let len = plan.len();
        if output.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: output.len(),
            });
        }
        Self::validate_plan_and_len(plan, input.len())?;
        let mut buffers = self.create_buffers(plan)?;
        self.kernel.execute_quantized_with_buffers(
            &self.device,
            input,
            mode,
            &mut buffers,
        )?;
        for (slot, &value) in output.iter_mut().zip(self.buffer_output(&buffers).iter()) {
            *slot = value as u32;
        }
        Ok(())
    }

    fn execute_quantized_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        buffers: &mut NttGpuBuffers,
        mode: NttMode,
    ) -> WgpuResult<()> {
        Self::validate_plan_input_and_buffers_len(plan, input.len(), buffers)?;
        self.kernel.execute_quantized_with_buffers(
            &self.device,
            input,
            mode,
            buffers,
        )
    }

    fn validate_plan_input_and_buffers(
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &NttGpuBuffers,
    ) -> WgpuResult<()> {
        Self::validate_plan_input_and_buffers_len(plan, input.len(), buffers)
    }

    fn validate_plan_input_and_buffers_len(
        plan: &NttWgpuPlan,
        input_len: usize,
        buffers: &NttGpuBuffers,
    ) -> WgpuResult<()> {
        Self::validate_plan_and_len(plan, input_len)?;
        if buffers.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: buffers.len(),
            });
        }
        Ok(())
    }

    fn validate_plan_and_input(plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<u64> {
        Self::validate_plan_and_len(plan, input.len())
    }

    fn validate_plan_and_len(plan: &NttWgpuPlan, input_len: usize) -> WgpuResult<u64> {
        let len = plan.len();
        let modulus = plan.modulus();
        let primitive_root = plan.primitive_root();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: length must be greater than zero"),
                });
        }
        if !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: length must be a power of two"),
                });
        }
        if modulus < 2 {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: modulus must be at least 2"),
                });
        }
        if modulus > u32::MAX as u64 || primitive_root > u32::MAX as u64 {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: current WGPU NTT surface supports 32-bit modulus and primitive root"),
                });
        }
        if (modulus - 1) % len as u64 != 0 {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: transform length is not supported by the modulus"),
                });
        }
        if input_len != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input_len,
            });
        }
        let root = mod_pow_u64(primitive_root, (modulus - 1) / len as u64, modulus);
        Ok(root)
    }
}

fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1_u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp >>= 1;
    }
    result
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> WgpuResult<Cow<'_, [T]>> {
    if let Some(slice) = view.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    let len = view.shape()[0];
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        values.push(*view.get([index]).map_err(|err| WgpuError::ShapeMismatch {
            message: format!("invalid Leto NTT 1D view: {err:?}"),
        })?);
    }
    Ok(Cow::Owned(values))
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::from_mnemosyne_slice([values.len()], values).map_err(|err| {
        WgpuError::InvalidPlan {
            message: format!("failed to allocate Mnemosyne-backed Leto NTT output: {err:?}"),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use leto::SliceArg;

    use super::leto_view1_cow;

    #[test]
    fn leto_view1_cow_borrows_contiguous_views() {
        let input = leto::Array1::from_shape_vec([4], vec![1_u64, 2, 3, 4]).expect("input");
        let cow = leto_view1_cow(input.view()).expect("contiguous view");
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn leto_view1_cow_materializes_strided_views() {
        let input =
            leto::Array1::from_shape_vec([8], vec![1_u64, 99, 2, 99, 3, 99, 4, 99]).expect("input");
        let view = input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");
        let cow = leto_view1_cow(view).expect("strided view");
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }
}
