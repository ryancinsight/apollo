//! Hephaestus device acquisition and NTT execution boundary.

use std::borrow::Cow;

use crate::{DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};
use hephaestus_wgpu::WgpuDevice;

use crate::infrastructure::transport::gpu::application::plan::NttWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{
    mod_pow_u64, NttGpuBuffers, NttGpuKernel, NttMode,
};

/// Return whether a default Hephaestus WGPU device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    NttWgpuBackend::try_default().is_ok()
}

/// Hephaestus WGPU backend for exact finite-field transforms.
#[derive(Debug, Clone)]
pub struct NttWgpuBackend {
    device: WgpuDevice,
}

impl NttWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Acquire the default Hephaestus WGPU device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-ntt-wgpu")?))
    }

    /// Return the operations implemented by this backend.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::full(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only plan with the canonical modulus and root.
    #[must_use]
    pub const fn plan(&self, len: usize) -> NttWgpuPlan {
        NttWgpuPlan::new(len, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT)
    }

    /// Create a metadata-only plan with an explicit modulus and primitive root.
    #[must_use]
    pub const fn plan_with_modulus(
        &self,
        len: usize,
        modulus: u64,
        primitive_root: u64,
    ) -> NttWgpuPlan {
        NttWgpuPlan::new(len, modulus, primitive_root)
    }

    /// Construct reusable host-side state for one validated plan.
    pub fn create_buffers(&self, plan: &NttWgpuPlan) -> WgpuResult<NttGpuBuffers> {
        let omega = Self::validate_plan_and_len(plan, plan.len())?;
        NttGpuKernel::create_buffers(plan.len(), plan.modulus(), omega)
    }

    /// Execute the forward NTT over the configured residue field.
    pub fn execute_forward(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>> {
        self.execute_allocating(plan, input, NttMode::Forward)
    }

    /// Execute the forward NTT into reusable host state.
    pub fn execute_forward_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        self.execute_with_buffers(plan, input, buffers, NttMode::Forward)
    }

    /// Execute the inverse NTT over the configured residue field.
    pub fn execute_inverse(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>> {
        self.execute_allocating(plan, input, NttMode::Inverse)
    }

    /// Execute the inverse NTT into reusable host state.
    pub fn execute_inverse_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        self.execute_with_buffers(plan, input, buffers, NttMode::Inverse)
    }

    /// Execute forward from exact `u32` residues into caller-owned storage.
    pub fn execute_forward_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
    ) -> WgpuResult<()> {
        self.execute_quantized_into(plan, input, output, NttMode::Forward)
    }

    /// Execute inverse from exact `u32` residues into caller-owned storage.
    pub fn execute_inverse_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
    ) -> WgpuResult<()> {
        self.execute_quantized_into(plan, input, output, NttMode::Inverse)
    }

    /// Execute forward exact residues into reusable host state.
    pub fn execute_forward_quantized_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        self.execute_quantized_with_buffers(plan, input, buffers, NttMode::Forward)
    }

    /// Execute inverse exact residues into reusable host state.
    pub fn execute_inverse_quantized_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        self.execute_quantized_with_buffers(plan, input, buffers, NttMode::Inverse)
    }

    /// Execute a forward transform from a Leto host view.
    pub fn execute_forward_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u64>,
    ) -> WgpuResult<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>> {
        let input = leto_view1_cow(input)?;
        self.execute_forward(plan, &input)
            .and_then(|output| leto_array1_from_slice(&output))
    }

    /// Execute an inverse transform from a Leto host view.
    pub fn execute_inverse_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u64>,
    ) -> WgpuResult<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>> {
        let input = leto_view1_cow(input)?;
        self.execute_inverse(plan, &input)
            .and_then(|output| leto_array1_from_slice(&output))
    }

    /// Execute forward exact residues from a Leto host view.
    pub fn execute_forward_quantized_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u32>,
    ) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>> {
        self.execute_quantized_leto(plan, input, NttMode::Forward)
    }

    /// Execute inverse exact residues from a Leto host view.
    pub fn execute_inverse_quantized_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u32>,
    ) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>> {
        self.execute_quantized_leto(plan, input, NttMode::Inverse)
    }

    /// Return the last readback in reusable host state.
    #[must_use]
    pub fn buffer_output<'a>(&self, buffers: &'a NttGpuBuffers) -> &'a [u64] {
        buffers.output()
    }

    fn execute_allocating(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        mode: NttMode,
    ) -> WgpuResult<Vec<u64>> {
        let mut buffers = self.create_buffers(plan)?;
        self.execute_with_buffers(plan, input, &mut buffers, mode)?;
        Ok(buffers.output().to_vec())
    }

    fn execute_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &mut NttGpuBuffers,
        mode: NttMode,
    ) -> WgpuResult<()> {
        Self::validate_plan_input_and_buffers_len(plan, input.len(), buffers)?;
        NttGpuKernel::execute_with_buffers(&self.device, input, mode, buffers)
    }

    fn execute_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
        mode: NttMode,
    ) -> WgpuResult<()> {
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let mut buffers = self.create_buffers(plan)?;
        self.execute_quantized_with_buffers(plan, input, &mut buffers, mode)?;
        for (target, value) in output.iter_mut().zip(buffers.output().iter().copied()) {
            *target = value as u32;
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
        NttGpuKernel::execute_quantized_with_buffers(&self.device, input, mode, buffers)
    }

    fn execute_quantized_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u32>,
        mode: NttMode,
    ) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>> {
        let input = leto_view1_cow(input)?;
        let mut output = vec![0; plan.len()];
        self.execute_quantized_into(plan, &input, &mut output, mode)?;
        leto_array1_from_slice(&output)
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
        if modulus > u64::from(u32::MAX) || primitive_root > u64::from(u32::MAX) {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: accelerator storage requires u32 field values"),
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
        Ok(mod_pow_u64(
            primitive_root,
            (modulus - 1) / len as u64,
            modulus,
        ))
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> WgpuResult<Cow<'_, [T]>> {
    if let Some(values) = view.as_slice() {
        return Ok(Cow::Borrowed(values));
    }
    let mut values = Vec::with_capacity(view.shape()[0]);
    for index in 0..view.shape()[0] {
        values.push(
            *view
                .get([index])
                .map_err(|error| WgpuError::ShapeMismatch {
                    message: format!("invalid Leto NTT 1D view: {error:?}"),
                })?,
        );
    }
    Ok(Cow::Owned(values))
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::from_mnemosyne_slice([values.len()], values).map_err(|error| {
        WgpuError::InvalidPlan {
            message: format!("failed to allocate Mnemosyne-backed Leto NTT output: {error:?}"),
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
