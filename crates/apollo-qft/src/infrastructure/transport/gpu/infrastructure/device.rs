//! Hephaestus device acquisition and unitary QFT execution boundary.

use apollo_fft::PrecisionProfile;
use eunomia::Complex32;
use hephaestus_wgpu::WgpuDevice;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::QftWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{QftGpuKernel, QftMode};
use crate::QftGpuStorage;

thread_local! {
    static GPU_INPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
    static GPU_OUTPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
}

/// Return whether a default Hephaestus WGPU device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    QftWgpuBackend::try_default().is_ok()
}

/// Hephaestus WGPU backend for direct unitary QFT execution.
#[derive(Debug, Clone)]
pub struct QftWgpuBackend {
    device: WgpuDevice,
}

impl QftWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Acquire the default Hephaestus WGPU device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-qft-wgpu")?))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::direct_unitary(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> QftWgpuPlan {
        QftWgpuPlan::new(len)
    }

    /// Execute the forward unitary QFT.
    pub fn execute_forward(
        &self,
        plan: &QftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        self.execute_allocating(plan, input, QftMode::Forward)
    }

    /// Execute the inverse unitary QFT.
    pub fn execute_inverse(
        &self,
        plan: &QftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        self.execute_allocating(plan, input, QftMode::Inverse)
    }

    /// Execute the forward unitary QFT into caller-owned storage.
    pub fn execute_forward_into(
        &self,
        plan: &QftWgpuPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        self.execute_into(plan, input, output, QftMode::Forward)
    }

    /// Execute the inverse unitary QFT into caller-owned storage.
    pub fn execute_inverse_into(
        &self,
        plan: &QftWgpuPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        self.execute_into(plan, input, output, QftMode::Inverse)
    }

    /// Execute forward from a Leto host view.
    pub fn execute_forward_leto(
        &self,
        plan: &QftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_forward(plan, &input).and_then(|output| {
            apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
                WgpuError::InvalidPlan {
                    message: "failed to allocate Mnemosyne-backed Leto QFT output".to_owned(),
                }
            })
        })
    }

    /// Execute inverse from a Leto host view.
    pub fn execute_inverse_leto(
        &self,
        plan: &QftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_inverse(plan, &input).and_then(|output| {
            apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
                WgpuError::InvalidPlan {
                    message: "failed to allocate Mnemosyne-backed Leto QFT output".to_owned(),
                }
            })
        })
    }

    /// Execute forward with storage admitted by the concrete `f32` GPU contract.
    pub fn execute_forward_typed_into<T: QftGpuStorage>(
        &self,
        plan: &QftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        self.execute_typed_into(plan, precision, input, output, QftMode::Forward)
    }

    /// Execute inverse with storage admitted by the concrete `f32` GPU contract.
    pub fn execute_inverse_typed_into<T: QftGpuStorage>(
        &self,
        plan: &QftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        self.execute_typed_into(plan, precision, input, output, QftMode::Inverse)
    }

    /// Execute typed forward QFT from a Leto host view.
    pub fn execute_forward_leto_typed<T: QftGpuStorage + Default>(
        &self,
        plan: &QftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        self.execute_typed_leto(plan, precision, input, QftMode::Forward)
    }

    /// Execute typed inverse QFT from a Leto host view.
    pub fn execute_inverse_leto_typed<T: QftGpuStorage + Default>(
        &self,
        plan: &QftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        self.execute_typed_leto(plan, precision, input, QftMode::Inverse)
    }

    fn execute_allocating(
        &self,
        plan: &QftWgpuPlan,
        input: &[Complex32],
        mode: QftMode,
    ) -> WgpuResult<Vec<Complex32>> {
        let mut output = vec![Complex32::new(0.0, 0.0); plan.len()];
        self.execute_into(plan, input, &mut output, mode)?;
        Ok(output)
    }

    fn execute_into(
        &self,
        plan: &QftWgpuPlan,
        input: &[Complex32],
        output: &mut [Complex32],
        mode: QftMode,
    ) -> WgpuResult<()> {
        Self::validate_input(plan, input)?;
        Self::validate_output(plan, output)?;
        QftGpuKernel::execute_into(&self.device, input, output, mode)
    }

    fn execute_typed_into<T: QftGpuStorage>(
        &self,
        plan: &QftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
        mode: QftMode,
    ) -> WgpuResult<()> {
        Self::validate_typed(plan, precision, input, output)?;
        if let (Some(input), Some(output)) = (T::as_c32_slice(input), T::as_c32_slice_mut(output)) {
            return self.execute_into(plan, input, output, mode);
        }
        GPU_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |represented| {
                for (target, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *target = value.to_gpu();
                }
                GPU_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |computed| {
                        self.execute_into(plan, represented, computed, mode)?;
                        for (target, value) in output.iter_mut().zip(computed.iter().copied()) {
                            *target = T::from_gpu(value);
                        }
                        Ok(())
                    })
                })
            })
        })
    }

    fn execute_typed_leto<T: QftGpuStorage + Default>(
        &self,
        plan: &QftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
        mode: QftMode,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        let output_slice = output
            .as_slice_mut()
            .expect("QFT Mnemosyne output must be contiguous");
        self.execute_typed_into(plan, precision, &input, output_slice, mode)?;
        Ok(output)
    }

    fn validate_typed<T: QftGpuStorage>(
        plan: &QftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &[T],
    ) -> WgpuResult<()> {
        if precision != T::PROFILE {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Self::validate_input_len(plan, input.len())?;
        Self::validate_output(plan, output)
    }

    fn validate_input(plan: &QftWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        Self::validate_input_len(plan, input.len())
    }

    fn validate_input_len(plan: &QftWgpuPlan, length: usize) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "transform length must be greater than zero".to_owned(),
            });
        }
        if length != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: length,
            });
        }
        Ok(())
    }

    fn validate_output<T>(plan: &QftWgpuPlan, output: &[T]) -> WgpuResult<()> {
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        Ok(())
    }
}
