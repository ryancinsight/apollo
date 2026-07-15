//! WGPU device acquisition for this transform backend.

use crate::application::execution::plan::dht::HartleyGpuStorage;
use apollo_fft::PrecisionProfile;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::DhtWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::DhtGpuKernel;
use hephaestus_wgpu::WgpuDevice;

thread_local! {
    static GPU_INPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
    static GPU_OUTPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
}

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    DhtWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct DhtWgpuBackend {
    device: WgpuDevice,
}

impl DhtWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-dht-wgpu")?))
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
    pub const fn plan(&self, len: usize) -> DhtWgpuPlan {
        DhtWgpuPlan::new(len)
    }

    /// Execute the unnormalized forward 1D DHT for a real-valued `f32` signal.
    pub fn execute_forward(&self, plan: &DhtWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_forward_into(plan, input, &mut output)?;
        Ok(output)
    }

    /// Execute the unnormalized forward DHT into caller-owned contiguous storage.
    pub fn execute_forward_into(
        &self,
        plan: &DhtWgpuPlan,
        input: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_plan_input(plan, input)?;
        Self::validate_output(plan, output)?;
        DhtGpuKernel::execute_into(&self.device, input, output, false)
    }

    /// Execute the unnormalized forward 1D DHT from a Leto `f32` view.
    ///
    /// Contiguous Leto views borrow host storage directly; strided views copy
    /// once into logical order before dispatching to the existing WGPU slice path.
    pub fn execute_forward_leto(
        &self,
        plan: &DhtWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_forward_into(
            plan,
            &input,
            output
                .as_slice_mut()
                .expect("DHT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the unnormalized forward 1D DHT with caller-owned typed storage.
    ///
    /// WGPU arithmetic remains `f32`; mixed `f16` storage is promoted once to
    /// represented `f32` before dispatch and quantized at the output boundary.
    pub fn execute_forward_typed_into<T: HartleyGpuStorage>(
        &self,
        plan: &DhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        self.execute_typed_into(input, output, false)
    }

    /// Execute the unnormalized forward 1D DHT from typed Leto storage.
    ///
    /// Precision-profile validation and host quantization match
    /// [`Self::execute_forward_typed_into`].
    pub fn execute_forward_leto_typed<T: HartleyGpuStorage + Default>(
        &self,
        plan: &DhtWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_forward_typed_into(
            plan,
            precision,
            &input,
            output
                .as_slice_mut()
                .expect("DHT typed Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the normalized inverse 1D DHT for a real-valued `f32` spectrum.
    pub fn execute_inverse(&self, plan: &DhtWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_inverse_into(plan, input, &mut output)?;
        Ok(output)
    }

    /// Execute the normalized inverse DHT into caller-owned contiguous storage.
    pub fn execute_inverse_into(
        &self,
        plan: &DhtWgpuPlan,
        input: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_plan_input(plan, input)?;
        Self::validate_output(plan, output)?;
        DhtGpuKernel::execute_into(&self.device, input, output, true)
    }

    /// Execute the normalized inverse 1D DHT from a Leto `f32` view.
    ///
    /// Output storage is Mnemosyne-backed Leto host memory.
    pub fn execute_inverse_leto(
        &self,
        plan: &DhtWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_inverse_into(
            plan,
            &input,
            output
                .as_slice_mut()
                .expect("DHT inverse Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the normalized inverse 1D DHT with caller-owned typed storage.
    pub fn execute_inverse_typed_into<T: HartleyGpuStorage>(
        &self,
        plan: &DhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        self.execute_typed_into(input, output, true)
    }

    /// Execute the normalized inverse 1D DHT from typed Leto storage.
    pub fn execute_inverse_leto_typed<T: HartleyGpuStorage + Default>(
        &self,
        plan: &DhtWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_inverse_typed_into(
            plan,
            precision,
            &input,
            output
                .as_slice_mut()
                .expect("DHT typed inverse Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    fn validate_plan_input(plan: &DhtWgpuPlan, input: &[f32]) -> WgpuResult<()> {
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

    fn validate_typed_plan_input<T: HartleyGpuStorage>(
        plan: &DhtWgpuPlan,
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
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        Ok(())
    }

    fn validate_output(plan: &DhtWgpuPlan, output: &[f32]) -> WgpuResult<()> {
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        Ok(())
    }

    fn execute_typed_into<T: HartleyGpuStorage>(
        &self,
        input: &[T],
        output: &mut [T],
        inverse: bool,
    ) -> WgpuResult<()> {
        if let (Some(input), Some(output)) = (T::as_f32_slice(input), T::as_f32_slice_mut(output)) {
            return DhtGpuKernel::execute_into(&self.device, input, output, inverse);
        }
        GPU_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |represented| {
                for (slot, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *slot = value.to_gpu();
                }
                GPU_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |computed| {
                        DhtGpuKernel::execute_into(&self.device, represented, computed, inverse)?;
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
