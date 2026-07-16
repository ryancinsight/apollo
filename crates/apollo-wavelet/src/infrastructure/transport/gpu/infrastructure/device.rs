//! Hephaestus device acquisition and multilevel Haar execution boundary.

use apollo_fft::PrecisionProfile;
use hephaestus_wgpu::WgpuDevice;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::WaveletWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::WaveletGpuKernel;
use crate::WaveletGpuStorage;

thread_local! {
    static GPU_INPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
    static GPU_OUTPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
}

/// Hephaestus WGPU backend for the orthonormal Haar DWT.
#[derive(Debug, Clone)]
pub struct WaveletWgpuBackend {
    device: WgpuDevice,
}

impl WaveletWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Acquire the default Hephaestus WGPU device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-wavelet-wgpu")?))
    }

    /// Return truthful forward/inverse capability descriptor.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::implemented(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize, levels: usize) -> WaveletWgpuPlan {
        WaveletWgpuPlan::new(len, levels)
    }

    /// Execute the forward multilevel Haar DWT in Mallat ordering.
    pub fn execute_forward(&self, plan: &WaveletWgpuPlan, signal: &[f32]) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0; plan.len()];
        self.execute_forward_into(plan, signal, &mut output)?;
        Ok(output)
    }

    /// Execute the inverse multilevel Haar DWT.
    pub fn execute_inverse(
        &self,
        plan: &WaveletWgpuPlan,
        coefficients: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0; plan.len()];
        self.execute_inverse_into(plan, coefficients, &mut output)?;
        Ok(output)
    }

    /// Execute forward Haar analysis into caller-owned storage.
    pub fn execute_forward_into(
        &self,
        plan: &WaveletWgpuPlan,
        signal: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate(plan, signal.len(), output.len())?;
        WaveletGpuKernel::execute_forward_into(&self.device, signal, output, plan.levels())
    }

    /// Execute inverse Haar synthesis into caller-owned storage.
    pub fn execute_inverse_into(
        &self,
        plan: &WaveletWgpuPlan,
        coefficients: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate(plan, coefficients.len(), output.len())?;
        WaveletGpuKernel::execute_inverse_into(&self.device, coefficients, output, plan.levels())
    }

    /// Execute forward Haar analysis from a Leto host view.
    pub fn execute_forward_leto(
        &self,
        plan: &WaveletWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let output = self.execute_forward(plan, &signal)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto Wavelet output".to_owned(),
        })
    }

    /// Execute inverse Haar synthesis from a Leto host view.
    pub fn execute_inverse_leto(
        &self,
        plan: &WaveletWgpuPlan,
        coefficients: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let coefficients = apollo_leto_interop::view_cow(&coefficients);
        let output = self.execute_inverse(plan, &coefficients)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto Wavelet output".to_owned(),
        })
    }

    /// Execute typed forward Haar analysis under the concrete `f32` GPU contract.
    pub fn execute_forward_typed_into<T: WaveletGpuStorage>(
        &self,
        plan: &WaveletWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        self.execute_typed_into(plan, precision, signal, output, false)
    }

    /// Execute typed inverse Haar synthesis under the concrete `f32` GPU contract.
    pub fn execute_inverse_typed_into<T: WaveletGpuStorage>(
        &self,
        plan: &WaveletWgpuPlan,
        precision: PrecisionProfile,
        coefficients: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        self.execute_typed_into(plan, precision, coefficients, output, true)
    }

    /// Execute typed forward Haar analysis from a Leto host view.
    pub fn execute_forward_leto_typed<T: WaveletGpuStorage + Default>(
        &self,
        plan: &WaveletWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        self.execute_typed_leto(plan, precision, signal, false)
    }

    /// Execute typed inverse Haar synthesis from a Leto host view.
    pub fn execute_inverse_leto_typed<T: WaveletGpuStorage + Default>(
        &self,
        plan: &WaveletWgpuPlan,
        precision: PrecisionProfile,
        coefficients: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        self.execute_typed_leto(plan, precision, coefficients, true)
    }

    fn execute_typed_into<T: WaveletGpuStorage>(
        &self,
        plan: &WaveletWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
        inverse: bool,
    ) -> WgpuResult<()> {
        if precision != T::PROFILE {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Self::validate(plan, input.len(), output.len())?;
        if let (Some(input), Some(output)) = (T::as_f32_slice(input), T::as_f32_slice_mut(output)) {
            return if inverse {
                self.execute_inverse_into(plan, input, output)
            } else {
                self.execute_forward_into(plan, input, output)
            };
        }
        GPU_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |represented| {
                for (target, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *target = value.to_gpu();
                }
                GPU_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |computed| {
                        if inverse {
                            self.execute_inverse_into(plan, represented, computed)?;
                        } else {
                            self.execute_forward_into(plan, represented, computed)?;
                        }
                        for (target, value) in output.iter_mut().zip(computed.iter().copied()) {
                            *target = T::from_gpu(value);
                        }
                        Ok(())
                    })
                })
            })
        })
    }

    fn execute_typed_leto<T: WaveletGpuStorage + Default>(
        &self,
        plan: &WaveletWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
        inverse: bool,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        let output_slice = output
            .as_slice_mut()
            .expect("Wavelet Mnemosyne output must be contiguous");
        self.execute_typed_into(plan, precision, &input, output_slice, inverse)?;
        Ok(output)
    }

    fn validate(plan: &WaveletWgpuPlan, input_len: usize, output_len: usize) -> WgpuResult<()> {
        let len = plan.len();
        let levels = plan.levels();
        if len == 0 || !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                message: format!("length {len} must be a non-zero power of two"),
            });
        }
        if levels == 0 || levels >= usize::BITS as usize || (1usize << levels) > len {
            return Err(WgpuError::InvalidPlan {
                message: format!("levels {levels} must satisfy 0 < levels <= log2({len})"),
            });
        }
        if input_len != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input_len,
            });
        }
        if output_len != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: output_len,
            });
        }
        Ok(())
    }
}
