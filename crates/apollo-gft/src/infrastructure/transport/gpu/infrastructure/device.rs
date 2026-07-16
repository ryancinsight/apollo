//! Hephaestus device acquisition and GFT execution boundary.

use apollo_fft::PrecisionProfile;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::GftWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{GftDirection, GftGpuKernel};
use crate::GftGpuStorage;
use hephaestus_wgpu::WgpuDevice;

thread_local! {
    static GPU_INPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
    static GPU_OUTPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
}

/// Hephaestus WGPU backend for graph Fourier execution.
#[derive(Debug, Clone)]
pub struct GftWgpuBackend {
    device: WgpuDevice,
}

impl GftWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Create a backend by requesting a default Hephaestus WGPU device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-gft-wgpu")?))
    }

    /// Return truthful current capabilities (forward and inverse are implemented).
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::implemented(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only graph-order descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> GftWgpuPlan {
        GftWgpuPlan::new(len)
    }

    /// Execute the forward GFT `X = U^T x`.
    pub fn execute_forward(
        &self,
        plan: &GftWgpuPlan,
        signal: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_forward_into(plan, signal, basis, &mut output)?;
        Ok(output)
    }

    /// Execute the forward GFT `X = U^T x` into caller-owned storage.
    pub fn execute_forward_into(
        &self,
        plan: &GftWgpuPlan,
        signal: &[f32],
        basis: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_plan_input(plan, signal, basis)?;
        Self::validate_output(plan, output)?;
        GftGpuKernel::execute_into(&self.device, signal, basis, output, GftDirection::Forward)
    }

    /// Execute the forward GFT from Leto host views.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    pub fn execute_forward_leto(
        &self,
        plan: &GftWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let basis = apollo_leto_interop::view_cow(&basis);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_forward_into(
            plan,
            &signal,
            &basis,
            output
                .as_slice_mut()
                .expect("GFT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the inverse GFT `x = U X`.
    pub fn execute_inverse(
        &self,
        plan: &GftWgpuPlan,
        spectrum: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_inverse_into(plan, spectrum, basis, &mut output)?;
        Ok(output)
    }

    /// Execute the inverse GFT `x = U X` into caller-owned storage.
    pub fn execute_inverse_into(
        &self,
        plan: &GftWgpuPlan,
        spectrum: &[f32],
        basis: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_plan_input(plan, spectrum, basis)?;
        Self::validate_output(plan, output)?;
        GftGpuKernel::execute_into(&self.device, spectrum, basis, output, GftDirection::Inverse)
    }

    /// Execute the inverse GFT from Leto host views.
    pub fn execute_inverse_leto(
        &self,
        plan: &GftWgpuPlan,
        spectrum: leto::ArrayView1<'_, f32>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        let basis = apollo_leto_interop::view_cow(&basis);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_inverse_into(
            plan,
            &spectrum,
            &basis,
            output
                .as_slice_mut()
                .expect("GFT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the forward GFT with storage admitted by the `f32` accelerator contract.
    pub fn execute_forward_typed_into<T: GftGpuStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_precision::<T>(precision)?;
        Self::validate_typed_plan_input(plan, signal, basis, output)?;
        self.execute_typed_into(signal, basis, output, GftDirection::Forward)
    }

    /// Execute typed forward GFT from Leto host views.
    pub fn execute_forward_leto_typed<T: GftGpuStorage + Default>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, T>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let basis = apollo_leto_interop::view_cow(&basis);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_forward_typed_into(
            plan,
            precision,
            &signal,
            &basis,
            output
                .as_slice_mut()
                .expect("typed GFT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the inverse GFT with storage admitted by the `f32` accelerator contract.
    pub fn execute_inverse_typed_into<T: GftGpuStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_precision::<T>(precision)?;
        Self::validate_typed_plan_input(plan, spectrum, basis, output)?;
        self.execute_typed_into(spectrum, basis, output, GftDirection::Inverse)
    }

    /// Execute typed inverse GFT from Leto host views.
    pub fn execute_inverse_leto_typed<T: GftGpuStorage + Default>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: leto::ArrayView1<'_, T>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        let basis = apollo_leto_interop::view_cow(&basis);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_inverse_typed_into(
            plan,
            precision,
            &spectrum,
            &basis,
            output
                .as_slice_mut()
                .expect("typed GFT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    fn execute_typed_into<T: GftGpuStorage>(
        &self,
        input: &[T],
        basis: &[f32],
        output: &mut [T],
        direction: GftDirection,
    ) -> WgpuResult<()> {
        if let (Some(input), Some(output)) = (T::as_f32_slice(input), T::as_f32_slice_mut(output)) {
            return GftGpuKernel::execute_into(&self.device, input, basis, output, direction);
        }
        GPU_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |represented| {
                for (slot, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *slot = value.to_gpu();
                }
                GPU_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |computed| {
                        GftGpuKernel::execute_into(
                            &self.device,
                            represented,
                            basis,
                            computed,
                            direction,
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

    fn validate_typed_precision<T: GftGpuStorage>(precision: PrecisionProfile) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    fn validate_plan_input(plan: &GftWgpuPlan, input: &[f32], basis: &[f32]) -> WgpuResult<()> {
        let n = plan.len();
        if n == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "length must be greater than zero".to_owned(),
            });
        }
        if input.len() != n {
            return Err(WgpuError::LengthMismatch {
                expected: n,
                actual: input.len(),
            });
        }
        Self::validate_basis_len(n, basis.len())
    }

    fn validate_typed_plan_input<T: GftGpuStorage>(
        plan: &GftWgpuPlan,
        input: &[T],
        basis: &[f32],
        output: &[T],
    ) -> WgpuResult<()> {
        let n = plan.len();
        if n == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "length must be greater than zero".to_owned(),
            });
        }
        if input.len() != n {
            return Err(WgpuError::LengthMismatch {
                expected: n,
                actual: input.len(),
            });
        }
        Self::validate_output(plan, output)?;
        Self::validate_basis_len(n, basis.len())
    }

    fn validate_output<T>(plan: &GftWgpuPlan, output: &[T]) -> WgpuResult<()> {
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        Ok(())
    }

    fn validate_basis_len(n: usize, actual: usize) -> WgpuResult<()> {
        let expected = n.checked_mul(n).ok_or_else(|| WgpuError::InvalidPlan {
            message: format!("basis element count overflows usize for graph order {n}"),
        })?;
        if actual != expected {
            return Err(WgpuError::ShapeMismatch {
                message: format!("expected {expected} elements for a {n}x{n} basis, got {actual}"),
            });
        }
        Ok(())
    }
}
