//! Hephaestus device acquisition and Hilbert execution boundary.

use apollo_fft::PrecisionProfile;
use eunomia::Complex32;
use hephaestus_wgpu::WgpuDevice;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::HilbertWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::HilbertGpuKernel;
use crate::HilbertGpuStorage;

thread_local! {
    static COMPLEX_INPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
    static COMPLEX_OUTPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
    static REAL_INPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
    static REAL_OUTPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
}

/// Hephaestus WGPU backend for analytic and inverse Hilbert execution.
#[derive(Debug, Clone)]
pub struct HilbertWgpuBackend {
    device: WgpuDevice,
}

impl HilbertWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_and_inverse(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> HilbertWgpuPlan {
        HilbertWgpuPlan::new(len)
    }

    /// Execute the analytic signal `x + i H{x}`.
    pub fn execute_analytic_signal(
        &self,
        plan: &HilbertWgpuPlan,
        input: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
        let mut output = vec![Complex32::new(0.0, 0.0); plan.len()];
        self.execute_analytic_signal_into(plan, input, &mut output)?;
        Ok(output)
    }

    /// Execute the analytic signal into caller-owned complex output storage.
    pub fn execute_analytic_signal_into(
        &self,
        plan: &HilbertWgpuPlan,
        input: &[f32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        Self::validate_lengths(plan, input.len(), output.len())?;
        COMPLEX_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |complex_input| {
                for (target, value) in complex_input.iter_mut().zip(input.iter().copied()) {
                    *target = Complex32::new(value, 0.0);
                }
                HilbertGpuKernel::execute_analytic_into(&self.device, complex_input, output)?;
                for (sample, original) in output.iter_mut().zip(input.iter().copied()) {
                    sample.re = original;
                }
                Ok(())
            })
        })
    }

    /// Execute the analytic signal from a Leto real-valued host view.
    pub fn execute_analytic_signal_leto(
        &self,
        plan: &HilbertWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([
                plan.len()
            ]);
        let output_slice = output
            .as_slice_mut()
            .expect("Hilbert Mnemosyne output must be contiguous");
        self.execute_analytic_signal_into(plan, &input, output_slice)?;
        Ok(output)
    }

    /// Execute the forward Hilbert quadrature component `H{x}`.
    pub fn execute_forward(&self, plan: &HilbertWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0; plan.len()];
        self.execute_forward_into(plan, input, &mut output)?;
        Ok(output)
    }

    /// Execute the forward Hilbert quadrature into caller-owned storage.
    pub fn execute_forward_into(
        &self,
        plan: &HilbertWgpuPlan,
        input: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_lengths(plan, input.len(), output.len())?;
        COMPLEX_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |complex_input| {
                for (target, value) in complex_input.iter_mut().zip(input.iter().copied()) {
                    *target = Complex32::new(value, 0.0);
                }
                COMPLEX_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |analytic| {
                        HilbertGpuKernel::execute_analytic_into(
                            &self.device,
                            complex_input,
                            analytic,
                        )?;
                        for (target, value) in output.iter_mut().zip(analytic.iter()) {
                            *target = value.im;
                        }
                        Ok(())
                    })
                })
            })
        })
    }

    /// Execute the forward Hilbert quadrature from a Leto host view.
    pub fn execute_forward_leto(
        &self,
        plan: &HilbertWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        let output_slice = output
            .as_slice_mut()
            .expect("Hilbert Mnemosyne output must be contiguous");
        self.execute_forward_into(plan, &input, output_slice)?;
        Ok(output)
    }

    /// Execute forward Hilbert quadrature with storage admitted by the concrete GPU contract.
    pub fn execute_forward_typed_into<T: HilbertGpuStorage>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        self.execute_typed_into(plan, precision, input, output, false)
    }

    /// Execute typed forward Hilbert quadrature from a Leto host view.
    pub fn execute_forward_leto_typed<T: HilbertGpuStorage + Default>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        self.execute_typed_leto(plan, precision, input, false)
    }

    /// Execute the inverse Hilbert transform.
    pub fn execute_inverse(
        &self,
        plan: &HilbertWgpuPlan,
        quadrature: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0; plan.len()];
        self.execute_inverse_into(plan, quadrature, &mut output)?;
        Ok(output)
    }

    /// Execute the inverse Hilbert transform into caller-owned storage.
    pub fn execute_inverse_into(
        &self,
        plan: &HilbertWgpuPlan,
        quadrature: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_lengths(plan, quadrature.len(), output.len())?;
        COMPLEX_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(quadrature.len(), |complex_input| {
                for (target, value) in complex_input.iter_mut().zip(quadrature.iter().copied()) {
                    *target = Complex32::new(value, 0.0);
                }
                COMPLEX_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |recovered| {
                        HilbertGpuKernel::execute_inverse_into(
                            &self.device,
                            complex_input,
                            recovered,
                        )?;
                        for (target, value) in output.iter_mut().zip(recovered.iter()) {
                            *target = value.re;
                        }
                        Ok(())
                    })
                })
            })
        })
    }

    /// Execute the inverse Hilbert transform from a Leto quadrature view.
    pub fn execute_inverse_leto(
        &self,
        plan: &HilbertWgpuPlan,
        quadrature: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let quadrature = apollo_leto_interop::view_cow(&quadrature);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        let output_slice = output
            .as_slice_mut()
            .expect("Hilbert Mnemosyne output must be contiguous");
        self.execute_inverse_into(plan, &quadrature, output_slice)?;
        Ok(output)
    }

    /// Execute inverse Hilbert quadrature with storage admitted by the concrete GPU contract.
    pub fn execute_inverse_typed_into<T: HilbertGpuStorage>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        quadrature: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        self.execute_typed_into(plan, precision, quadrature, output, true)
    }

    /// Execute typed inverse Hilbert quadrature from a Leto host view.
    pub fn execute_inverse_leto_typed<T: HilbertGpuStorage + Default>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        quadrature: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        self.execute_typed_leto(plan, precision, quadrature, true)
    }

    fn execute_typed_into<T: HilbertGpuStorage>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
        inverse: bool,
    ) -> WgpuResult<()> {
        Self::validate_typed(plan, precision, input, output)?;
        if let (Some(input), Some(output)) = (T::as_f32_slice(input), T::as_f32_slice_mut(output)) {
            return if inverse {
                self.execute_inverse_into(plan, input, output)
            } else {
                self.execute_forward_into(plan, input, output)
            };
        }
        REAL_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |represented| {
                for (target, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *target = value.to_gpu();
                }
                REAL_OUTPUT_SCRATCH.with(|output_pool| {
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

    fn execute_typed_leto<T: HilbertGpuStorage + Default>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
        inverse: bool,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        let output_slice = output
            .as_slice_mut()
            .expect("Hilbert Mnemosyne output must be contiguous");
        self.execute_typed_into(plan, precision, &input, output_slice, inverse)?;
        Ok(output)
    }

    fn validate_typed<T: HilbertGpuStorage>(
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &[T],
    ) -> WgpuResult<()> {
        if precision != T::PROFILE {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Self::validate_lengths(plan, input.len(), output.len())
    }

    fn validate_lengths(
        plan: &HilbertWgpuPlan,
        input_len: usize,
        output_len: usize,
    ) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "transform length must be greater than zero".to_owned(),
            });
        }
        if input_len != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: input_len,
            });
        }
        if output_len != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output_len,
            });
        }
        Ok(())
    }
}
