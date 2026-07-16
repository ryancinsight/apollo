//! WGPU device acquisition for this transform backend.

use crate::CztStorage;
use apollo_fft::PrecisionProfile;
use eunomia::Complex32;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::CztWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::CztGpuKernel;
use hephaestus_wgpu::WgpuDevice;

thread_local! {
    static GPU_INPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
    static GPU_OUTPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct CztWgpuBackend {
    device: WgpuDevice,
}

impl CztWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_inverse(true)
    }

    /// Return the acquired Hephaestus device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub fn plan(
        &self,
        input_len: usize,
        output_len: usize,
        a: Complex32,
        w: Complex32,
    ) -> CztWgpuPlan {
        CztWgpuPlan::new(
            input_len,
            output_len,
            [a.re.to_bits(), a.im.to_bits()],
            [w.re.to_bits(), w.im.to_bits()],
        )
    }

    /// Execute the direct forward CZT for a complex-valued `f32` signal.
    pub fn execute_forward(
        &self,
        plan: &CztWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        let mut output = vec![Complex32::new(0.0, 0.0); plan.output_len()];
        self.execute_forward_into(plan, input, &mut output)?;
        Ok(output)
    }

    /// Execute the direct forward CZT into caller-owned contiguous storage.
    pub fn execute_forward_into(
        &self,
        plan: &CztWgpuPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        Self::validate_plan_input(plan, input)?;
        if output.len() != plan.output_len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.output_len(),
                actual: output.len(),
            });
        }
        CztGpuKernel::execute_forward_into(&self.device, plan, input, output)
    }

    /// Execute the direct forward CZT from a Leto host view.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    pub fn execute_forward_leto(
        &self,
        plan: &CztWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([
                plan.output_len()
            ]);
        self.execute_forward_into(
            plan,
            &input,
            output
                .as_slice_mut()
                .expect("CZT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the forward CZT with typed `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    ///
    /// Promotes represented input once to `Complex32`, dispatches the GPU kernel,
    /// and quantizes the output back to the requested storage type.
    pub fn execute_forward_typed_into<T: CztStorage>(
        &self,
        plan: &CztWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_czt_typed_precision::<T>(precision)?;
        let output_len = plan.output_len();
        if output.len() != output_len {
            return Err(WgpuError::LengthMismatch {
                expected: output_len,
                actual: output.len(),
            });
        }
        if let (Some(input), Some(output)) = (T::as_c32_slice(input), T::as_c32_slice_mut(output)) {
            return self.execute_forward_into(plan, input, output);
        }
        GPU_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |represented| {
                for (slot, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *slot = value.to_complex32();
                }
                GPU_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |computed| {
                        self.execute_forward_into(plan, represented, computed)?;
                        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
                            *slot = T::from_complex32(value);
                        }
                        Ok(())
                    })
                })
            })
        })
    }

    /// Execute typed forward CZT from a Leto host view into Mnemosyne-backed Leto storage.
    pub fn execute_forward_leto_typed<T: CztStorage + Default>(
        &self,
        plan: &CztWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.output_len()]);
        self.execute_forward_typed_into(
            plan,
            precision,
            &input,
            output
                .as_slice_mut()
                .expect("CZT typed Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    fn validate_czt_typed_precision<T: CztStorage>(precision: PrecisionProfile) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    /// Execute the adjoint inverse CZT.
    ///
    /// Computes `x[n] = (A^n / N) · Σ_k X[k] · W^{-nk}` on the GPU.
    ///
    /// **Exactness**: exact when |A| = 1, |W| = 1, and W is an N-th root of
    /// unity (DFT case).  For general spiral parameters this is the
    /// minimum-norm adjoint solution; the CPU crate's `CztPlan::inverse` uses
    /// the Björck–Pereyra Vandermonde solve for exact general inversion.
    ///
    /// # Errors
    ///
    /// Returns [`WgpuError::LengthMismatch`] when the plan is not square.
    pub fn execute_inverse(
        &self,
        plan: &CztWgpuPlan,
        spectrum: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        let mut output = vec![Complex32::new(0.0, 0.0); plan.input_len()];
        self.execute_inverse_into(plan, spectrum, &mut output)?;
        Ok(output)
    }

    /// Execute the square-plan adjoint inverse CZT into caller-owned storage.
    pub fn execute_inverse_into(
        &self,
        plan: &CztWgpuPlan,
        spectrum: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        if plan.input_len() != plan.output_len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.input_len(),
                actual: plan.output_len(),
            });
        }
        if spectrum.len() != plan.output_len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.output_len(),
                actual: spectrum.len(),
            });
        }
        let n = plan.input_len();
        if n == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("CZT lengths input={n}, output={n} must be greater than zero"),
            });
        }
        if output.len() != n {
            return Err(WgpuError::LengthMismatch {
                expected: n,
                actual: output.len(),
            });
        }
        CztGpuKernel::execute_inverse_into(&self.device, plan, spectrum, output)
    }

    /// Execute the adjoint inverse CZT from a Leto host view.
    ///
    /// This preserves the existing WGPU inverse contract: exact for DFT
    /// parameters and adjoint/minimum-norm for general CZT spirals.
    pub fn execute_inverse_leto(
        &self,
        plan: &CztWgpuPlan,
        spectrum: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        let mut output =
            leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([
                plan.input_len()
            ]);
        self.execute_inverse_into(
            plan,
            &spectrum,
            output
                .as_slice_mut()
                .expect("CZT inverse Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    fn validate_plan_input(plan: &CztWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        let input_len = plan.input_len();
        let output_len = plan.output_len();
        if input_len == 0 || output_len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "CZT lengths input={input_len}, output={output_len} must be greater than zero"
                ),
            });
        }
        if input.len() != input_len {
            return Err(WgpuError::LengthMismatch {
                expected: input_len,
                actual: input.len(),
            });
        }
        let a = plan.a();
        let w = plan.w();
        let a_norm = a.norm();
        let w_norm = w.norm();
        if !a_norm.is_finite() || !w_norm.is_finite() || a_norm == 0.0 || w_norm == 0.0 {
            return Err(WgpuError::InvalidPlan {
                message: "CZT spiral parameters must have finite non-zero magnitude".to_owned(),
            });
        }
        Ok(())
    }
}
