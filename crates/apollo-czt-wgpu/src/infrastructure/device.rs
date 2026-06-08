//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_czt::CztStorage;
use apollo_fft::PrecisionProfile;
use num_complex::{Complex32, Complex64};

use crate::application::plan::CztWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::CztGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    CztWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct CztWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<CztGpuKernel>,
}

impl CztWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(CztGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-czt-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_inverse(true)
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
        Self::validate_plan_input(plan, input)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            input,
        )
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
        let represented: Vec<Complex32> = input
            .iter()
            .map(|v| {
                let c = v.to_complex64();
                Complex32::new(c.re as f32, c.im as f32)
            })
            .collect();
        let computed = self.execute_forward(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_complex64(Complex64::new(f64::from(value.re), f64::from(value.im)));
        }
        Ok(())
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
    /// Returns `WgpuError::NotSquare` when `plan.input_len() != plan.output_len()`.
    pub fn execute_inverse(
        &self,
        plan: &CztWgpuPlan,
        spectrum: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
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
        self.kernel.execute_inverse(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            spectrum,
        )
    }

    fn validate_plan_input(plan: &CztWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        let input_len = plan.input_len();
        let output_len = plan.output_len();
        if input_len == 0 || output_len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("CZT lengths input={input_len}, output={output_len} must be greater than zero"),
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
