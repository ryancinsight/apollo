//! WGPU device acquisition and GFT execution backend.

use std::sync::Arc;

use apollo_fft::PrecisionProfile;
use apollo_gft::GftStorage;

use crate::application::plan::GftWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::GftGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    GftWgpuBackend::try_default().is_ok()
}

/// WGPU backend for GFT execution.
#[derive(Debug, Clone)]
pub struct GftWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<GftGpuKernel>,
}

impl GftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(GftGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-gft-wgpu")?)
    }

    /// Return truthful current capabilities (forward and inverse both implemented).
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::implemented(true)
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

    /// Create a metadata plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> GftWgpuPlan {
        GftWgpuPlan::new(len)
    }

    /// Execute the forward GFT: `X[k] = sum_i U[i,k] * signal[i]`  (U^T x).
    ///
    /// Requires signal.len() == plan.len() and basis.len() == len*len.
    pub fn execute_forward(
        &self,
        plan: &GftWgpuPlan,
        signal: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        Self::validate(plan, signal, basis)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            signal,
            basis,
            plan.len(),
            0,
        )
    }

    /// Execute the inverse GFT: `x[i] = sum_k U[i,k] * spectrum[k]`  (U X).
    ///
    /// Requires spectrum.len() == plan.len() and basis.len() == len*len.
    pub fn execute_inverse(
        &self,
        plan: &GftWgpuPlan,
        spectrum: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        Self::validate(plan, spectrum, basis)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            spectrum,
            basis,
            plan.len(),
            1,
        )
    }

    /// Execute the forward GFT with typed `f64`, `f32`, or mixed `f16` storage.
    ///
    /// The graph basis matrix must always be supplied as `f32`.
    pub fn execute_forward_typed_into<T: GftStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_gft_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = signal.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_forward(plan, &represented, basis)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    /// Execute the inverse GFT with typed `f64`, `f32`, or mixed `f16` storage.
    pub fn execute_inverse_typed_into<T: GftStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_gft_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = spectrum.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_inverse(plan, &represented, basis)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    fn validate_gft_typed_precision<T: GftStorage>(precision: PrecisionProfile) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    fn validate(plan: &GftWgpuPlan, signal: &[f32], basis: &[f32]) -> WgpuResult<()> {
        let n = plan.len();
        if n == 0 {
            return Err(WgpuError::InvalidPlan {
                len: 0,
                message: "length must be greater than zero",
            });
        }
        if signal.len() != n {
            return Err(WgpuError::LengthMismatch {
                expected: n,
                actual: signal.len(),
            });
        }
        if basis.len() != n * n {
            return Err(WgpuError::BasisLengthMismatch {
                expected: n * n,
                actual: basis.len(),
            });
        }
        Ok(())
    }
}
