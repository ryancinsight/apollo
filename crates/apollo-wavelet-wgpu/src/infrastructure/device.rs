//! WGPU device acquisition and backend orchestration for the Haar DWT.

use std::sync::Arc;

use apollo_fft::PrecisionProfile;
use apollo_wavelet::WaveletStorage;

use crate::application::plan::WaveletWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::WaveletGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    WaveletWgpuBackend::try_default().is_ok()
}

/// WGPU backend for the Haar DWT.
///
/// Owns an acquired device/queue pair and a cached kernel pipeline.
#[derive(Debug, Clone)]
pub struct WaveletWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<WaveletGpuKernel>,
}

impl WaveletWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(WaveletGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-wavelet-wgpu")?)
    }

    /// Return truthful forward+inverse capability descriptor.
    #[must_use]
    pub fn capabilities(&self) -> WgpuCapabilities {
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

    /// Create a plan descriptor for the given signal length and decomposition levels.
    #[must_use]
    pub const fn plan(&self, len: usize, levels: usize) -> WaveletWgpuPlan {
        WaveletWgpuPlan::new(len, levels)
    }

    /// Execute the forward multi-level Haar DWT on .
    ///
    /// Returns a flat coefficient buffer in Mallat ordering:
    /// .
    ///
    /// Validation:  must be a non-zero power of two,  must be
    /// non-zero, , and .
    pub fn execute_forward(&self, plan: &WaveletWgpuPlan, signal: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan(plan)?;
        if signal.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: signal.len(),
            });
        }
        self.kernel.execute_forward(
            self.device.device(),
            self.device.queue(),
            signal,
            plan.len(),
            plan.levels(),
        )
    }

    /// Execute the inverse multi-level Haar DWT on .
    ///
    /// Expects input in Mallat ordering (output of ).
    /// Returns the reconstructed signal of length .
    ///
    /// Validation mirrors .
    pub fn execute_inverse(
        &self,
        plan: &WaveletWgpuPlan,
        coefficients: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        Self::validate_plan(plan)?;
        if coefficients.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: coefficients.len(),
            });
        }
        self.kernel.execute_inverse(
            self.device.device(),
            self.device.queue(),
            coefficients,
            plan.len(),
            plan.levels(),
        )
    }

    /// Execute the forward Haar DWT with typed `f64`, `f32`, or mixed `f16` storage.
    ///
    /// Promotes represented input once to `f32`, dispatches the GPU forward kernel,
    /// and quantizes output back to the requested storage type.
    pub fn execute_forward_typed_into<T: WaveletStorage>(
        &self,
        plan: &WaveletWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = signal.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_forward(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    /// Execute the inverse Haar DWT with typed `f64`, `f32`, or mixed `f16` storage.
    pub fn execute_inverse_typed_into<T: WaveletStorage>(
        &self,
        plan: &WaveletWgpuPlan,
        precision: PrecisionProfile,
        coefficients: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = coefficients.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_inverse(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    fn validate_typed_precision<T: WaveletStorage>(precision: PrecisionProfile) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    /// Validate plan parameters before GPU dispatch.
    ///
    /// Invariants:
    /// -  and  is a power of two (Haar requires dyadic length).
    /// -  (at least one decomposition pass).
    /// -  (each level halves the approximation subband).
    fn validate_plan(plan: &WaveletWgpuPlan) -> WgpuResult<()> {
        let len = plan.len();
        let levels = plan.levels();
        if len == 0 || !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid length {len}, levels {levels}: len must be a non-zero power of two"
                ),
            });
        }
        if levels == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}, levels {levels}: levels must be non-zero"),
            });
        }
        if (1usize << levels) > len {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid length {len}, levels {levels}: 2^levels must not exceed len"
                ),
            });
        }
        Ok(())
    }
}
