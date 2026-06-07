//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_fft::PrecisionProfile;
use apollo_mellin::MellinStorage;
use num_complex::Complex32;

use crate::application::plan::MellinWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::MellinGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    MellinWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct MellinWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<MellinGpuKernel>,
}

impl MellinWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(MellinGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-mellin-wgpu")?)
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
    pub fn plan(&self, samples: usize, min_scale: f64, max_scale: f64) -> MellinWgpuPlan {
        MellinWgpuPlan::new(samples, min_scale.to_bits(), max_scale.to_bits())
    }

    /// Execute the forward Mellin log-frequency spectrum for a real-valued `f32` signal.
    pub fn execute_forward(
        &self,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_inputs(plan, signal, signal_min, signal_max)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            signal,
            signal_min,
            signal_max,
        )
    }

    /// Inverse or adjoint execution is unsupported until the owning Mellin crate defines it.
    pub fn execute_inverse(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f64,
        out_max: f64,
        out_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        if plan.samples() == 0 || out_len == 0 {
            return Err(WgpuError::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if spectrum.len() != plan.samples() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.samples(),
                actual: spectrum.len(),
            });
        }
        if !out_min.is_finite() || !out_max.is_finite() || out_min <= 0.0 || out_max <= 0.0 {
            return Err(WgpuError::InvalidSignalDomain {
                signal_min: out_min,
                signal_max: out_max,
                message: "output domain bounds must be finite and positive",
            });
        }
        if out_min >= out_max {
            return Err(WgpuError::InvalidSignalDomain {
                signal_min: out_min,
                signal_max: out_max,
                message: "out_min must be less than out_max",
            });
        }
        self.kernel.execute_inverse(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            spectrum,
            out_min,
            out_max,
            out_len,
        )
    }

    /// Execute the forward Mellin transform with typed `f64`, `f32`, or mixed `f16` input storage.
    ///
    /// Promotes represented input once to `f32` before dispatch.
    /// Returns the log-frequency spectrum as `Vec<Complex32>`.
    pub fn execute_forward_typed<T: MellinStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<Vec<Complex32>> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let represented: Vec<f32> = signal.iter().map(|v| v.to_f64() as f32).collect();
        self.execute_forward(plan, &represented, signal_min, signal_max)
    }

    fn validate_inputs(
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<()> {
        if plan.samples() == 0 {
            return Err(WgpuError::InvalidPlan {
                samples: plan.samples(),
                min_scale: plan.min_scale(),
                max_scale: plan.max_scale(),
                message: "sample count must be greater than zero",
            });
        }
        if !plan.min_scale().is_finite()
            || !plan.max_scale().is_finite()
            || plan.min_scale() <= 0.0
            || plan.max_scale() <= 0.0
        {
            return Err(WgpuError::InvalidPlan {
                samples: plan.samples(),
                min_scale: plan.min_scale(),
                max_scale: plan.max_scale(),
                message: "plan scales must be finite and positive",
            });
        }
        if plan.min_scale() >= plan.max_scale() {
            return Err(WgpuError::InvalidPlan {
                samples: plan.samples(),
                min_scale: plan.min_scale(),
                max_scale: plan.max_scale(),
                message: "min_scale must be less than max_scale",
            });
        }
        if signal.is_empty() {
            return Err(WgpuError::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if !signal_min.is_finite()
            || !signal_max.is_finite()
            || signal_min <= 0.0
            || signal_max <= 0.0
        {
            return Err(WgpuError::InvalidSignalDomain {
                signal_min,
                signal_max,
                message: "signal bounds must be finite and positive",
            });
        }
        if signal_min >= signal_max {
            return Err(WgpuError::InvalidSignalDomain {
                signal_min,
                signal_max,
                message: "signal_min must be less than signal_max",
            });
        }
        Ok(())
    }
}
