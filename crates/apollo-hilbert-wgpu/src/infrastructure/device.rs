//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use num_complex::Complex32;

use apollo_fft::PrecisionProfile;
use apollo_hilbert::HilbertStorage;

use crate::application::plan::HilbertWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::HilbertGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    HilbertWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct HilbertWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<HilbertGpuKernel>,
}

impl HilbertWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> WgpuResult<Self> {
        Ok(Self {
            kernel: Arc::new(HilbertGpuKernel::new(device.as_ref())),
            device,
            queue,
        })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|error| WgpuError::AdapterUnavailable {
            message: error.to_string(),
        })?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("apollo-hilbert-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor)).map_err(|error| {
                WgpuError::DeviceUnavailable {
                    message: error.to_string(),
                }
            })?;
        Self::new(Arc::new(device), Arc::new(queue))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_and_inverse(true)
    }

    /// Return the acquired WGPU device.
    #[must_use]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Return the acquired WGPU queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> HilbertWgpuPlan {
        HilbertWgpuPlan::new(len)
    }

    /// Execute the analytic signal `x + i H{x}` for a real-valued `f32` signal.
    pub fn execute_analytic_signal(
        &self,
        plan: &HilbertWgpuPlan,
        input: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_plan_input(plan, input)?;
        self.kernel
            .execute(self.device.as_ref(), self.queue.as_ref(), input)
    }

    /// Execute the forward Hilbert quadrature component `H{x}` for a real-valued `f32` signal.
    pub fn execute_forward(&self, plan: &HilbertWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Ok(self
            .execute_analytic_signal(plan, input)?
            .into_iter()
            .map(|value| value.im)
            .collect())
    }

    /// Execute the forward Hilbert quadrature transform with typed `f64`, `f32`, or mixed `f16` storage.
    ///
    /// Promotes represented input once to `f32`, dispatches the GPU analytic-signal kernel,
    /// extracts the imaginary (quadrature) component, and quantizes output back to storage type.
    pub fn execute_forward_typed_into<T: HilbertStorage>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_hilbert_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_forward(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    fn validate_hilbert_typed_precision<T: HilbertStorage>(
        precision: PrecisionProfile,
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    /// Execute the inverse Hilbert transform: recover the original real signal from its quadrature component.
    ///
    /// By the Hilbert inversion theorem H(H(x)) = -I, the original signal is
    /// `x[n] = -H{H{x}[n]} = -H{quadrature[n]}`.
    pub fn execute_inverse(
        &self,
        plan: &HilbertWgpuPlan,
        quadrature: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, quadrature)?;
        self.kernel
            .execute_inverse(self.device.as_ref(), self.queue.as_ref(), quadrature)
    }

    /// Execute the inverse Hilbert transform with typed storage.
    pub fn execute_inverse_typed_into<T: HilbertStorage>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        quadrature: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_hilbert_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = quadrature.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_inverse(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    fn validate_plan_input(plan: &HilbertWgpuPlan, input: &[f32]) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidLength {
                len,
                message: "length must be greater than zero",
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
}
