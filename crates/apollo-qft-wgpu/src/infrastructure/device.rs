//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_fft::PrecisionProfile;
use apollo_qft::QftStorage;
use num_complex::{Complex32, Complex64};

use crate::application::plan::QftWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::{QftGpuKernel, QftMode};
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    QftWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct QftWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<QftGpuKernel>,
}

impl QftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(QftGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-qft-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::direct_unitary(true)
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
    pub const fn plan(&self, len: usize) -> QftWgpuPlan {
        QftWgpuPlan::new(len)
    }

    /// Execute the forward unitary QFT.
    pub fn execute_forward(
        &self,
        plan: &QftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_inputs(plan, input)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            plan.len(),
            QftMode::Forward,
        )
    }

    /// Execute the inverse unitary QFT.
    pub fn execute_inverse(
        &self,
        plan: &QftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_inputs(plan, input)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            plan.len(),
            QftMode::Inverse,
        )
    }

    /// Execute the forward unitary QFT with typed `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    pub fn execute_forward_typed_into<T: QftStorage>(
        &self,
        plan: &QftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_qft_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
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

    /// Execute the inverse unitary QFT with typed storage.
    pub fn execute_inverse_typed_into<T: QftStorage>(
        &self,
        plan: &QftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_qft_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
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
        let computed = self.execute_inverse(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_complex64(Complex64::new(f64::from(value.re), f64::from(value.im)));
        }
        Ok(())
    }

    fn validate_qft_typed_precision<T: QftStorage>(precision: PrecisionProfile) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    fn validate_inputs(plan: &QftWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan len={}: transform length must be greater than zero",
                    plan.len()
                ),
            });
        }
        if input.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: input.len(),
            });
        }
        Ok(())
    }
}
