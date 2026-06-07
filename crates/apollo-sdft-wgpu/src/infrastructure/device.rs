//! WGPU device acquisition for the SDFT transform backend.

use std::sync::Arc;

use num_complex::{Complex32, Complex64};

use apollo_fft::PrecisionProfile;
use apollo_sdft::{SdftBinStorage, SdftRealStorage};

use crate::application::plan::SdftWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::SdftGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    SdftWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct SdftWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<SdftGpuKernel>,
}

impl SdftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(SdftGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-sdft-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_and_inverse(true)
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
    pub const fn plan(&self, window_len: usize, bin_count: usize) -> SdftWgpuPlan {
        SdftWgpuPlan::new(window_len, bin_count)
    }

    /// Execute the direct SDFT bins computation for a real-valued window.
    ///
    /// Returns `Vec<Complex32>` with `plan.bin_count()` complex outputs.
    pub fn execute_forward(
        &self,
        plan: &SdftWgpuPlan,
        window: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_plan_window(plan, window)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            window,
            plan.window_len(),
            plan.bin_count(),
        )
    }

    /// Execute the direct SDFT bins computation with typed real input and typed complex output.
    ///
    /// `input_precision` must match `I::PROFILE`; `output_precision` must match `O::PROFILE`.
    /// WGPU arithmetic is `f32`; host storage is promoted/quantized at the dispatch boundary.
    pub fn execute_forward_typed_into<I: SdftRealStorage, O: SdftBinStorage>(
        &self,
        plan: &SdftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        window: &[I],
        output: &mut [O],
    ) -> WgpuResult<()> {
        let expected_in = I::PROFILE;
        if input_precision.storage != expected_in.storage
            || input_precision.compute != expected_in.compute
        {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let expected_out = O::PROFILE;
        if output_precision.storage != expected_out.storage
            || output_precision.compute != expected_out.compute
        {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        if output.len() != plan.bin_count() {
            return Err(WgpuError::WindowLengthMismatch {
                expected: plan.bin_count(),
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = window.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_forward(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = O::from_complex64(Complex64::new(f64::from(value.re), f64::from(value.im)));
        }
        Ok(())
    }

    /// Execute the inverse SDFT: reconstruct a real signal from K complex DFT bins.
    ///
    /// Given `plan.bin_count()` complex bins, computes the N-point inverse DFT
    /// and returns `plan.window_len()` real samples.
    ///
    /// Mathematical contract: `x[n] = (1/K) Σ_{b=0}^{K-1} X[b]·exp(+2πi·b·n/K)`.
    pub fn execute_inverse(&self, plan: &SdftWgpuPlan, bins: &[Complex32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_bins(plan, bins)?;
        self.kernel.execute_inverse(
            self.device.inner(),
            self.device.queue().as_ref(),
            bins,
            plan.bin_count(),
            plan.window_len(),
        )
    }

    /// Execute the inverse SDFT with typed complex bin input and typed real output.
    ///
    /// Accepts `Complex32` bins directly (the GPU kernel operates at f32 precision).
    /// Writes real output by converting each computed f32 value to f64 and delegating
    /// to `O::from_f64` if available, or by encoding the real value as a complex number
    /// with zero imaginary part via `O::from_complex64` (requires `SdftBinStorage` bound).
    pub fn execute_inverse_typed_into(
        &self,
        plan: &SdftWgpuPlan,
        output_precision: PrecisionProfile,
        bins: &[Complex32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        let expected_precision = PrecisionProfile::LOW_PRECISION_F32;
        if output_precision.storage != expected_precision.storage
            || output_precision.compute != expected_precision.compute
        {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        if output.len() != plan.window_len() {
            return Err(WgpuError::WindowLengthMismatch {
                expected: plan.window_len(),
                actual: output.len(),
            });
        }
        let computed = self.execute_inverse(plan, bins)?;
        output.copy_from_slice(&computed);
        Ok(())
    }

    fn validate_plan_window(plan: &SdftWgpuPlan, window: &[f32]) -> WgpuResult<()> {
        if plan.window_len() == 0 || plan.bin_count() == 0 {
            return Err(WgpuError::InvalidPlan {
                window_len: plan.window_len(),
                bin_count: plan.bin_count(),
                message: "window_len and bin_count must each be greater than zero",
            });
        }
        if plan.bin_count() > plan.window_len() {
            return Err(WgpuError::InvalidPlan {
                window_len: plan.window_len(),
                bin_count: plan.bin_count(),
                message: "bin_count must not exceed window_len",
            });
        }
        if window.len() != plan.window_len() {
            return Err(WgpuError::WindowLengthMismatch {
                expected: plan.window_len(),
                actual: window.len(),
            });
        }
        Ok(())
    }

    fn validate_plan_bins(plan: &SdftWgpuPlan, bins: &[Complex32]) -> WgpuResult<()> {
        if plan.window_len() == 0 || plan.bin_count() == 0 {
            return Err(WgpuError::InvalidPlan {
                window_len: plan.window_len(),
                bin_count: plan.bin_count(),
                message: "window_len and bin_count must each be greater than zero",
            });
        }
        if bins.len() != plan.bin_count() {
            return Err(WgpuError::WindowLengthMismatch {
                expected: plan.bin_count(),
                actual: bins.len(),
            });
        }
        Ok(())
    }
}
