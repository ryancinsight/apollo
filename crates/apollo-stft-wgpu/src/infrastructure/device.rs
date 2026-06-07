//! WGPU device acquisition and backend orchestration for the STFT.

use std::sync::Arc;

use num_complex::{Complex32, Complex64};

use apollo_fft::PrecisionProfile;
use apollo_stft::{StftRealOutputStorage, StftRealStorage, StftSpectrumInput, StftSpectrumStorage};

use crate::application::plan::StftWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::buffers::StftGpuBuffers;
use crate::infrastructure::kernel::StftGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    StftWgpuBackend::try_default().is_ok()
}

/// WGPU backend for the STFT.
///
/// Owns an acquired device/queue pair and a cached kernel pipeline.
#[derive(Debug, Clone)]
pub struct StftWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<StftGpuKernel>,
}

impl StftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(StftGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-stft-wgpu")?)
    }

    /// Return truthful forward-and-inverse capability descriptor.
    #[must_use]
    pub fn capabilities(&self) -> WgpuCapabilities {
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

    /// Execute the forward STFT on `signal` using the supplied plan.
    ///
    /// Returns `frame_count * frame_len` complex values in row-major order:
    /// element `m * frame_len + k` is X[frame_m=m, bin_k=k].
    ///
    /// Frame count: `1 + signal.len().div_ceil(hop_len)` when signal is long enough.
    pub fn execute_forward(
        &self,
        plan: &StftWgpuPlan,
        signal: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
        if plan.frame_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "frame_len must be non-zero",
            });
        }
        if plan.hop_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "hop_len must be non-zero",
            });
        }
        if plan.hop_len() > plan.frame_len() {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "hop_len must not exceed frame_len",
            });
        }
        if signal.len() < plan.frame_len() {
            return Err(WgpuError::InputTooShort {
                min: plan.frame_len(),
                actual: signal.len(),
            });
        }
        // Mirror the CPU StftPlan::frame_count formula exactly:
        //   1 + signal_len.div_ceil(hop_len)
        let frame_count = 1 + signal.len().div_ceil(plan.hop_len());
        self.kernel.execute_forward_fft(
            self.device.device(),
            self.device.queue(),
            signal,
            plan.frame_len(),
            plan.hop_len(),
            frame_count,
        )
    }

    /// Execute the inverse STFT (WOLA reconstruction) on the GPU.
    ///
    /// `frame_count` is derived as `1 + signal_len.div_ceil(hop_len)` and
    /// `spectrum.len()` must equal `frame_count * frame_len`.
    pub fn execute_inverse(
        &self,
        plan: &StftWgpuPlan,
        spectrum: &[Complex32],
        signal_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        if plan.frame_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "frame_len must be non-zero",
            });
        }
        if plan.hop_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "hop_len must be non-zero",
            });
        }
        if plan.hop_len() > plan.frame_len() {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "hop_len must not exceed frame_len",
            });
        }
        if signal_len == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "signal_len must be non-zero",
            });
        }
        let frame_count = 1 + signal_len.div_ceil(plan.hop_len());
        let expected = frame_count * plan.frame_len();
        if spectrum.len() != expected {
            return Err(WgpuError::LengthMismatch {
                expected,
                actual: spectrum.len(),
            });
        }
        self.kernel.execute_inverse(
            self.device.device(),
            self.device.queue(),
            spectrum,
            plan.frame_len(),
            plan.hop_len(),
            frame_count,
            signal_len,
        )
    }

    /// Execute the inverse STFT with typed complex spectrum input and typed real output.
    ///
    /// `input_precision` must match `I::PROFILE`; `output_precision` must match `O::PROFILE`.
    /// WGPU arithmetic is `f32`; host storage is promoted / quantized at the dispatch boundary.
    /// `output.len()` must equal `signal_len`.
    pub fn execute_inverse_typed_into<I: StftSpectrumInput, O: StftRealOutputStorage>(
        &self,
        plan: &StftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        spectrum: &[I],
        signal_len: usize,
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
        if output.len() != signal_len {
            return Err(WgpuError::LengthMismatch {
                expected: signal_len,
                actual: output.len(),
            });
        }
        // Promote typed spectrum to f32 complex before GPU dispatch.
        let promoted: Vec<Complex32> = spectrum
            .iter()
            .map(|v| {
                let c = v.to_complex64();
                Complex32::new(c.re as f32, c.im as f32)
            })
            .collect();
        let computed = self.execute_inverse(plan, &promoted, signal_len)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = O::from_f64(f64::from(value));
        }
        Ok(())
    }

    /// Execute the forward STFT with typed real input and typed complex spectrum output.
    ///
    /// `input_precision` must match `I::PROFILE`; `output_precision` must match `O::PROFILE`.
    /// WGPU arithmetic is `f32`; host storage is promoted/quantized at the dispatch boundary.
    /// Output length must equal `(1 + signal.len().div_ceil(hop_len)) * frame_len`.
    pub fn execute_forward_typed_into<I: StftRealStorage, O: StftSpectrumStorage>(
        &self,
        plan: &StftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        signal: &[I],
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
        let represented: Vec<f32> = signal.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_forward(plan, &represented)?;
        if output.len() != computed.len() {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "output length does not match computed frame count * frame_len",
            });
        }
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = O::from_complex64(Complex64::new(f64::from(value.re), f64::from(value.im)));
        }
        Ok(())
    }

    /// Allocate pre-allocated GPU buffers for repeated STFT dispatches with the given plan.
    ///
    /// `signal_len` must be the exact length of signals that will be passed to
    /// `execute_forward_with_buffers`. `frame_count` is derived as
    /// `1 + signal_len.div_ceil(hop_len)`.
    ///
    /// Closure XIX: `frame_len` may be arbitrary (power-of-two or non-power-of-two);
    /// scratch buffers are automatically sized via `chirp_padded_len` when non-PoT.
    ///
    /// ## Errors
    /// Returns `WgpuError::InvalidPlan` if plan parameters are invalid.
    pub fn make_buffers(
        &self,
        plan: &StftWgpuPlan,
        signal_len: usize,
    ) -> WgpuResult<StftGpuBuffers> {
        if plan.frame_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "frame_len must be non-zero",
            });
        }
        if plan.hop_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "hop_len must be non-zero",
            });
        }
        let frame_count = 1 + signal_len.div_ceil(plan.hop_len());
        Ok(StftGpuBuffers::new(
            self.device.device(),
            &self.kernel,
            frame_count,
            plan.frame_len(),
            signal_len,
            plan.hop_len(),
        ))
    }

    /// Execute the forward STFT using pre-allocated GPU buffers.
    ///
    /// Reuses all GPU resources from `buffers`; uploads only `signal` data.
    /// `signal.len()` must equal `buffers.signal_len()`.
    /// Result is written into `buffers` and accessible via `buffers.fwd_output()`.
    pub fn execute_forward_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        signal: &[f32],
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
        if plan.frame_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "frame_len must be non-zero",
            });
        }
        if plan.hop_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "hop_len must be non-zero",
            });
        }
        // Non-power-of-two frame_len: delegate to allocating Chirp-Z path.
        if !plan.frame_len().is_power_of_two() {
            let result = self.execute_forward(plan, signal)?;
            buffers.fwd_output_host.copy_from_slice(&result);
            return Ok(());
        }
        self.kernel.execute_forward_fft_with_buffers(
            self.device.device(),
            self.device.queue(),
            signal,
            buffers,
        )
    }

    /// Execute the inverse STFT using pre-allocated GPU buffers.
    ///
    /// Reuses GPU resources from buffers.
    /// Pre-allocated GPU resources in `buffers` are reused to avoid per-call allocation.
    /// Non-power-of-two `frame_len` plans delegate to the allocating Chirp-Z path and
    /// copy the result into `buffers.inv_output_host`.
    ///
    /// # Errors
    /// Returns `WgpuError::InvalidPlan` if `frame_len == 0` or `hop_len == 0`.
    pub fn execute_inverse_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        spectrum: &[Complex32],
        signal_len: usize,
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
        if plan.frame_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "frame_len must be non-zero",
            });
        }
        if plan.hop_len() == 0 {
            return Err(WgpuError::InvalidPlan {
                frame_len: plan.frame_len(),
                hop_len: plan.hop_len(),
                message: "hop_len must be non-zero",
            });
        }
        if !plan.frame_len().is_power_of_two() {
            let result = self.execute_inverse(plan, spectrum, signal_len)?;
            buffers.inv_output_host.copy_from_slice(&result);
            return Ok(());
        }
        self.kernel.execute_inverse_with_buffers(
            self.device.device(),
            self.device.queue(),
            spectrum,
            signal_len,
            buffers,
        )
    }
}
