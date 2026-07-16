//! Hephaestus device acquisition and SDFT execution boundary.
//!
//! Apollo validates SDFT contracts and owns host representation decisions.
//! Hephaestus owns concrete device allocation, kernel preparation, binding,
//! dispatch, submission, synchronization, and transfer.

use apollo_fft::PrecisionProfile;
use eunomia::Complex32;
use hephaestus_wgpu::WgpuDevice;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::SdftWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{
    SdftForwardGpuKernel, SdftInverseGpuKernel,
};
use crate::{SdftGpuBinStorage, SdftGpuRealStorage};

thread_local! {
    static REAL_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
    static COMPLEX_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
}

/// Hephaestus WGPU backend for direct SDFT bin execution.
#[derive(Debug, Clone)]
pub struct SdftWgpuBackend {
    device: WgpuDevice,
}

impl SdftWgpuBackend {
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
    pub const fn plan(&self, window_len: usize, bin_count: usize) -> SdftWgpuPlan {
        SdftWgpuPlan::new(window_len, bin_count)
    }

    /// Execute direct SDFT bin initialization for a real-valued window.
    pub fn execute_forward(
        &self,
        plan: &SdftWgpuPlan,
        window: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_forward(plan, window.len(), plan.bin_count())?;
        let mut output = vec![Complex32::new(0.0, 0.0); plan.bin_count()];
        SdftForwardGpuKernel::execute_into(&self.device, window, &mut output)?;
        Ok(output)
    }

    /// Execute direct SDFT bin initialization from a Leto real-window view.
    ///
    /// Contiguous views borrow host storage directly. Strided views copy once
    /// into logical order. Generated bins occupy Mnemosyne-backed Leto storage.
    pub fn execute_forward_leto(
        &self,
        plan: &SdftWgpuPlan,
        window: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let window = apollo_leto_interop::view_cow(&window);
        Self::validate_forward(plan, window.len(), plan.bin_count())?;
        let mut output = complex_output(plan.bin_count());
        let output_slice = output
            .as_slice_mut()
            .expect("invariant: Mnemosyne SDFT output is contiguous");
        SdftForwardGpuKernel::execute_into(&self.device, &window, output_slice)?;
        Ok(output)
    }

    /// Execute direct SDFT bins with storage admitted by the concrete GPU contract.
    ///
    /// The GPU accepts only `f32` real input and `Complex32` bin storage, with
    /// explicit `f16` conversion. CPU `f64`/`Complex64` storage is excluded by
    /// the sealed storage traits and therefore cannot narrow implicitly.
    pub fn execute_forward_typed_into<I: SdftGpuRealStorage, O: SdftGpuBinStorage>(
        &self,
        plan: &SdftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        window: &[I],
        output: &mut [O],
    ) -> WgpuResult<()> {
        if input_precision != I::PROFILE || output_precision != O::PROFILE {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Self::validate_forward(plan, window.len(), output.len())?;
        if let Some(input) = I::as_gpu_slice(window) {
            return self.execute_forward_with_input(input, output);
        }
        REAL_SCRATCH.with(|pool| {
            pool.with_scratch(window.len(), |represented| {
                for (target, value) in represented.iter_mut().zip(window.iter().copied()) {
                    *target = value.to_gpu();
                }
                self.execute_forward_with_input(represented, output)
            })
        })
    }

    /// Execute direct SDFT bins from typed Leto real storage.
    pub fn execute_forward_leto_typed<I: SdftGpuRealStorage, O: SdftGpuBinStorage + Default>(
        &self,
        plan: &SdftWgpuPlan,
        input_precision: PrecisionProfile,
        output_precision: PrecisionProfile,
        window: leto::ArrayView1<'_, I>,
    ) -> WgpuResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>> {
        let window = apollo_leto_interop::view_cow(&window);
        let mut output =
            leto::Array::<O, leto::MnemosyneStorage<O>, 1>::zeros_mnemosyne([plan.bin_count()]);
        let output_slice = output
            .as_slice_mut()
            .expect("invariant: Mnemosyne SDFT typed output is contiguous");
        self.execute_forward_typed_into(
            plan,
            input_precision,
            output_precision,
            &window,
            output_slice,
        )?;
        Ok(output)
    }

    /// Execute complete-bin inverse SDFT into a real-valued host vector.
    ///
    /// The inverse is defined only for `bin_count == window_len`; a partial
    /// spectrum is a projection, not an inverse of the original SDFT window.
    pub fn execute_inverse(&self, plan: &SdftWgpuPlan, bins: &[Complex32]) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0; plan.window_len()];
        self.execute_inverse_into(plan, bins, &mut output)?;
        Ok(output)
    }

    /// Execute complete-bin inverse SDFT into caller-owned real storage.
    pub fn execute_inverse_into(
        &self,
        plan: &SdftWgpuPlan,
        bins: &[Complex32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_inverse(plan, bins.len(), output.len())?;
        COMPLEX_SCRATCH.with(|pool| {
            pool.with_scratch(output.len(), |computed| {
                SdftInverseGpuKernel::execute_into(&self.device, bins, computed)?;
                for (target, value) in output.iter_mut().zip(computed.iter().copied()) {
                    *target = value.re;
                }
                Ok(())
            })
        })
    }

    /// Execute complete-bin inverse SDFT from a Leto complex-bin view.
    pub fn execute_inverse_leto(
        &self,
        plan: &SdftWgpuPlan,
        bins: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let bins = apollo_leto_interop::view_cow(&bins);
        let mut output = real_output(plan.window_len());
        let output_slice = output
            .as_slice_mut()
            .expect("invariant: Mnemosyne SDFT inverse output is contiguous");
        self.execute_inverse_into(plan, &bins, output_slice)?;
        Ok(output)
    }

    /// Execute complete-bin inverse SDFT with the concrete `f32` output profile.
    pub fn execute_inverse_typed_into(
        &self,
        plan: &SdftWgpuPlan,
        output_precision: PrecisionProfile,
        bins: &[Complex32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        if output_precision != PrecisionProfile::LOW_PRECISION_F32 {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        self.execute_inverse_into(plan, bins, output)
    }

    fn execute_forward_with_input<O: SdftGpuBinStorage>(
        &self,
        input: &[f32],
        output: &mut [O],
    ) -> WgpuResult<()> {
        if let Some(output) = O::as_gpu_slice_mut(output) {
            return SdftForwardGpuKernel::execute_into(&self.device, input, output);
        }
        COMPLEX_SCRATCH.with(|pool| {
            pool.with_scratch(output.len(), |computed| {
                SdftForwardGpuKernel::execute_into(&self.device, input, computed)?;
                for (target, value) in output.iter_mut().zip(computed.iter().copied()) {
                    *target = O::from_gpu(value);
                }
                Ok(())
            })
        })
    }

    fn validate_plan(plan: &SdftWgpuPlan) -> WgpuResult<()> {
        if plan.window_len() == 0 || plan.bin_count() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan window_len={}, bin_count={}: both values must be greater than zero",
                    plan.window_len(),
                    plan.bin_count()
                ),
            });
        }
        if plan.bin_count() > plan.window_len() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan window_len={}, bin_count={}: bin count must not exceed window length",
                    plan.window_len(),
                    plan.bin_count()
                ),
            });
        }
        if u32::try_from(plan.window_len()).is_err() || u32::try_from(plan.bin_count()).is_err() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan window_len={}, bin_count={}: values exceed the accelerator parameter range",
                    plan.window_len(),
                    plan.bin_count()
                ),
            });
        }
        Ok(())
    }

    fn validate_forward(
        plan: &SdftWgpuPlan,
        input_len: usize,
        output_len: usize,
    ) -> WgpuResult<()> {
        Self::validate_plan(plan)?;
        validate_length(plan.window_len(), input_len)?;
        validate_length(plan.bin_count(), output_len)
    }

    fn validate_inverse(
        plan: &SdftWgpuPlan,
        input_len: usize,
        output_len: usize,
    ) -> WgpuResult<()> {
        Self::validate_plan(plan)?;
        if plan.bin_count() != plan.window_len() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "inverse SDFT requires a complete spectrum: bin_count {} must equal window_len {}",
                    plan.bin_count(),
                    plan.window_len()
                ),
            });
        }
        validate_length(plan.bin_count(), input_len)?;
        validate_length(plan.window_len(), output_len)
    }
}

fn validate_length(expected: usize, actual: usize) -> WgpuResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(WgpuError::LengthMismatch { expected, actual })
    }
}

fn complex_output(len: usize) -> leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1> {
    leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([len])
}

fn real_output(len: usize) -> leto::Array<f32, leto::MnemosyneStorage<f32>, 1> {
    leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([len])
}

#[cfg(test)]
mod tests {
    use super::SdftWgpuBackend;
    use crate::infrastructure::transport::gpu::{SdftWgpuPlan, WgpuError};

    #[test]
    fn inverse_rejects_partial_spectrum_before_device_acquisition() {
        let error = SdftWgpuBackend::validate_inverse(&SdftWgpuPlan::new(8, 4), 4, 8)
            .expect_err("partial SDFT spectrum must not claim inverse reconstruction");
        match error {
            WgpuError::InvalidPlan { message } => assert_eq!(
                message,
                "inverse SDFT requires a complete spectrum: bin_count 4 must equal window_len 8"
            ),
            other => panic!("expected complete-spectrum rejection, got {other}"),
        }
    }
}
