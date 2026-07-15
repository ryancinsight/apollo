//! Hephaestus device acquisition and Mellin execution boundary.

use apollo_fft::PrecisionProfile;
use eunomia::Complex32;
use hephaestus_wgpu::WgpuDevice;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::MellinWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::MellinGpuKernel;
use crate::MellinGpuStorage;

thread_local! {
    static REAL_INPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
}

/// Return whether a default Hephaestus WGPU device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    MellinWgpuBackend::try_default().is_ok()
}

/// Hephaestus WGPU backend for forward and inverse Mellin execution.
#[derive(Debug, Clone)]
pub struct MellinWgpuBackend {
    device: WgpuDevice,
}

impl MellinWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Acquire the default Hephaestus WGPU device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-mellin-wgpu")?))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_inverse(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only concrete-`f32` accelerator plan descriptor.
    #[must_use]
    pub const fn plan(&self, samples: usize, min_scale: f32, max_scale: f32) -> MellinWgpuPlan {
        MellinWgpuPlan::new(samples, min_scale, max_scale)
    }

    /// Execute the forward Mellin log-frequency spectrum for a real-valued `f32` signal.
    pub fn execute_forward(
        &self,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_forward(plan, signal, signal_min, signal_max, plan.samples())?;
        let mut output = vec![Complex32::new(0.0, 0.0); plan.samples()];
        self.execute_forward_into(plan, signal, signal_min, signal_max, &mut output)?;
        Ok(output)
    }

    /// Execute the forward Mellin spectrum into caller-owned storage.
    pub fn execute_forward_into(
        &self,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f32,
        signal_max: f32,
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        Self::validate_forward(plan, signal, signal_min, signal_max, output.len())?;
        MellinGpuKernel::execute_forward_into(
            &self.device,
            plan,
            signal,
            signal_min,
            signal_max,
            output,
        )
    }

    /// Execute the forward Mellin spectrum from a Leto real-valued host view.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before provider upload.
    pub fn execute_forward_leto(
        &self,
        plan: &MellinWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        Self::validate_forward(plan, &signal, signal_min, signal_max, plan.samples())?;
        let mut output = mellin_complex_output(plan.samples());
        let output_slice = output
            .as_slice_mut()
            .expect("Mellin Mnemosyne output must be contiguous");
        self.execute_forward_into(plan, &signal, signal_min, signal_max, output_slice)?;
        Ok(output)
    }

    /// Execute the inverse Mellin transform to a real-valued `f32` signal.
    pub fn execute_inverse(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f32,
        out_max: f32,
        out_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        Self::validate_inverse(plan, spectrum, out_min, out_max, out_len)?;
        let mut output = vec![0.0; out_len];
        self.execute_inverse_into(plan, spectrum, out_min, out_max, &mut output)?;
        Ok(output)
    }

    /// Execute the inverse Mellin transform into caller-owned storage.
    pub fn execute_inverse_into(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f32,
        out_max: f32,
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_inverse(plan, spectrum, out_min, out_max, output.len())?;
        MellinGpuKernel::execute_inverse_into(
            &self.device,
            plan,
            spectrum,
            out_min,
            out_max,
            output,
        )
    }

    /// Execute the inverse Mellin transform from a Leto spectrum host view.
    pub fn execute_inverse_leto(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: leto::ArrayView1<'_, Complex32>,
        out_min: f32,
        out_max: f32,
        out_len: usize,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        Self::validate_inverse(plan, &spectrum, out_min, out_max, out_len)?;
        let mut output = mellin_real_output(out_len);
        let output_slice = output
            .as_slice_mut()
            .expect("Mellin Mnemosyne output must be contiguous");
        self.execute_inverse_into(plan, &spectrum, out_min, out_max, output_slice)?;
        Ok(output)
    }

    /// Execute the forward Mellin spectrum with admitted typed input storage.
    pub fn execute_forward_typed<T: MellinGpuStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_typed_forward::<T>(plan, precision, signal, signal_min, signal_max)?;
        let mut output = vec![Complex32::new(0.0, 0.0); plan.samples()];
        self.execute_forward_typed_into(
            plan,
            precision,
            signal,
            signal_min,
            signal_max,
            &mut output,
        )?;
        Ok(output)
    }

    /// Execute the typed forward Mellin spectrum into caller-owned storage.
    pub fn execute_forward_typed_into<T: MellinGpuStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        signal_min: f32,
        signal_max: f32,
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        Self::validate_typed_forward::<T>(plan, precision, signal, signal_min, signal_max)?;
        if let Some(signal) = T::as_gpu_slice(signal) {
            return self.execute_forward_into(plan, signal, signal_min, signal_max, output);
        }
        REAL_INPUT_SCRATCH.with(|pool| {
            pool.with_scratch(signal.len(), |represented| {
                for (target, value) in represented.iter_mut().zip(signal.iter().copied()) {
                    *target = value.to_gpu();
                }
                self.execute_forward_into(plan, represented, signal_min, signal_max, output)
            })
        })
    }

    /// Execute the typed forward Mellin spectrum from a Leto host view.
    pub fn execute_forward_leto_typed<T: MellinGpuStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, T>,
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        Self::validate_typed_forward::<T>(plan, precision, &signal, signal_min, signal_max)?;
        let mut output = mellin_complex_output(plan.samples());
        let output_slice = output
            .as_slice_mut()
            .expect("Mellin Mnemosyne output must be contiguous");
        self.execute_forward_typed_into(
            plan,
            precision,
            &signal,
            signal_min,
            signal_max,
            output_slice,
        )?;
        Ok(output)
    }

    fn validate_forward(
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f32,
        signal_max: f32,
        output_len: usize,
    ) -> WgpuResult<()> {
        Self::validate_plan(plan)?;
        Self::validate_accelerator_length("signal length", signal.len())?;
        if signal.is_empty() {
            return Err(WgpuError::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if output_len != plan.samples() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.samples(),
                actual: output_len,
            });
        }
        Self::validate_domain("signal", signal_min, signal_max)
    }

    fn validate_inverse(
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f32,
        out_max: f32,
        output_len: usize,
    ) -> WgpuResult<()> {
        Self::validate_plan(plan)?;
        if spectrum.len() != plan.samples() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.samples(),
                actual: spectrum.len(),
            });
        }
        if output_len == 0 {
            return Err(WgpuError::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        Self::validate_accelerator_length("output length", output_len)?;
        Self::validate_domain("output", out_min, out_max)
    }

    fn validate_plan(plan: &MellinWgpuPlan) -> WgpuResult<()> {
        if plan.samples() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "Mellin sample count must be greater than zero".to_owned(),
            });
        }
        Self::validate_accelerator_length("Mellin sample count", plan.samples())?;
        Self::validate_domain("plan scale", plan.min_scale(), plan.max_scale()).map_err(|error| {
            match error {
                WgpuError::InvalidSignalDomain { message } => WgpuError::InvalidPlan { message },
                _ => error,
            }
        })
    }

    fn validate_typed_forward<T: MellinGpuStorage>(
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<()> {
        if precision != T::PROFILE {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Self::validate_plan(plan)?;
        Self::validate_accelerator_length("signal length", signal.len())?;
        if signal.is_empty() {
            return Err(WgpuError::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        Self::validate_domain("signal", signal_min, signal_max)
    }

    fn validate_accelerator_length(label: &str, value: usize) -> WgpuResult<()> {
        if u32::try_from(value).is_err() {
            return Err(WgpuError::InvalidPlan {
                message: format!("{label} {value} exceeds the accelerator parameter range"),
            });
        }
        Ok(())
    }

    fn validate_domain(label: &str, min: f32, max: f32) -> WgpuResult<()> {
        if !min.is_finite() || !max.is_finite() || min <= 0.0 || max <= 0.0 {
            return Err(WgpuError::InvalidSignalDomain {
                message: format!(
                    "{label} bounds must be finite and positive: min={min}, max={max}"
                ),
            });
        }
        if min >= max {
            return Err(WgpuError::InvalidSignalDomain {
                message: format!("{label} minimum must be less than maximum: min={min}, max={max}"),
            });
        }
        Ok(())
    }
}

fn mellin_complex_output(
    samples: usize,
) -> leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1> {
    leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([samples])
}

fn mellin_real_output(samples: usize) -> leto::Array<f32, leto::MnemosyneStorage<f32>, 1> {
    leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([samples])
}
