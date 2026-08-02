#![warn(missing_docs)]
//! WGPU backend boundary for Apollo Mellin.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the Mellin kernels, their domain names,
//! and the resampled execution surface. Every Mellin operation carries
//! per-call value-domain bounds (the signal or output min/max over which
//! log/exponential resampling runs), so the marker implements only the
//! planner contract and the surface lives on [`ResampledExecution`].

/// Infrastructure boundary for the Mellin kernels.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

use apollo_fft::{Complex32, GpuElement, GpuStorage, PrecisionProfile};

use infrastructure::kernel::{validate_domain, MellinGpuKernel as Kernel};

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::MellinGpuKernel;

/// Plan payload for the Mellin transform: sample count and the scale
/// range of the generated spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalePlan {
    samples: usize,
    min_scale: f32,
    max_scale: f32,
}

impl ScalePlan {
    /// Create a scale-plan payload.
    ///
    /// Validation (positive ordered finite scales, accelerator range)
    /// runs at dispatch.
    #[must_use]
    pub const fn new(samples: usize, min_scale: f32, max_scale: f32) -> Self {
        Self {
            samples,
            min_scale,
            max_scale,
        }
    }

    /// Return the spectrum sample count carried by this payload.
    #[must_use]
    pub const fn samples(self) -> usize {
        self.samples
    }

    /// Return the smallest generated scale.
    #[must_use]
    pub const fn min_scale(self) -> f32 {
        self.min_scale
    }

    /// Return the largest generated scale.
    #[must_use]
    pub const fn max_scale(self) -> f32 {
        self.max_scale
    }
}

/// Metadata-preserving WGPU plan descriptor.
pub type MellinWgpuPlan = apollo_fft::WgpuTransformPlan<MellinGpuKernel>;

/// WGPU backend descriptor.
pub type MellinWgpuBackend = apollo_fft::WgpuTransformBackend<MellinGpuKernel>;

/// Resampled execution surface of the Mellin backend.
///
/// The forward direction log-resamples a positive-valued signal over
/// per-call `[signal_min, signal_max]` bounds before projecting onto the
/// Mellin basis; the inverse reconstructs over per-call output bounds.
/// The bounds are operands, not plan state, so this surface extends the
/// scaffold rather than instantiating its slice contract.
pub trait ResampledExecution {
    /// Execute the forward Mellin spectrum.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, invalid-signal-domain, length-mismatch,
    /// or provider failure.
    fn execute_forward(
        &self,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<Vec<Complex32>>;

    /// Execute the forward Mellin spectrum into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, invalid-signal-domain, length-mismatch,
    /// or provider failure.
    fn execute_forward_into(
        &self,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f32,
        signal_max: f32,
        output: &mut [Complex32],
    ) -> WgpuResult<()>;

    /// Execute the forward Mellin spectrum from a Leto host view.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before provider upload.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, invalid-signal-domain, length-mismatch,
    /// or provider failure.
    fn execute_forward_leto(
        &self,
        plan: &MellinWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>>;

    /// Execute the forward Mellin spectrum with admitted typed input
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, invalid-signal-domain, length-mismatch,
    /// precision-profile, or provider failure.
    fn execute_forward_typed<T: GpuStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<Vec<Complex32>>;

    /// Execute the typed forward Mellin spectrum into caller-owned
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, invalid-signal-domain, length-mismatch,
    /// precision-profile, or provider failure.
    fn execute_forward_typed_into<T: GpuStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        signal_min: f32,
        signal_max: f32,
        output: &mut [Complex32],
    ) -> WgpuResult<()>;

    /// Execute the typed forward Mellin spectrum from a Leto host view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, invalid-signal-domain, length-mismatch,
    /// precision-profile, or provider failure.
    fn execute_forward_leto_typed<T: GpuStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, T>,
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>>;

    /// Execute the inverse Mellin reconstruction.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, invalid-signal-domain, length-mismatch,
    /// or provider failure.
    fn execute_inverse(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f32,
        out_max: f32,
        output_len: usize,
    ) -> WgpuResult<Vec<f32>>;

    /// Execute the inverse Mellin reconstruction into caller-owned
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, invalid-signal-domain, length-mismatch,
    /// or provider failure.
    fn execute_inverse_into(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f32,
        out_max: f32,
        output: &mut [f32],
    ) -> WgpuResult<()>;

    /// Execute the inverse Mellin reconstruction from a Leto host view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, invalid-signal-domain, length-mismatch,
    /// or provider failure.
    fn execute_inverse_leto(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: leto::ArrayView1<'_, Complex32>,
        out_min: f32,
        out_max: f32,
        output_len: usize,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>>;
}

impl ResampledExecution for MellinWgpuBackend {
    fn execute_forward(
        &self,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<Vec<Complex32>> {
        let mut output = vec![Complex32::new(0.0, 0.0); plan.payload().samples()];
        self.execute_forward_into(plan, signal, signal_min, signal_max, &mut output)?;
        Ok(output)
    }

    fn execute_forward_into(
        &self,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f32,
        signal_max: f32,
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        validate_forward(plan, signal.len(), signal_min, signal_max, output.len())?;
        Kernel::execute_forward_into(
            self.device(),
            plan.payload(),
            signal,
            signal_min,
            signal_max,
            output,
        )
    }

    fn execute_forward_leto(
        &self,
        plan: &MellinWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let mut output =
            leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([plan
                .payload()
                .samples()]);
        let output_slice = output
            .as_slice_mut()
            .expect("Mellin Mnemosyne output must be contiguous");
        self.execute_forward_into(plan, &signal, signal_min, signal_max, output_slice)?;
        Ok(output)
    }

    fn execute_forward_typed<T: GpuStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<Vec<Complex32>> {
        let mut output = vec![Complex32::new(0.0, 0.0); plan.payload().samples()];
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

    fn execute_forward_typed_into<T: GpuStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        signal_min: f32,
        signal_max: f32,
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        if precision != T::PROFILE {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        if let Some(signal) = T::as_element_slice(signal) {
            return self.execute_forward_into(plan, signal, signal_min, signal_max, output);
        }
        f32::with_input_scratch(signal.len(), |represented| {
            for (target, value) in represented.iter_mut().zip(signal.iter().copied()) {
                *target = value.to_gpu();
            }
            self.execute_forward_into(plan, represented, signal_min, signal_max, output)
        })
    }

    fn execute_forward_leto_typed<T: GpuStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, T>,
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let mut output =
            leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([plan
                .payload()
                .samples()]);
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

    fn execute_inverse(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f32,
        out_max: f32,
        output_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; output_len];
        self.execute_inverse_into(plan, spectrum, out_min, out_max, &mut output)?;
        Ok(output)
    }

    fn execute_inverse_into(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f32,
        out_max: f32,
        output: &mut [f32],
    ) -> WgpuResult<()> {
        validate_inverse(plan, spectrum.len(), out_min, out_max, output.len())?;
        Kernel::execute_inverse_into(
            self.device(),
            plan.payload(),
            spectrum,
            out_min,
            out_max,
            output,
        )
    }

    fn execute_inverse_leto(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: leto::ArrayView1<'_, Complex32>,
        out_min: f32,
        out_max: f32,
        output_len: usize,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([output_len]);
        let output_slice = output
            .as_slice_mut()
            .expect("Mellin Mnemosyne inverse output must be contiguous");
        self.execute_inverse_into(plan, &spectrum, out_min, out_max, output_slice)?;
        Ok(output)
    }
}

fn validate_forward(
    plan: &MellinWgpuPlan,
    signal_len: usize,
    signal_min: f32,
    signal_max: f32,
    output_len: usize,
) -> WgpuResult<()> {
    plan.validate()?;
    validate_accelerator_length("signal length", signal_len)?;
    if signal_len == 0 {
        return Err(WgpuError::LengthMismatch {
            expected: 1,
            actual: 0,
        });
    }
    if output_len != plan.payload().samples() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.payload().samples(),
            actual: output_len,
        });
    }
    validate_domain("signal", signal_min, signal_max)
}

fn validate_inverse(
    plan: &MellinWgpuPlan,
    spectrum_len: usize,
    out_min: f32,
    out_max: f32,
    output_len: usize,
) -> WgpuResult<()> {
    plan.validate()?;
    if spectrum_len != plan.payload().samples() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.payload().samples(),
            actual: spectrum_len,
        });
    }
    if output_len == 0 {
        return Err(WgpuError::LengthMismatch {
            expected: 1,
            actual: 0,
        });
    }
    validate_accelerator_length("output length", output_len)?;
    validate_domain("output", out_min, out_max)
}

fn validate_accelerator_length(label: &str, value: usize) -> WgpuResult<()> {
    if u32::try_from(value).is_err() {
        return Err(WgpuError::InvalidPlan {
            message: format!("{label} {value} exceeds the accelerator parameter range"),
        });
    }
    Ok(())
}
