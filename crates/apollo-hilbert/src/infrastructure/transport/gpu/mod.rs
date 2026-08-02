#![warn(missing_docs)]
//! WGPU backend boundary for Apollo Hilbert.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the Hilbert kernels, their domain names,
//! and the analytic-signal extension surface (real in, complex out) that
//! sits beside the scaffold's symmetric real contract.

/// Infrastructure boundary for the Hilbert kernels.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

use apollo_fft::{Complex32, GpuElement};

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::HilbertGpuKernel;

/// Metadata-preserving WGPU plan descriptor.
pub type HilbertWgpuPlan = apollo_fft::WgpuTransformPlan<HilbertGpuKernel>;

/// WGPU backend descriptor.
pub type HilbertWgpuBackend = apollo_fft::WgpuTransformBackend<HilbertGpuKernel>;

/// Analytic-signal surface of the Hilbert backend.
///
/// The analytic signal `x + i·H{x}` maps a real signal to complex
/// output, so it extends the scaffold's symmetric real contract rather
/// than instantiating it.
pub trait AnalyticSignal {
    /// Execute the analytic signal `x + i·H{x}`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_analytic_signal(
        &self,
        plan: &HilbertWgpuPlan,
        input: &[f32],
    ) -> WgpuResult<Vec<Complex32>>;

    /// Execute the analytic signal into caller-owned complex storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_analytic_signal_into(
        &self,
        plan: &HilbertWgpuPlan,
        input: &[f32],
        output: &mut [Complex32],
    ) -> WgpuResult<()>;

    /// Execute the analytic signal from a Leto real-valued host view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_analytic_signal_leto(
        &self,
        plan: &HilbertWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>>;
}

impl AnalyticSignal for HilbertWgpuBackend {
    fn execute_analytic_signal(
        &self,
        plan: &HilbertWgpuPlan,
        input: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
        let mut output = vec![Complex32::new(0.0, 0.0); plan.len()];
        self.execute_analytic_signal_into(plan, input, &mut output)?;
        Ok(output)
    }

    fn execute_analytic_signal_into(
        &self,
        plan: &HilbertWgpuPlan,
        input: &[f32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        validate_analytic_lengths(plan, input.len(), output.len())?;
        Complex32::with_input_scratch(input.len(), |promoted| {
            for (target, value) in promoted.iter_mut().zip(input.iter().copied()) {
                *target = Complex32::new(value, 0.0);
            }
            HilbertGpuKernel::execute_analytic_into(self.device(), promoted, output)?;
            for (sample, original) in output.iter_mut().zip(input.iter().copied()) {
                sample.re = original;
            }
            Ok(())
        })
    }

    fn execute_analytic_signal_leto(
        &self,
        plan: &HilbertWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([
                plan.len()
            ]);
        let output_slice = output
            .as_slice_mut()
            .expect("Hilbert Mnemosyne output must be contiguous");
        self.execute_analytic_signal_into(plan, &input, output_slice)?;
        Ok(output)
    }
}

fn validate_analytic_lengths(
    plan: &HilbertWgpuPlan,
    input_len: usize,
    output_len: usize,
) -> WgpuResult<()> {
    if plan.is_empty() {
        return Err(WgpuError::InvalidPlan {
            message: "transform length must be greater than zero".to_owned(),
        });
    }
    if input_len != plan.len() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.len(),
            actual: input_len,
        });
    }
    if output_len != plan.len() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.len(),
            actual: output_len,
        });
    }
    Ok(())
}
