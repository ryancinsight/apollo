#![warn(missing_docs)]
//! WGPU backend boundary for Apollo STFT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the STFT kernels, their domain names,
//! and the framed surface. The spectrum length derives from the signal
//! length (frames = 1 + ceil(len / hop)), not from the plan alone, so
//! the marker implements only the planner contract and the surface —
//! including the reusable-buffer dispatch path — lives on
//! [`FramedExecution`].

mod execution;
mod frame;
/// Infrastructure boundary for the STFT kernels.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

use apollo_fft::{Complex32, GpuStorage, PrecisionProfile};

use infrastructure::buffers::StftGpuBuffers;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use execution::{forward_output_len, required_device_limits};
pub use frame::FramePlan;
pub use infrastructure::buffers::StftGpuBuffers as StftBuffers;
pub use infrastructure::kernel::StftGpuKernel;

/// Metadata-preserving WGPU plan descriptor.
pub type StftWgpuPlan = apollo_fft::WgpuTransformPlan<StftGpuKernel>;

/// WGPU backend descriptor.
pub type StftWgpuBackend = apollo_fft::WgpuTransformBackend<StftGpuKernel>;

/// Framed surface of the STFT backend.
///
/// Forward analysis produces `frame_count * frame_len` complex bins for
/// a real signal; inverse WOLA reconstruction recovers `signal_len`
/// real samples. The reusable-buffer path amortizes device allocations
/// across repeated dispatches of one geometry.
pub trait FramedExecution {
    /// Execute the forward STFT on `signal`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, input-too-short, or provider failure.
    fn execute_forward(&self, plan: &StftWgpuPlan, signal: &[f32]) -> WgpuResult<Vec<Complex32>>;

    /// Execute the forward STFT from a Leto real signal view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, input-too-short, or provider failure.
    fn execute_forward_leto(
        &self,
        plan: &StftWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>>;

    /// Execute the forward STFT with typed real input and typed complex
    /// spectrum output.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, input-too-short, length-mismatch,
    /// precision-profile, or provider failure.
    fn execute_forward_typed_into<I, O>(
        &self,
        plan: &StftWgpuPlan,
        precision: PrecisionProfile,
        signal: &[I],
        output: &mut [O],
    ) -> WgpuResult<()>
    where
        I: GpuStorage,
        O: GpuStorage<Complex32>;

    /// Execute the typed forward STFT from a Leto real signal view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, input-too-short, precision-profile, or
    /// provider failure.
    fn execute_forward_leto_typed<I, O>(
        &self,
        plan: &StftWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, I>,
    ) -> WgpuResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>>
    where
        I: GpuStorage,
        O: GpuStorage<Complex32> + Default;

    /// Execute the inverse STFT (WOLA reconstruction).
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_inverse(
        &self,
        plan: &StftWgpuPlan,
        spectrum: &[Complex32],
        signal_len: usize,
    ) -> WgpuResult<Vec<f32>>;

    /// Execute the inverse STFT from a Leto complex spectrum view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_inverse_leto(
        &self,
        plan: &StftWgpuPlan,
        spectrum: leto::ArrayView1<'_, Complex32>,
        signal_len: usize,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>>;

    /// Execute the inverse STFT with typed spectrum input and typed real
    /// output.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    fn execute_inverse_typed_into<I, O>(
        &self,
        plan: &StftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &[I],
        signal_len: usize,
        output: &mut [O],
    ) -> WgpuResult<()>
    where
        I: GpuStorage<Complex32>,
        O: GpuStorage;

    /// Execute the typed inverse STFT from a Leto spectrum view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    fn execute_inverse_leto_typed<I, O>(
        &self,
        plan: &StftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: leto::ArrayView1<'_, I>,
        signal_len: usize,
    ) -> WgpuResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>>
    where
        I: GpuStorage<Complex32>,
        O: GpuStorage + Default;

    /// Allocate pre-allocated GPU buffers for repeated STFT dispatches
    /// with the given plan and signal length.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan or provider failure.
    fn make_buffers(&self, plan: &StftWgpuPlan, signal_len: usize) -> WgpuResult<StftGpuBuffers>;

    /// Execute the forward STFT using pre-allocated GPU buffers.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, input-too-short, or provider failure.
    fn execute_forward_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        signal: &[f32],
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()>;

    /// Execute the inverse STFT using pre-allocated GPU buffers.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_inverse_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        spectrum: &[Complex32],
        signal_len: usize,
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()>;
}
