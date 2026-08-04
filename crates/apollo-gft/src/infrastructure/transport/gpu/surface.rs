use apollo_fft::{GpuStorage, PrecisionProfile};

use super::{GftWgpuPlan, WgpuResult};

/// Basis-parameterized execution surface of the GFT backend.
///
/// The graph Fourier basis `U` (row-major `n x n`, eigenvectors of the
/// graph Laplacian in columns) is an operand of every operation, so
/// this surface extends the scaffold rather than instantiating its
/// slice contract.
pub trait BasisTransform {
    /// Execute the forward GFT `X = U^T x`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch, or
    /// provider failure.
    fn execute_forward(
        &self,
        plan: &GftWgpuPlan,
        signal: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>>;

    /// Execute the forward GFT into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch, or
    /// provider failure.
    fn execute_forward_into(
        &self,
        plan: &GftWgpuPlan,
        signal: &[f32],
        basis: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()>;

    /// Execute the forward GFT from Leto host views.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch, or
    /// provider failure.
    fn execute_forward_leto(
        &self,
        plan: &GftWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>>;

    /// Execute the inverse GFT `x = U X`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch, or
    /// provider failure.
    fn execute_inverse(
        &self,
        plan: &GftWgpuPlan,
        spectrum: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>>;

    /// Execute the inverse GFT into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch, or
    /// provider failure.
    fn execute_inverse_into(
        &self,
        plan: &GftWgpuPlan,
        spectrum: &[f32],
        basis: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()>;

    /// Execute the inverse GFT from Leto host views.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch, or
    /// provider failure.
    fn execute_inverse_leto(
        &self,
        plan: &GftWgpuPlan,
        spectrum: leto::ArrayView1<'_, f32>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>>;

    /// Execute the forward GFT with storage admitted by the `f32`
    /// accelerator contract.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch,
    /// precision-profile, or provider failure.
    fn execute_forward_typed_into<T: GpuStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Execute typed forward GFT from Leto host views.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch,
    /// precision-profile, or provider failure.
    fn execute_forward_leto_typed<T: GpuStorage + Default>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, T>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;

    /// Execute the inverse GFT with storage admitted by the `f32`
    /// accelerator contract.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch,
    /// precision-profile, or provider failure.
    fn execute_inverse_typed_into<T: GpuStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Execute typed inverse GFT from Leto host views.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, shape-mismatch,
    /// precision-profile, or provider failure.
    fn execute_inverse_leto_typed<T: GpuStorage + Default>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: leto::ArrayView1<'_, T>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;
}
