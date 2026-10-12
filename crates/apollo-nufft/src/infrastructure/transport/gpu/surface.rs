//! Non-uniform execution surfaces of the NUFFT backend.
//!
//! Every NUFFT operation takes the non-uniform sample positions as a
//! per-call operand, so the marker implements only the planner contract and
//! these surfaces extend the shared scaffold instead of instantiating its
//! slice contract. [`DirectTransform`] carries the exact direct-sum paths;
//! [`GriddedTransform`] carries the Kaiser-Bessel gridded paths and their
//! reusable-buffer dispatch.

use crate::NufftComplexStorage;
use apollo_fft::{PrecisionProfile, WgpuResult};
use eunomia::{Complex32, Complex64};
use leto::Array3;

#[cfg(any(test, feature = "diagnostics"))]
use super::NufftType2GridDiagnostics;
use super::{NufftGpuBuffers1D, NufftGpuBuffers3D, NufftWgpuPlan};

/// Exact direct-sum NUFFT execution over the shared WGPU backend.
///
/// Type-1 spreads non-uniform samples onto the uniform mode grid; Type-2
/// evaluates the mode field at non-uniform positions. Both cost `O(M * N)`
/// per dispatch and satisfy the adjoint law in exact arithmetic.
pub trait DirectTransform {
    /// Execute exact direct Type-1 1D NUFFT on WGPU.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_type1_1d(
        &self,
        plan: &NufftWgpuPlan,
        positions: &[f32],
        values: &[Complex32],
    ) -> WgpuResult<leto::Array1<Complex64>>;

    /// Execute exact direct Type-1 1D NUFFT with caller-owned typed storage.
    ///
    /// WGPU arithmetic remains `f32`. Mixed `[F16; 2]` storage is promoted once
    /// to represented `Complex32` before dispatch, then quantized at the output
    /// boundary.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_type1_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: &[f32],
        values: &[T],
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Execute exact direct Type-1 1D NUFFT from Leto views.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_type1_1d_leto(
        &self,
        plan: &NufftWgpuPlan,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>>;

    /// Execute exact direct Type-1 1D NUFFT from typed Leto views.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_type1_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;

    /// Execute exact direct Type-1 3D NUFFT on WGPU.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_type1_3d(
        &self,
        plan: &NufftWgpuPlan,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> WgpuResult<Array3<Complex64>>;

    /// Execute exact direct Type-1 3D NUFFT with caller-owned typed storage.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, shape-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_type1_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: &[(f32, f32, f32)],
        values: &[T],
        output: &mut Array3<T>,
    ) -> WgpuResult<()>;

    /// Execute exact direct Type-1 3D NUFFT from Leto views.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_type1_3d_leto(
        &self,
        plan: &NufftWgpuPlan,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 3>>;

    /// Execute exact direct Type-1 3D NUFFT from typed Leto views.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_type1_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 3>>;

    /// Execute exact direct Type-2 1D NUFFT on WGPU.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_type2_1d(
        &self,
        plan: &NufftWgpuPlan,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> WgpuResult<Vec<Complex64>>;

    /// Execute exact direct Type-2 1D NUFFT with caller-owned typed storage.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_type2_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        fourier_coeffs: &[T],
        positions: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Execute exact direct Type-2 1D NUFFT from Leto views.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_type2_1d_leto(
        &self,
        plan: &NufftWgpuPlan,
        fourier_coeffs: leto::ArrayView1<'_, Complex32>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>>;

    /// Execute exact direct Type-2 1D NUFFT from typed Leto views.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_type2_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        fourier_coeffs: leto::ArrayView1<'_, T>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;

    /// Execute exact direct Type-2 3D NUFFT on WGPU.
    ///
    /// # Errors
    ///
    /// Returns a shape-mismatch, invalid-plan, or provider failure.
    fn execute_type2_3d(
        &self,
        plan: &NufftWgpuPlan,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> WgpuResult<Vec<Complex64>>;

    /// Execute exact direct Type-2 3D NUFFT with caller-owned typed storage.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_type2_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        modes: &Array3<T>,
        positions: &[(f32, f32, f32)],
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Execute exact direct Type-2 3D NUFFT from Leto views.
    ///
    /// # Errors
    ///
    /// Returns a shape-mismatch, invalid-plan, or provider failure.
    fn execute_type2_3d_leto(
        &self,
        plan: &NufftWgpuPlan,
        modes: leto::ArrayView3<'_, Complex32>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>>;

    /// Execute exact direct Type-2 3D NUFFT from typed Leto views.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_type2_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        modes: leto::ArrayView3<'_, T>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;
}

/// Kaiser-Bessel gridded fast NUFFT execution over the shared WGPU backend.
///
/// The gridded paths trade the direct sum's exactness for `O(M + N log N)`
/// complexity with a plan-derived accuracy bound. Reusable buffers amortize
/// device allocation and pipeline preparation across repeated dispatches of
/// one geometry.
pub trait GriddedTransform {
    /// Execute fast gridded Type-1 1D NUFFT on WGPU.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type1_1d(
        &self,
        plan: &NufftWgpuPlan,
        positions: &[f32],
        values: &[Complex32],
    ) -> WgpuResult<leto::Array1<Complex64>>;

    /// Execute fast gridded Type-1 1D NUFFT with caller-owned typed storage.
    ///
    /// WGPU arithmetic remains `f32`. `Complex32` storage is passed through and
    /// mixed `[F16; 2]` storage is promoted once to represented `Complex32`
    /// before dispatch, then quantized back at the output boundary.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_fast_type1_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: &[f32],
        values: &[T],
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Execute fast gridded Type-1 1D NUFFT from Leto views.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type1_1d_leto(
        &self,
        plan: &NufftWgpuPlan,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>>;

    /// Execute fast gridded Type-1 1D NUFFT from typed Leto views.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_fast_type1_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;

    /// Execute fast gridded Type-1 1D NUFFT with exclusively borrowed
    /// pre-allocated buffers.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type1_1d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers1D,
        positions: &[f32],
        values: &[Complex32],
        output: &mut [Complex64],
    ) -> WgpuResult<()>;

    /// Execute fast gridded Type-1 3D NUFFT on WGPU.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type1_3d(
        &self,
        plan: &NufftWgpuPlan,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> WgpuResult<Array3<Complex64>>;

    /// Execute fast gridded Type-1 3D NUFFT with caller-owned typed storage.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, shape-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_fast_type1_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: &[(f32, f32, f32)],
        values: &[T],
        output: &mut Array3<T>,
    ) -> WgpuResult<()>;

    /// Execute fast gridded Type-1 3D NUFFT from Leto views.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type1_3d_leto(
        &self,
        plan: &NufftWgpuPlan,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 3>>;

    /// Execute fast gridded Type-1 3D NUFFT from typed Leto views.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_fast_type1_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 3>>;

    /// Execute fast gridded Type-1 3D NUFFT with exclusively borrowed
    /// pre-allocated buffers.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type1_3d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers3D,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
        output: &mut [Complex64],
    ) -> WgpuResult<()>;

    /// Execute fast gridded Type-2 1D NUFFT on WGPU.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type2_1d(
        &self,
        plan: &NufftWgpuPlan,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> WgpuResult<Vec<Complex64>>;

    /// Execute fast gridded Type-2 1D NUFFT with caller-owned typed storage.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_fast_type2_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        fourier_coeffs: &[T],
        positions: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Execute fast gridded Type-2 1D NUFFT from Leto views.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type2_1d_leto(
        &self,
        plan: &NufftWgpuPlan,
        fourier_coeffs: leto::ArrayView1<'_, Complex32>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>>;

    /// Execute fast gridded Type-2 1D NUFFT from typed Leto views.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_fast_type2_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        fourier_coeffs: leto::ArrayView1<'_, T>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;

    /// Execute fast gridded Type-2 1D NUFFT with exclusively borrowed
    /// pre-allocated buffers.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type2_1d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers1D,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
        output: &mut [Complex64],
    ) -> WgpuResult<()>;

    /// Execute fast gridded Type-2 1D NUFFT and return debug grid snapshots.
    #[cfg(any(test, feature = "diagnostics"))]
    fn execute_fast_type2_1d_with_diagnostics(
        &self,
        plan: &NufftWgpuPlan,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> WgpuResult<(Vec<Complex64>, NufftType2GridDiagnostics)>;

    /// Execute fast gridded Type-2 3D NUFFT on WGPU.
    ///
    /// # Errors
    ///
    /// Returns a shape-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type2_3d(
        &self,
        plan: &NufftWgpuPlan,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> WgpuResult<Vec<Complex64>>;

    /// Execute fast gridded Type-2 3D NUFFT with caller-owned typed storage.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_fast_type2_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        modes: &Array3<T>,
        positions: &[(f32, f32, f32)],
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Execute fast gridded Type-2 3D NUFFT from Leto views.
    ///
    /// # Errors
    ///
    /// Returns a shape-mismatch, invalid-plan, or provider failure.
    fn execute_fast_type2_3d_leto(
        &self,
        plan: &NufftWgpuPlan,
        modes: leto::ArrayView3<'_, Complex32>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>>;

    /// Execute fast gridded Type-2 3D NUFFT from typed Leto views.
    ///
    /// # Errors
    ///
    /// Returns a precision-profile, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_fast_type2_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan,
        precision: PrecisionProfile,
        modes: leto::ArrayView3<'_, T>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;

    /// Execute fast gridded Type-2 3D NUFFT with exclusively borrowed
    /// pre-allocated buffers.
    ///
    /// # Errors
    ///
    /// Returns a shape-mismatch, length-mismatch, invalid-plan, or
    /// provider failure.
    fn execute_fast_type2_3d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers3D,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
        output: &mut [Complex64],
    ) -> WgpuResult<()>;

    /// Execute fast gridded Type-2 3D NUFFT and return debug grid snapshots.
    #[cfg(any(test, feature = "diagnostics"))]
    fn execute_fast_type2_3d_with_diagnostics(
        &self,
        plan: &NufftWgpuPlan,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> WgpuResult<(Vec<Complex64>, NufftType2GridDiagnostics)>;
}
