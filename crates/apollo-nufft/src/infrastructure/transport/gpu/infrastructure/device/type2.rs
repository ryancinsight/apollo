use apollo_fft::PrecisionProfile;
use crate::NufftComplexStorage;
use leto::Array3;
use eunomia::{Complex32, Complex64};

use crate::infrastructure::transport::gpu::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::device::helpers::{
    array3_from_leto_view, leto_array1_from_slice, leto_view1_cow, positions3_from_leto_view,
    typed_to_complex32, validate_typed_profile, validate_usize_to_u32, write_typed_output,
};
use crate::infrastructure::transport::gpu::infrastructure::device::NufftWgpuBackend;

impl NufftWgpuBackend {
    /// Execute exact direct Type-2 1D NUFFT on WGPU.
    pub fn execute_type2_1d(
        &self,
        plan: &NufftWgpuPlan1D,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex64>> {
        if fourier_coeffs.len() != plan.domain().n {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected: plan.domain().n,
                actual: fourier_coeffs.len(),
            });
        }
        validate_usize_to_u32(plan.domain().n)?;
        validate_usize_to_u32(positions.len())?;
        let output = self.kernel.execute_type2_1d(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan.domain().n,
            plan.domain().length() as f32,
            fourier_coeffs,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect())
    }

    /// Execute exact direct Type-2 1D NUFFT with caller-owned typed storage.
    pub fn execute_type2_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan1D,
        precision: PrecisionProfile,
        fourier_coeffs: &[T],
        positions: &[f32],
        output: &mut [T],
    ) -> NufftWgpuResult<()> {
        validate_typed_profile::<T>(precision)?;
        if output.len() != positions.len() {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected: positions.len(),
                actual: output.len(),
            });
        }
        let coefficients32 = typed_to_complex32(fourier_coeffs);
        let computed = self.execute_type2_1d(plan, &coefficients32, positions)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    /// Execute exact direct Type-2 1D NUFFT from Leto views.
    pub fn execute_type2_1d_leto(
        &self,
        plan: &NufftWgpuPlan1D,
        fourier_coeffs: leto::ArrayView1<'_, Complex32>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> NufftWgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let fourier_coeffs = leto_view1_cow(fourier_coeffs);
        let positions = leto_view1_cow(positions);
        let output = self.execute_type2_1d(plan, fourier_coeffs.as_ref(), positions.as_ref())?;
        leto_array1_from_slice(&output)
    }

    /// Execute exact direct Type-2 1D NUFFT from typed Leto views.
    pub fn execute_type2_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan1D,
        precision: PrecisionProfile,
        fourier_coeffs: leto::ArrayView1<'_, T>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let fourier_coeffs = leto_view1_cow(fourier_coeffs);
        let positions = leto_view1_cow(positions);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); positions.len()];
        self.execute_type2_1d_leto_typed_into(
            plan,
            precision,
            &fourier_coeffs,
            &positions,
            &mut output,
        )?;
        leto_array1_from_slice(&output)
    }

    // private helper for typed leto execution
    fn execute_type2_1d_leto_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan1D,
        precision: PrecisionProfile,
        fourier_coeffs: &[T],
        positions: &[f32],
        output: &mut [T],
    ) -> NufftWgpuResult<()> {
        self.execute_type2_1d_typed_into(plan, precision, fourier_coeffs, positions, output)
    }

    /// Execute exact direct Type-2 3D NUFFT on WGPU.
    pub fn execute_type2_3d(
        &self,
        plan: &NufftWgpuPlan3D,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex64>> {
        let grid = plan.grid();
        if modes.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(NufftWgpuError::InvalidPlan {
                message: "mode shape must match 3D plan grid dimensions",
            });
        }
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let (lx, ly, lz) = grid.lengths();
        let coefficients: Vec<Complex32> = modes.iter().copied().collect();
        let output = self.kernel.execute_type2_3d(
            self.device.inner(),
            self.device.queue().as_ref(),
            (grid.nx, grid.ny, grid.nz),
            (lx as f32, ly as f32, lz as f32),
            &coefficients,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect())
    }

    /// Execute exact direct Type-2 3D NUFFT with caller-owned typed storage.
    pub fn execute_type2_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan3D,
        precision: PrecisionProfile,
        modes: &Array3<T>,
        positions: &[(f32, f32, f32)],
        output: &mut [T],
    ) -> NufftWgpuResult<()> {
        validate_typed_profile::<T>(precision)?;
        if output.len() != positions.len() {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected: positions.len(),
                actual: output.len(),
            });
        }
        let modes32 = modes.mapv(|value| {
            let represented = value.to_complex64();
            Complex32::new(represented.re as f32, represented.im as f32)
        });
        let computed = self.execute_type2_3d(plan, &modes32, positions)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    /// Execute exact direct Type-2 3D NUFFT from Leto views.
    pub fn execute_type2_3d_leto(
        &self,
        plan: &NufftWgpuPlan3D,
        modes: leto::ArrayView3<'_, Complex32>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> NufftWgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let modes = array3_from_leto_view(modes);
        let positions = positions3_from_leto_view(positions)?;
        let output = self.execute_type2_3d(plan, &modes, &positions)?;
        leto_array1_from_slice(&output)
    }

    /// Execute exact direct Type-2 3D NUFFT from typed Leto views.
    pub fn execute_type2_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan3D,
        precision: PrecisionProfile,
        modes: leto::ArrayView3<'_, T>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let modes = array3_from_leto_view(modes);
        let positions = positions3_from_leto_view(positions)?;
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); positions.len()];
        self.execute_type2_3d_typed_into(plan, precision, &modes, &positions, &mut output)?;
        leto_array1_from_slice(&output)
    }
}
