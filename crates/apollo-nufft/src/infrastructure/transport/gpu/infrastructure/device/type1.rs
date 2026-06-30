use apollo_fft::PrecisionProfile;
use crate::NufftComplexStorage;
use leto::{Array1, Array3};
use eunomia::{Complex32, Complex64};

use crate::infrastructure::transport::gpu::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::device::helpers::{
    leto_array1_from_slice, leto_array3_from_ndarray, leto_view1_cow, positions3_from_leto_view,
    typed_to_complex32, validate_pair_lengths, validate_typed_profile, validate_usize_to_u32,
    write_typed_output,
};
use crate::infrastructure::transport::gpu::infrastructure::device::NufftWgpuBackend;

impl NufftWgpuBackend {
    /// Execute exact direct Type-1 1D NUFFT on WGPU.
    pub fn execute_type1_1d(
        &self,
        plan: &NufftWgpuPlan1D,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Array1<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(plan.domain().n)?;
        validate_usize_to_u32(positions.len())?;
        let output = self.kernel.execute_type1_1d(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan.domain().n,
            plan.domain().length() as f32,
            positions,
            values,
        )?;
        Ok(Array1::from(
            output
                .into_iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        ))
    }

    /// Execute exact direct Type-1 1D NUFFT with caller-owned typed storage.
    ///
    /// WGPU arithmetic remains `f32`. Mixed `[f16; 2]` storage is promoted once
    /// to represented `Complex32` before dispatch, then quantized at the output
    /// boundary.
    pub fn execute_type1_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan1D,
        precision: PrecisionProfile,
        positions: &[f32],
        values: &[T],
        output: &mut [T],
    ) -> NufftWgpuResult<()> {
        validate_typed_profile::<T>(precision)?;
        if output.len() != plan.domain().n {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected: plan.domain().n,
                actual: output.len(),
            });
        }
        let values32 = typed_to_complex32(values);
        let computed = self.execute_type1_1d(plan, positions, &values32)?;
        write_typed_output(computed.as_slice().unwrap_or(&[]), output);
        Ok(())
    }

    /// Execute exact direct Type-1 1D NUFFT from Leto views.
    pub fn execute_type1_1d_leto(
        &self,
        plan: &NufftWgpuPlan1D,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> NufftWgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let positions = leto_view1_cow(positions);
        let values = leto_view1_cow(values);
        let output = self.execute_type1_1d(plan, positions.as_ref(), values.as_ref())?;
        leto_array1_from_slice(output.as_slice().ok_or(NufftWgpuError::InvalidPlan {
            message: "type1 1D Leto output must be contiguous",
        })?)
    }

    /// Execute exact direct Type-1 1D NUFFT from typed Leto views.
    pub fn execute_type1_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan1D,
        precision: PrecisionProfile,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let positions = leto_view1_cow(positions);
        let values = leto_view1_cow(values);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); plan.domain().n];
        self.execute_type1_1d_typed_into(
            plan,
            precision,
            positions.as_ref(),
            values.as_ref(),
            &mut output,
        )?;
        leto_array1_from_slice(&output)
    }

    /// Execute exact direct Type-1 3D NUFFT on WGPU.
    pub fn execute_type1_3d(
        &self,
        plan: &NufftWgpuPlan3D,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Array3<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        let grid = plan.grid();
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let (lx, ly, lz) = grid.lengths();
        let output = self.kernel.execute_type1_3d(
            self.device.inner(),
            self.device.queue().as_ref(),
            [grid.nx, grid.ny, grid.nz],
            (lx as f32, ly as f32, lz as f32),
            positions,
            values,
        )?;
        let converted: Vec<Complex64> = output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        Array3::from_shape_vec([grid.nx, grid.ny, grid.nz], converted).map_err(|_| {
            NufftWgpuError::InvalidPlan {
                message: "3D output shape does not match grid dimensions",
            }
        })
    }

    /// Execute exact direct Type-1 3D NUFFT with caller-owned typed storage.
    pub fn execute_type1_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan3D,
        precision: PrecisionProfile,
        positions: &[(f32, f32, f32)],
        values: &[T],
        output: &mut Array3<T>,
    ) -> NufftWgpuResult<()> {
        validate_typed_profile::<T>(precision)?;
        let grid = plan.grid();
        if output.shape() != (grid.nx, grid.ny, grid.nz) {
            return Err(NufftWgpuError::InvalidPlan {
                message: "typed output shape must match 3D plan grid dimensions",
            });
        }
        let values32 = typed_to_complex32(values);
        let computed = self.execute_type1_3d(plan, positions, &values32)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_complex64(value);
        }
        Ok(())
    }

    /// Execute exact direct Type-1 3D NUFFT from Leto views.
    pub fn execute_type1_3d_leto(
        &self,
        plan: &NufftWgpuPlan3D,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> NufftWgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 3>> {
        let positions = positions3_from_leto_view(positions)?;
        let values = leto_view1_cow(values);
        let output = self.execute_type1_3d(plan, &positions, values.as_ref())?;
        leto_array3_from_ndarray(&output)
    }

    /// Execute exact direct Type-1 3D NUFFT from typed Leto views.
    pub fn execute_type1_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan3D,
        precision: PrecisionProfile,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 3>> {
        let positions = positions3_from_leto_view(positions)?;
        let values = leto_view1_cow(values);
        let grid = plan.grid();
        let mut output = Array3::from_elem(
            [grid.nx, grid.ny, grid.nz],
            T::from_complex64(Complex64::new(0.0, 0.0)),
        );
        self.execute_type1_3d_typed_into(
            plan,
            precision,
            &positions,
            values.as_ref(),
            &mut output,
        )?;
        leto_array3_from_ndarray(&output)
    }
}
