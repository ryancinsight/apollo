use apollo_fft::PrecisionProfile;
use crate::NufftComplexStorage;
use leto::{Array1, Array3};
use eunomia::{Complex32, Complex64};

use crate::infrastructure::transport::gpu::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::device::helpers::{
    fast_1d_metadata, fast_3d_metadata, leto_array1_from_slice, leto_array3_from_ndarray,
    leto_view1_cow, positions3_from_leto_view, typed_to_complex32, validate_fast_1d_plan,
    validate_pair_lengths, validate_typed_profile, validate_usize_to_u32,
};
use crate::infrastructure::transport::gpu::infrastructure::device::NufftWgpuBackend;
use crate::infrastructure::transport::gpu::infrastructure::kernel::{NufftGpuBuffers1D, NufftGpuBuffers3D};

impl NufftWgpuBackend {
    /// Execute fast gridded Type-1 1D NUFFT on WGPU.
    pub fn execute_fast_type1_1d(
        &self,
        plan: &NufftWgpuPlan1D,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Array1<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_fast_1d_plan(plan)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_1d_metadata(plan)?;
        let output = self.kernel.execute_fast_type1_1d(
            self.device.device(),
            self.device.queue(),
            plan.domain().n,
            fast.oversampled_len,
            plan.kernel_width(),
            plan.domain().length() as f32,
            fast.beta as f32,
            fast.i0_beta as f32,
            &fast.deconv,
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

    /// Execute fast gridded Type-1 1D NUFFT with caller-owned typed storage.
    ///
    /// WGPU arithmetic remains `f32`. `Complex32` storage is passed through and
    /// mixed `[f16; 2]` storage is promoted once to represented `Complex32`
    /// before dispatch, then quantized back at the output boundary.
    pub fn execute_fast_type1_1d_typed_into<T: NufftComplexStorage>(
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
        let computed = self.execute_fast_type1_1d(plan, positions, &values32)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_complex64(value);
        }
        Ok(())
    }

    /// Execute fast gridded Type-1 1D NUFFT from Leto views.
    pub fn execute_fast_type1_1d_leto(
        &self,
        plan: &NufftWgpuPlan1D,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> NufftWgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let positions = leto_view1_cow(positions);
        let values = leto_view1_cow(values);
        let output = self.execute_fast_type1_1d(plan, positions.as_ref(), values.as_ref())?;
        leto_array1_from_slice(output.as_slice().ok_or(NufftWgpuError::InvalidPlan {
            message: "fast type1 1D Leto output must be contiguous",
        })?)
    }

    /// Execute fast gridded Type-1 1D NUFFT from typed Leto views.
    pub fn execute_fast_type1_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan1D,
        precision: PrecisionProfile,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let positions = leto_view1_cow(positions);
        let values = leto_view1_cow(values);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); plan.domain().n];
        self.execute_fast_type1_1d_typed_into(
            plan,
            precision,
            positions.as_ref(),
            values.as_ref(),
            &mut output,
        )?;
        leto_array1_from_slice(&output)
    }

    /// Execute fast gridded Type-1 3D NUFFT on WGPU.
    pub fn execute_fast_type1_3d(
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
        let fast = fast_3d_metadata(plan)?;
        let (lx, ly, lz) = grid.lengths();
        let output = self.kernel.execute_fast_type1_3d(
            self.device.device(),
            self.device.queue(),
            [grid.nx, grid.ny, grid.nz],
            [fast.mx, fast.my, fast.mz],
            plan.kernel_width(),
            (lx as f32, ly as f32, lz as f32),
            fast.beta as f32,
            fast.i0_beta as f32,
            &fast.deconv_xyz,
            positions,
            values,
        )?;
        let converted: Vec<Complex64> = output
            .into_iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .collect();
        Array3::from_shape_vec([grid.nx, grid.ny, grid.nz], converted).map_err(|_| {
            NufftWgpuError::InvalidPlan {
                message: "fast 3D type1 output shape does not match grid dimensions",
            }
        })
    }

    /// Execute fast gridded Type-1 3D NUFFT with caller-owned typed storage.
    pub fn execute_fast_type1_3d_typed_into<T: NufftComplexStorage>(
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
        let computed = self.execute_fast_type1_3d(plan, positions, &values32)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_complex64(value);
        }
        Ok(())
    }

    /// Execute fast gridded Type-1 3D NUFFT from Leto views.
    pub fn execute_fast_type1_3d_leto(
        &self,
        plan: &NufftWgpuPlan3D,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> NufftWgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 3>> {
        let positions = positions3_from_leto_view(positions)?;
        let values = leto_view1_cow(values);
        let output = self.execute_fast_type1_3d(plan, &positions, values.as_ref())?;
        leto_array3_from_ndarray(&output)
    }

    /// Execute fast gridded Type-1 3D NUFFT from typed Leto views.
    pub fn execute_fast_type1_3d_leto_typed<T: NufftComplexStorage>(
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
        self.execute_fast_type1_3d_typed_into(
            plan,
            precision,
            &positions,
            values.as_ref(),
            &mut output,
        )?;
        leto_array3_from_ndarray(&output)
    }

    /// Execute fast gridded Type-1 1D NUFFT with pre-allocated GPU buffers.
    pub fn execute_fast_type1_1d_with_buffers(
        &self,
        plan: &NufftWgpuPlan1D,
        buffers: &NufftGpuBuffers1D,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_fast_1d_plan(plan)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_1d_metadata(plan)?;
        let output = self.kernel.execute_fast_type1_1d_with_buffers(
            self.device.device(),
            self.device.queue(),
            buffers,
            plan.kernel_width(),
            plan.domain().length() as f32,
            fast.beta as f32,
            fast.i0_beta as f32,
            &fast.deconv,
            positions,
            values,
        )?;
        Ok(output
            .into_iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .collect())
    }

    /// Execute fast gridded Type-1 3D NUFFT with pre-allocated GPU buffers.
    pub fn execute_fast_type1_3d_with_buffers(
        &self,
        plan: &NufftWgpuPlan3D,
        buffers: &NufftGpuBuffers3D,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Array3<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        let grid = plan.grid();
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_3d_metadata(plan)?;
        let (lx, ly, lz) = grid.lengths();
        let output = self.kernel.execute_fast_type1_3d_with_buffers(
            self.device.device(),
            self.device.queue(),
            buffers,
            plan.kernel_width(),
            (lx as f32, ly as f32, lz as f32),
            fast.beta as f32,
            fast.i0_beta as f32,
            &fast.deconv_xyz,
            positions,
            values,
        )?;
        let converted: Vec<Complex64> = output
            .into_iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .collect();
        Array3::from_shape_vec([grid.nx, grid.ny, grid.nz], converted).map_err(|_| {
            NufftWgpuError::InvalidPlan {
                message: "fast 3D type1 with_buffers output shape does not match grid",
            }
        })
    }
}
