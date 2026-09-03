use crate::NufftComplexStorage;
use apollo_fft::PrecisionProfile;
use eunomia::{Complex32, Complex64};
use leto::{Array1, Array3};

use crate::infrastructure::transport::gpu::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::device::conversion::{
    host_array_error, positions3_from_leto_view, typed_to_complex32, validate_pair_lengths,
    validate_typed_profile, validate_usize_to_u32, write_complex_output,
};
use crate::infrastructure::transport::gpu::infrastructure::device::NufftWgpuBackend;
use crate::infrastructure::transport::gpu::infrastructure::kernel::{
    NufftGpuBuffers1D, NufftGpuBuffers3D, NufftGpuKernel,
};

impl NufftWgpuBackend {
    /// Execute fast gridded Type-1 1D NUFFT on WGPU.
    pub fn execute_fast_type1_1d(
        &self,
        plan: &NufftWgpuPlan1D,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Array1<Complex64>> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(positions.len())?;
        let mut buffers = NufftGpuBuffers1D::new(&self.device, plan, positions.len())?;
        let mut output = vec![Complex64::new(0.0, 0.0); plan.domain().n];
        self.execute_fast_type1_1d_with_buffers(&mut buffers, positions, values, &mut output)?;
        Ok(Array1::from(output))
    }

    /// Execute fast gridded Type-1 1D NUFFT with caller-owned typed storage.
    ///
    /// WGPU arithmetic remains `f32`. `Complex32` storage is passed through and
    /// mixed `[F16; 2]` storage is promoted once to represented `Complex32`
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
            *slot = T::from_cpu(value);
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
        let positions = apollo_leto_interop::view_cow(&positions);
        let values = apollo_leto_interop::view_cow(&values);
        let output = self.execute_fast_type1_1d(plan, positions.as_ref(), values.as_ref())?;
        apollo_leto_interop::try_array1_from_slice(output.as_slice().ok_or(
            NufftWgpuError::InvalidPlan {
                message: "fast type1 1D Leto output must be contiguous",
            },
        )?)
        .ok_or_else(host_array_error)
    }

    /// Execute fast gridded Type-1 1D NUFFT from typed Leto views.
    pub fn execute_fast_type1_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan1D,
        precision: PrecisionProfile,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let positions = apollo_leto_interop::view_cow(&positions);
        let values = apollo_leto_interop::view_cow(&values);
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); plan.domain().n];
        self.execute_fast_type1_1d_typed_into(
            plan,
            precision,
            positions.as_ref(),
            values.as_ref(),
            &mut output,
        )?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
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
        let mut buffers = NufftGpuBuffers3D::new(&self.device, plan, positions.len())?;
        let output_len = grid
            .nx
            .checked_mul(grid.ny)
            .and_then(|value| value.checked_mul(grid.nz))
            .ok_or(NufftWgpuError::InvalidPlan {
                message: "fast 3D type1 output length overflows usize",
            })?;
        let mut output = vec![Complex64::new(0.0, 0.0); output_len];
        self.execute_fast_type1_3d_with_buffers(&mut buffers, positions, values, &mut output)?;
        Array3::from_shape_vec([grid.nx, grid.ny, grid.nz], output).map_err(|_| {
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
        if output.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(NufftWgpuError::InvalidPlan {
                message: "typed output shape must match 3D plan grid dimensions",
            });
        }
        let values32 = typed_to_complex32(values);
        let computed = self.execute_fast_type1_3d(plan, positions, &values32)?;
        for (slot, value) in output
            .as_slice_mut()
            .expect("contiguous")
            .iter_mut()
            .zip(computed.iter().copied())
        {
            *slot = T::from_cpu(value);
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
        let values = apollo_leto_interop::view_cow(&values);
        let output = self.execute_fast_type1_3d(plan, &positions, values.as_ref())?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(|| {
            NufftWgpuError::HostArrayLayout {
                message: "failed to allocate Mnemosyne-backed Leto NUFFT-WGPU 3D output".to_owned(),
            }
        })
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
        let values = apollo_leto_interop::view_cow(&values);
        let grid = plan.grid();
        let mut output = Array3::from_elem(
            [grid.nx, grid.ny, grid.nz],
            T::from_cpu(Complex64::new(0.0, 0.0)),
        );
        self.execute_fast_type1_3d_typed_into(
            plan,
            precision,
            &positions,
            values.as_ref(),
            &mut output,
        )?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(|| {
            NufftWgpuError::HostArrayLayout {
                message: "failed to allocate Mnemosyne-backed Leto NUFFT-WGPU 3D output".to_owned(),
            }
        })
    }

    /// Execute fast gridded Type-1 1D NUFFT with exclusively borrowed pre-allocated buffers.
    pub fn execute_fast_type1_1d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers1D,
        positions: &[f32],
        values: &[Complex32],
        output: &mut [Complex64],
    ) -> NufftWgpuResult<()> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(positions.len())?;
        if output.len() != buffers.n {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected: buffers.n,
                actual: output.len(),
            });
        }
        buffers.validate_device(&self.device)?;
        NufftGpuKernel::execute_fast_type1_1d_with_buffers(
            &self.device,
            buffers,
            positions,
            values,
        )?;
        write_complex_output(buffers.readback_prefix(output.len())?, output);
        Ok(())
    }

    /// Execute fast gridded Type-1 3D NUFFT with exclusively borrowed pre-allocated buffers.
    pub fn execute_fast_type1_3d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers3D,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
        output: &mut [Complex64],
    ) -> NufftWgpuResult<()> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(positions.len())?;
        let expected = buffers
            .shape
            .0
            .checked_mul(buffers.shape.1)
            .and_then(|value| value.checked_mul(buffers.shape.2))
            .ok_or(NufftWgpuError::InvalidPlan {
                message: "fast 3D type1 output length overflows usize",
            })?;
        if output.len() != expected {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected,
                actual: output.len(),
            });
        }
        buffers.validate_device(&self.device)?;
        NufftGpuKernel::execute_fast_type1_3d_with_buffers(
            &self.device,
            buffers,
            positions,
            values,
        )?;
        write_complex_output(buffers.readback_prefix(output.len())?, output);
        Ok(())
    }
}
