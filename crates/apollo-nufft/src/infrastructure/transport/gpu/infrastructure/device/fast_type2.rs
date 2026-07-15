use crate::NufftComplexStorage;
use apollo_fft::PrecisionProfile;
use eunomia::{Complex32, Complex64};
use leto::Array3;

use crate::infrastructure::transport::gpu::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::device::helpers::{
    array3_from_leto_view, fast_1d_metadata, fast_3d_metadata, leto_array1_from_slice,
    leto_view1_cow, positions3_from_leto_view, typed_to_complex32, validate_fast_1d_plan,
    validate_typed_profile, validate_usize_to_u32, write_typed_output,
};
use crate::infrastructure::transport::gpu::infrastructure::device::NufftWgpuBackend;
use crate::infrastructure::transport::gpu::infrastructure::kernel::{
    KaiserBesselOne, KaiserBesselThree, NufftGpuBuffers1D, NufftGpuBuffers3D, NufftGpuKernel,
};

impl NufftWgpuBackend {
    /// Execute fast gridded Type-2 1D NUFFT on WGPU.
    pub fn execute_fast_type2_1d(
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
        validate_fast_1d_plan(plan)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_1d_metadata(plan)?;
        let configuration = KaiserBesselOne {
            kernel_width: plan.kernel_width(),
            length: plan.domain().length() as f32,
            beta: fast.beta as f32,
            i0_beta: fast.i0_beta as f32,
            deconvolution: &fast.deconv,
        };
        let output = NufftGpuKernel::execute_fast_type2_1d(
            &self.device,
            plan.domain().n,
            fast.oversampled_len,
            configuration,
            fourier_coeffs,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect())
    }

    /// Execute fast gridded Type-2 1D NUFFT with caller-owned typed storage.
    pub fn execute_fast_type2_1d_typed_into<T: NufftComplexStorage>(
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
        let computed = self.execute_fast_type2_1d(plan, &coefficients32, positions)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    /// Execute fast gridded Type-2 1D NUFFT from Leto views.
    pub fn execute_fast_type2_1d_leto(
        &self,
        plan: &NufftWgpuPlan1D,
        fourier_coeffs: leto::ArrayView1<'_, Complex32>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> NufftWgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let fourier_coeffs = leto_view1_cow(fourier_coeffs);
        let positions = leto_view1_cow(positions);
        let output =
            self.execute_fast_type2_1d(plan, fourier_coeffs.as_ref(), positions.as_ref())?;
        leto_array1_from_slice(&output)
    }

    /// Execute fast gridded Type-2 1D NUFFT from typed Leto views.
    pub fn execute_fast_type2_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan1D,
        precision: PrecisionProfile,
        fourier_coeffs: leto::ArrayView1<'_, T>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let fourier_coeffs = leto_view1_cow(fourier_coeffs);
        let positions = leto_view1_cow(positions);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); positions.len()];
        self.execute_fast_type2_1d_typed_into(
            plan,
            precision,
            &fourier_coeffs,
            &positions,
            &mut output,
        )?;
        leto_array1_from_slice(&output)
    }

    /// Execute fast gridded Type-2 1D NUFFT and return debug grid snapshots.
    #[cfg(any(test, feature = "diagnostics"))]
    pub fn execute_fast_type2_1d_with_diagnostics(
        &self,
        plan: &NufftWgpuPlan1D,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<(
        Vec<Complex64>,
        crate::infrastructure::transport::gpu::NufftType2GridDiagnostics,
    )> {
        if fourier_coeffs.len() != plan.domain().n {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected: plan.domain().n,
                actual: fourier_coeffs.len(),
            });
        }
        validate_fast_1d_plan(plan)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_1d_metadata(plan)?;
        let buffers = crate::infrastructure::transport::gpu::NufftGpuBuffers1D::new(
            &self.device,
            plan.domain().n,
            fast.oversampled_len,
            positions.len(),
        )?;
        let configuration = KaiserBesselOne {
            kernel_width: plan.kernel_width(),
            length: plan.domain().length() as f32,
            beta: fast.beta as f32,
            i0_beta: fast.i0_beta as f32,
            deconvolution: &fast.deconv,
        };
        let (output, diagnostics) = NufftGpuKernel::execute_fast_type2_1d_with_diagnostics(
            &self.device,
            &buffers,
            configuration,
            fourier_coeffs,
            positions,
        )?;
        Ok((
            output
                .into_iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>(),
            diagnostics,
        ))
    }

    /// Execute fast gridded Type-2 3D NUFFT on WGPU.
    pub fn execute_fast_type2_3d(
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
        let fast = fast_3d_metadata(plan)?;
        let (lx, ly, lz) = grid.lengths();
        let flat_modes: Vec<Complex32> = modes.iter().copied().collect();
        let configuration = KaiserBesselThree {
            kernel_width: plan.kernel_width(),
            lengths: (lx as f32, ly as f32, lz as f32),
            beta: fast.beta as f32,
            i0_beta: fast.i0_beta as f32,
            deconvolution: &fast.deconv_xyz,
        };
        let output = NufftGpuKernel::execute_fast_type2_3d(
            &self.device,
            (grid.nx, grid.ny, grid.nz),
            (fast.mx, fast.my, fast.mz),
            configuration,
            &flat_modes,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .collect())
    }

    /// Execute fast gridded Type-2 3D NUFFT with caller-owned typed storage.
    pub fn execute_fast_type2_3d_typed_into<T: NufftComplexStorage>(
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
        let computed = self.execute_fast_type2_3d(plan, &modes32, positions)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    /// Execute fast gridded Type-2 3D NUFFT from Leto views.
    pub fn execute_fast_type2_3d_leto(
        &self,
        plan: &NufftWgpuPlan3D,
        modes: leto::ArrayView3<'_, Complex32>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> NufftWgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let modes = array3_from_leto_view(modes);
        let positions = positions3_from_leto_view(positions)?;
        let output = self.execute_fast_type2_3d(plan, &modes, &positions)?;
        leto_array1_from_slice(&output)
    }

    /// Execute fast gridded Type-2 3D NUFFT from typed Leto views.
    pub fn execute_fast_type2_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &NufftWgpuPlan3D,
        precision: PrecisionProfile,
        modes: leto::ArrayView3<'_, T>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let modes = array3_from_leto_view(modes);
        let positions = positions3_from_leto_view(positions)?;
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); positions.len()];
        self.execute_fast_type2_3d_typed_into(plan, precision, &modes, &positions, &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Execute fast gridded Type-2 3D NUFFT and return debug grid snapshots.
    #[cfg(any(test, feature = "diagnostics"))]
    pub fn execute_fast_type2_3d_with_diagnostics(
        &self,
        plan: &NufftWgpuPlan3D,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<(
        Vec<Complex64>,
        crate::infrastructure::transport::gpu::NufftType2GridDiagnostics,
    )> {
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
        let fast = fast_3d_metadata(plan)?;
        let (lx, ly, lz) = grid.lengths();
        let flat_modes: Vec<Complex32> = modes.iter().copied().collect();
        let buffers = crate::infrastructure::transport::gpu::NufftGpuBuffers3D::new(
            &self.device,
            (grid.nx, grid.ny, grid.nz),
            (fast.mx, fast.my, fast.mz),
            positions.len(),
        )?;
        let configuration = KaiserBesselThree {
            kernel_width: plan.kernel_width(),
            lengths: (lx as f32, ly as f32, lz as f32),
            beta: fast.beta as f32,
            i0_beta: fast.i0_beta as f32,
            deconvolution: &fast.deconv_xyz,
        };
        let (output, diagnostics) = NufftGpuKernel::execute_fast_type2_3d_with_diagnostics(
            &self.device,
            &buffers,
            configuration,
            &flat_modes,
            positions,
        )?;
        Ok((
            output
                .into_iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>(),
            diagnostics,
        ))
    }

    /// Execute fast gridded Type-2 1D NUFFT with pre-allocated GPU buffers.
    pub fn execute_fast_type2_1d_with_buffers(
        &self,
        plan: &NufftWgpuPlan1D,
        buffers: &NufftGpuBuffers1D,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex64>> {
        if fourier_coeffs.len() != plan.domain().n {
            return Err(NufftWgpuError::InputLengthMismatch {
                expected: plan.domain().n,
                actual: fourier_coeffs.len(),
            });
        }
        validate_fast_1d_plan(plan)?;
        validate_usize_to_u32(positions.len())?;
        let fast = fast_1d_metadata(plan)?;
        let configuration = KaiserBesselOne {
            kernel_width: plan.kernel_width(),
            length: plan.domain().length() as f32,
            beta: fast.beta as f32,
            i0_beta: fast.i0_beta as f32,
            deconvolution: &fast.deconv,
        };
        let output = NufftGpuKernel::execute_fast_type2_1d_with_buffers(
            &self.device,
            buffers,
            configuration,
            fourier_coeffs,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .collect())
    }

    /// Execute fast gridded Type-2 3D NUFFT with pre-allocated GPU buffers.
    pub fn execute_fast_type2_3d_with_buffers(
        &self,
        plan: &NufftWgpuPlan3D,
        buffers: &NufftGpuBuffers3D,
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
        let fast = fast_3d_metadata(plan)?;
        let (lx, ly, lz) = grid.lengths();
        let flat_modes: Vec<Complex32> = modes.iter().copied().collect();
        let configuration = KaiserBesselThree {
            kernel_width: plan.kernel_width(),
            lengths: (lx as f32, ly as f32, lz as f32),
            beta: fast.beta as f32,
            i0_beta: fast.i0_beta as f32,
            deconvolution: &fast.deconv_xyz,
        };
        let output = NufftGpuKernel::execute_fast_type2_3d_with_buffers(
            &self.device,
            buffers,
            configuration,
            &flat_modes,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .collect())
    }
}
