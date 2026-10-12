//! Direct-sum and gridded NUFFT execution over the shared WGPU backend.

use crate::NufftComplexStorage;
use apollo_fft::{PrecisionProfile, WgpuError, WgpuResult};
use eunomia::{Complex32, Complex64};
use hephaestus_wgpu::{DeviceLimits, WgpuDevice};
use leto::{Array1, Array3};

use super::conversion::{
    array3_from_leto_view, host_array_error, host_dense_error, positions3_from_leto_view,
    typed_to_complex32, validate_pair_lengths, validate_typed_profile, validate_usize_to_u32,
    write_complex_output, write_typed_output,
};
use super::infrastructure::kernel::{NufftGpuBuffers1D, NufftGpuBuffers3D, NufftGpuKernel};
use super::surface::{DirectTransform, GriddedTransform};
use super::NufftWgpuBackend;

/// Fast NUFFT shaders bind storage buffers `0..=6` in one shader stage.
const FAST_NUFFT_STORAGE_BINDINGS: u32 = 7;

/// Return the Hephaestus limits required by the fast NUFFT kernels.
///
/// The fast one- and three-dimensional shader descriptors bind seven
/// storage buffers in one stage. The returned request preserves the
/// provider default limits and raises only that lower bound.
#[must_use]
pub fn required_device_limits() -> DeviceLimits {
    let mut limits = WgpuDevice::default_device_limits();
    limits.max_storage_buffers_per_shader_stage = Some(FAST_NUFFT_STORAGE_BINDINGS);
    limits
}

impl DirectTransform for NufftWgpuBackend {
    fn execute_type1_1d(
        &self,
        plan: &super::NufftWgpuPlan,
        positions: &[f32],
        values: &[Complex32],
    ) -> WgpuResult<Array1<Complex64>> {
        NufftWgpuBackend::validate_plan(plan)?;
        let domain = plan.payload().domain()?;
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(domain.n)?;
        validate_usize_to_u32(positions.len())?;
        let output = NufftGpuKernel::execute_type1_1d(
            self.device(),
            domain.n,
            domain.length() as f32,
            positions,
            values,
        )?;
        Ok(Array1::from(
            output
                .into_iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>(),
        ))
    }

    fn execute_type1_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: &[f32],
        values: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        let domain = plan.payload().domain()?;
        validate_typed_profile::<T>(precision)?;
        if output.len() != domain.n {
            return Err(WgpuError::LengthMismatch {
                expected: domain.n,
                actual: output.len(),
            });
        }
        let values32 = typed_to_complex32(values);
        let computed = self.execute_type1_1d(plan, positions, &values32)?;
        write_typed_output(computed.as_slice().unwrap_or(&[]), output);
        Ok(())
    }

    fn execute_type1_1d_leto(
        &self,
        plan: &super::NufftWgpuPlan,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let positions = apollo_leto_interop::view_cow(&positions);
        let values = apollo_leto_interop::view_cow(&values);
        let output = self.execute_type1_1d(plan, positions.as_ref(), values.as_ref())?;
        apollo_leto_interop::try_array1_from_slice(output.as_slice().ok_or_else(|| {
            WgpuError::InvalidPlan {
                message: "type1 1D Leto output must be contiguous".to_owned(),
            }
        })?)
        .ok_or_else(host_array_error)
    }

    fn execute_type1_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let positions = apollo_leto_interop::view_cow(&positions);
        let values = apollo_leto_interop::view_cow(&values);
        let n = plan.payload().domain()?.n;
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); n];
        self.execute_type1_1d_typed_into(
            plan,
            precision,
            positions.as_ref(),
            values.as_ref(),
            &mut output,
        )?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }

    fn execute_type1_3d(
        &self,
        plan: &super::NufftWgpuPlan,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> WgpuResult<Array3<Complex64>> {
        NufftWgpuBackend::validate_plan(plan)?;
        let grid = plan.payload().grid()?;
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let (lx, ly, lz) = grid.lengths();
        let output = NufftGpuKernel::execute_type1_3d(
            self.device(),
            (grid.nx, grid.ny, grid.nz),
            (lx as f32, ly as f32, lz as f32),
            positions,
            values,
        )?;
        let converted: Vec<Complex64> = output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        Array3::from_shape_vec([grid.nx, grid.ny, grid.nz], converted).map_err(|_| {
            WgpuError::ShapeMismatch {
                message: "3D output shape does not match grid dimensions".to_owned(),
            }
        })
    }

    fn execute_type1_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: &[(f32, f32, f32)],
        values: &[T],
        output: &mut Array3<T>,
    ) -> WgpuResult<()> {
        let grid = plan.payload().grid()?;
        validate_typed_profile::<T>(precision)?;
        if output.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(WgpuError::ShapeMismatch {
                message: "typed output shape must match 3D plan grid dimensions".to_owned(),
            });
        }
        let values32 = typed_to_complex32(values);
        let computed = self.execute_type1_3d(plan, positions, &values32)?;
        for (slot, value) in output
            .as_slice_mut()
            .expect("invariant: Leto Array3 exposes contiguous mutable storage")
            .iter_mut()
            .zip(computed.iter().copied())
        {
            *slot = T::from_cpu(value);
        }
        Ok(())
    }

    fn execute_type1_3d_leto(
        &self,
        plan: &super::NufftWgpuPlan,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 3>> {
        let positions = positions3_from_leto_view(positions)?;
        let values = apollo_leto_interop::view_cow(&values);
        let output = self.execute_type1_3d(plan, &positions, values.as_ref())?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(host_dense_error)
    }

    fn execute_type1_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 3>> {
        let positions = positions3_from_leto_view(positions)?;
        let values = apollo_leto_interop::view_cow(&values);
        let grid = plan.payload().grid()?;
        let mut output = Array3::from_elem(
            [grid.nx, grid.ny, grid.nz],
            T::from_cpu(Complex64::new(0.0, 0.0)),
        );
        self.execute_type1_3d_typed_into(
            plan,
            precision,
            &positions,
            values.as_ref(),
            &mut output,
        )?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(host_dense_error)
    }

    fn execute_type2_1d(
        &self,
        plan: &super::NufftWgpuPlan,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> WgpuResult<Vec<Complex64>> {
        NufftWgpuBackend::validate_plan(plan)?;
        let domain = plan.payload().domain()?;
        if fourier_coeffs.len() != domain.n {
            return Err(WgpuError::LengthMismatch {
                expected: domain.n,
                actual: fourier_coeffs.len(),
            });
        }
        validate_usize_to_u32(domain.n)?;
        validate_usize_to_u32(positions.len())?;
        let output = NufftGpuKernel::execute_type2_1d(
            self.device(),
            domain.n,
            domain.length() as f32,
            fourier_coeffs,
            positions,
        )?;
        Ok(output
            .into_iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect())
    }

    fn execute_type2_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        fourier_coeffs: &[T],
        positions: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        validate_typed_profile::<T>(precision)?;
        if output.len() != positions.len() {
            return Err(WgpuError::LengthMismatch {
                expected: positions.len(),
                actual: output.len(),
            });
        }
        let coefficients32 = typed_to_complex32(fourier_coeffs);
        let computed = self.execute_type2_1d(plan, &coefficients32, positions)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    fn execute_type2_1d_leto(
        &self,
        plan: &super::NufftWgpuPlan,
        fourier_coeffs: leto::ArrayView1<'_, Complex32>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let fourier_coeffs = apollo_leto_interop::view_cow(&fourier_coeffs);
        let positions = apollo_leto_interop::view_cow(&positions);
        let output = self.execute_type2_1d(plan, fourier_coeffs.as_ref(), positions.as_ref())?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }

    fn execute_type2_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        fourier_coeffs: leto::ArrayView1<'_, T>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let fourier_coeffs = apollo_leto_interop::view_cow(&fourier_coeffs);
        let positions = apollo_leto_interop::view_cow(&positions);
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); positions.len()];
        self.execute_type2_1d_typed_into(
            plan,
            precision,
            fourier_coeffs.as_ref(),
            positions.as_ref(),
            &mut output,
        )?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }

    fn execute_type2_3d(
        &self,
        plan: &super::NufftWgpuPlan,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> WgpuResult<Vec<Complex64>> {
        NufftWgpuBackend::validate_plan(plan)?;
        let grid = plan.payload().grid()?;
        if modes.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(WgpuError::ShapeMismatch {
                message: "mode shape must match 3D plan grid dimensions".to_owned(),
            });
        }
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let (lx, ly, lz) = grid.lengths();
        let coefficients: Vec<Complex32> = modes.iter().copied().collect();
        let output = NufftGpuKernel::execute_type2_3d(
            self.device(),
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

    fn execute_type2_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        modes: &Array3<T>,
        positions: &[(f32, f32, f32)],
        output: &mut [T],
    ) -> WgpuResult<()> {
        validate_typed_profile::<T>(precision)?;
        if output.len() != positions.len() {
            return Err(WgpuError::LengthMismatch {
                expected: positions.len(),
                actual: output.len(),
            });
        }
        let modes32 = modes.mapv(|value| {
            let represented = value.to_cpu();
            Complex32::new(represented.re as f32, represented.im as f32)
        });
        let computed = self.execute_type2_3d(plan, &modes32, positions)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    fn execute_type2_3d_leto(
        &self,
        plan: &super::NufftWgpuPlan,
        modes: leto::ArrayView3<'_, Complex32>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let modes = array3_from_leto_view(modes);
        let positions = positions3_from_leto_view(positions)?;
        let output = self.execute_type2_3d(plan, &modes, &positions)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }

    fn execute_type2_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        modes: leto::ArrayView3<'_, T>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let modes = array3_from_leto_view(modes);
        let positions = positions3_from_leto_view(positions)?;
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); positions.len()];
        self.execute_type2_3d_typed_into(plan, precision, &modes, &positions, &mut output)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }
}

impl GriddedTransform for NufftWgpuBackend {
    fn execute_fast_type1_1d(
        &self,
        plan: &super::NufftWgpuPlan,
        positions: &[f32],
        values: &[Complex32],
    ) -> WgpuResult<Array1<Complex64>> {
        NufftWgpuBackend::validate_plan(plan)?;
        let domain = plan.payload().domain()?;
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(positions.len())?;
        let mut buffers = NufftGpuBuffers1D::new(self.device(), plan, positions.len())?;
        let mut output = vec![Complex64::new(0.0, 0.0); domain.n];
        self.execute_fast_type1_1d_with_buffers(&mut buffers, positions, values, &mut output)?;
        Ok(Array1::from(output))
    }

    fn execute_fast_type1_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: &[f32],
        values: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        let domain = plan.payload().domain()?;
        validate_typed_profile::<T>(precision)?;
        if output.len() != domain.n {
            return Err(WgpuError::LengthMismatch {
                expected: domain.n,
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

    fn execute_fast_type1_1d_leto(
        &self,
        plan: &super::NufftWgpuPlan,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let positions = apollo_leto_interop::view_cow(&positions);
        let values = apollo_leto_interop::view_cow(&values);
        let output = self.execute_fast_type1_1d(plan, positions.as_ref(), values.as_ref())?;
        apollo_leto_interop::try_array1_from_slice(output.as_slice().ok_or_else(|| {
            WgpuError::InvalidPlan {
                message: "fast type1 1D Leto output must be contiguous".to_owned(),
            }
        })?)
        .ok_or_else(host_array_error)
    }

    fn execute_fast_type1_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: leto::ArrayView1<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let positions = apollo_leto_interop::view_cow(&positions);
        let values = apollo_leto_interop::view_cow(&values);
        let n = plan.payload().domain()?.n;
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); n];
        self.execute_fast_type1_1d_typed_into(
            plan,
            precision,
            positions.as_ref(),
            values.as_ref(),
            &mut output,
        )?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }

    fn execute_fast_type1_1d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers1D,
        positions: &[f32],
        values: &[Complex32],
        output: &mut [Complex64],
    ) -> WgpuResult<()> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(positions.len())?;
        if output.len() != buffers.n {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.n,
                actual: output.len(),
            });
        }
        buffers.validate_device(self.device())?;
        NufftGpuKernel::execute_fast_type1_1d_with_buffers(
            self.device(),
            buffers,
            positions,
            values,
        )?;
        write_complex_output(buffers.readback_prefix(output.len())?, output);
        Ok(())
    }

    fn execute_fast_type1_3d(
        &self,
        plan: &super::NufftWgpuPlan,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> WgpuResult<Array3<Complex64>> {
        NufftWgpuBackend::validate_plan(plan)?;
        let grid = plan.payload().grid()?;
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let mut buffers = NufftGpuBuffers3D::new(self.device(), plan, positions.len())?;
        let output_len = grid
            .nx
            .checked_mul(grid.ny)
            .and_then(|value| value.checked_mul(grid.nz))
            .ok_or_else(|| WgpuError::InvalidPlan {
                message: "fast 3D type1 output length overflows usize".to_owned(),
            })?;
        let mut output = vec![Complex64::new(0.0, 0.0); output_len];
        self.execute_fast_type1_3d_with_buffers(&mut buffers, positions, values, &mut output)?;
        Array3::from_shape_vec([grid.nx, grid.ny, grid.nz], output).map_err(|_| {
            WgpuError::ShapeMismatch {
                message: "fast 3D type1 output shape does not match grid dimensions".to_owned(),
            }
        })
    }

    fn execute_fast_type1_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: &[(f32, f32, f32)],
        values: &[T],
        output: &mut Array3<T>,
    ) -> WgpuResult<()> {
        let grid = plan.payload().grid()?;
        validate_typed_profile::<T>(precision)?;
        if output.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(WgpuError::ShapeMismatch {
                message: "typed output shape must match 3D plan grid dimensions".to_owned(),
            });
        }
        let values32 = typed_to_complex32(values);
        let computed = self.execute_fast_type1_3d(plan, positions, &values32)?;
        for (slot, value) in output
            .as_slice_mut()
            .expect("invariant: Leto Array3 exposes contiguous mutable storage")
            .iter_mut()
            .zip(computed.iter().copied())
        {
            *slot = T::from_cpu(value);
        }
        Ok(())
    }

    fn execute_fast_type1_3d_leto(
        &self,
        plan: &super::NufftWgpuPlan,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 3>> {
        let positions = positions3_from_leto_view(positions)?;
        let values = apollo_leto_interop::view_cow(&values);
        let output = self.execute_fast_type1_3d(plan, &positions, values.as_ref())?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(host_dense_error)
    }

    fn execute_fast_type1_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        positions: leto::ArrayView2<'_, f32>,
        values: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 3>> {
        let positions = positions3_from_leto_view(positions)?;
        let values = apollo_leto_interop::view_cow(&values);
        let grid = plan.payload().grid()?;
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
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(host_dense_error)
    }

    fn execute_fast_type1_3d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers3D,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
        output: &mut [Complex64],
    ) -> WgpuResult<()> {
        validate_pair_lengths(positions.len(), values.len())?;
        validate_usize_to_u32(positions.len())?;
        let expected = buffers
            .shape
            .0
            .checked_mul(buffers.shape.1)
            .and_then(|value| value.checked_mul(buffers.shape.2))
            .ok_or_else(|| WgpuError::InvalidPlan {
                message: "fast 3D type1 output length overflows usize".to_owned(),
            })?;
        if output.len() != expected {
            return Err(WgpuError::LengthMismatch {
                expected,
                actual: output.len(),
            });
        }
        buffers.validate_device(self.device())?;
        NufftGpuKernel::execute_fast_type1_3d_with_buffers(
            self.device(),
            buffers,
            positions,
            values,
        )?;
        write_complex_output(buffers.readback_prefix(output.len())?, output);
        Ok(())
    }

    fn execute_fast_type2_1d(
        &self,
        plan: &super::NufftWgpuPlan,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> WgpuResult<Vec<Complex64>> {
        NufftWgpuBackend::validate_plan(plan)?;
        let domain = plan.payload().domain()?;
        if fourier_coeffs.len() != domain.n {
            return Err(WgpuError::LengthMismatch {
                expected: domain.n,
                actual: fourier_coeffs.len(),
            });
        }
        validate_usize_to_u32(positions.len())?;
        let mut buffers = NufftGpuBuffers1D::new(self.device(), plan, positions.len())?;
        let mut output = vec![Complex64::new(0.0, 0.0); positions.len()];
        self.execute_fast_type2_1d_with_buffers(
            &mut buffers,
            fourier_coeffs,
            positions,
            &mut output,
        )?;
        Ok(output)
    }

    fn execute_fast_type2_1d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        fourier_coeffs: &[T],
        positions: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        validate_typed_profile::<T>(precision)?;
        if output.len() != positions.len() {
            return Err(WgpuError::LengthMismatch {
                expected: positions.len(),
                actual: output.len(),
            });
        }
        let coefficients32 = typed_to_complex32(fourier_coeffs);
        let computed = self.execute_fast_type2_1d(plan, &coefficients32, positions)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    fn execute_fast_type2_1d_leto(
        &self,
        plan: &super::NufftWgpuPlan,
        fourier_coeffs: leto::ArrayView1<'_, Complex32>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let fourier_coeffs = apollo_leto_interop::view_cow(&fourier_coeffs);
        let positions = apollo_leto_interop::view_cow(&positions);
        let output =
            self.execute_fast_type2_1d(plan, fourier_coeffs.as_ref(), positions.as_ref())?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }

    fn execute_fast_type2_1d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        fourier_coeffs: leto::ArrayView1<'_, T>,
        positions: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let fourier_coeffs = apollo_leto_interop::view_cow(&fourier_coeffs);
        let positions = apollo_leto_interop::view_cow(&positions);
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); positions.len()];
        self.execute_fast_type2_1d_typed_into(
            plan,
            precision,
            fourier_coeffs.as_ref(),
            positions.as_ref(),
            &mut output,
        )?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }

    fn execute_fast_type2_1d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers1D,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
        output: &mut [Complex64],
    ) -> WgpuResult<()> {
        if fourier_coeffs.len() != buffers.n {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.n,
                actual: fourier_coeffs.len(),
            });
        }
        if output.len() != positions.len() {
            return Err(WgpuError::LengthMismatch {
                expected: positions.len(),
                actual: output.len(),
            });
        }
        validate_usize_to_u32(positions.len())?;
        buffers.validate_device(self.device())?;
        buffers.write_coefficients(self.device(), fourier_coeffs)?;
        NufftGpuKernel::execute_fast_type2_1d_with_buffers(self.device(), buffers, positions)?;
        write_complex_output(buffers.readback_prefix(output.len())?, output);
        Ok(())
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn execute_fast_type2_1d_with_diagnostics(
        &self,
        plan: &super::NufftWgpuPlan,
        fourier_coeffs: &[Complex32],
        positions: &[f32],
    ) -> WgpuResult<(Vec<Complex64>, super::NufftType2GridDiagnostics)> {
        let domain = plan.payload().domain()?;
        if fourier_coeffs.len() != domain.n {
            return Err(WgpuError::LengthMismatch {
                expected: domain.n,
                actual: fourier_coeffs.len(),
            });
        }
        validate_usize_to_u32(positions.len())?;
        let mut buffers = NufftGpuBuffers1D::new(self.device(), plan, positions.len())?;
        buffers.write_coefficients(self.device(), fourier_coeffs)?;
        let diagnostics = NufftGpuKernel::execute_fast_type2_1d_with_diagnostics(
            self.device(),
            &mut buffers,
            positions,
        )?;
        let output = buffers
            .readback_prefix(positions.len())?
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        Ok((output, diagnostics))
    }

    fn execute_fast_type2_3d(
        &self,
        plan: &super::NufftWgpuPlan,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> WgpuResult<Vec<Complex64>> {
        NufftWgpuBackend::validate_plan(plan)?;
        let grid = plan.payload().grid()?;
        if modes.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(WgpuError::ShapeMismatch {
                message: "mode shape must match 3D plan grid dimensions".to_owned(),
            });
        }
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let mut buffers = NufftGpuBuffers3D::new(self.device(), plan, positions.len())?;
        let mut output = vec![Complex64::new(0.0, 0.0); positions.len()];
        self.execute_fast_type2_3d_with_buffers(&mut buffers, modes, positions, &mut output)?;
        Ok(output)
    }

    fn execute_fast_type2_3d_typed_into<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        modes: &Array3<T>,
        positions: &[(f32, f32, f32)],
        output: &mut [T],
    ) -> WgpuResult<()> {
        validate_typed_profile::<T>(precision)?;
        if output.len() != positions.len() {
            return Err(WgpuError::LengthMismatch {
                expected: positions.len(),
                actual: output.len(),
            });
        }
        let modes32 = modes.mapv(|value| {
            let represented = value.to_cpu();
            Complex32::new(represented.re as f32, represented.im as f32)
        });
        let computed = self.execute_fast_type2_3d(plan, &modes32, positions)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    fn execute_fast_type2_3d_leto(
        &self,
        plan: &super::NufftWgpuPlan,
        modes: leto::ArrayView3<'_, Complex32>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let modes = array3_from_leto_view(modes);
        let positions = positions3_from_leto_view(positions)?;
        let output = self.execute_fast_type2_3d(plan, &modes, &positions)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }

    fn execute_fast_type2_3d_leto_typed<T: NufftComplexStorage>(
        &self,
        plan: &super::NufftWgpuPlan,
        precision: PrecisionProfile,
        modes: leto::ArrayView3<'_, T>,
        positions: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let modes = array3_from_leto_view(modes);
        let positions = positions3_from_leto_view(positions)?;
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); positions.len()];
        self.execute_fast_type2_3d_typed_into(plan, precision, &modes, &positions, &mut output)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(host_array_error)
    }

    fn execute_fast_type2_3d_with_buffers(
        &self,
        buffers: &mut NufftGpuBuffers3D,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
        output: &mut [Complex64],
    ) -> WgpuResult<()> {
        if modes.shape() != [buffers.shape.0, buffers.shape.1, buffers.shape.2] {
            return Err(WgpuError::ShapeMismatch {
                message: "mode shape must match reusable 3D buffer dimensions".to_owned(),
            });
        }
        if output.len() != positions.len() {
            return Err(WgpuError::LengthMismatch {
                expected: positions.len(),
                actual: output.len(),
            });
        }
        validate_usize_to_u32(positions.len())?;
        buffers.validate_device(self.device())?;
        buffers.write_coefficients(self.device(), modes)?;
        NufftGpuKernel::execute_fast_type2_3d_with_buffers(self.device(), buffers, positions)?;
        write_complex_output(buffers.readback_prefix(output.len())?, output);
        Ok(())
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn execute_fast_type2_3d_with_diagnostics(
        &self,
        plan: &super::NufftWgpuPlan,
        modes: &Array3<Complex32>,
        positions: &[(f32, f32, f32)],
    ) -> WgpuResult<(Vec<Complex64>, super::NufftType2GridDiagnostics)> {
        let grid = plan.payload().grid()?;
        if modes.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(WgpuError::ShapeMismatch {
                message: "mode shape must match 3D plan grid dimensions".to_owned(),
            });
        }
        validate_usize_to_u32(grid.nx)?;
        validate_usize_to_u32(grid.ny)?;
        validate_usize_to_u32(grid.nz)?;
        validate_usize_to_u32(positions.len())?;
        let mut buffers = NufftGpuBuffers3D::new(self.device(), plan, positions.len())?;
        buffers.write_coefficients(self.device(), modes)?;
        let diagnostics = NufftGpuKernel::execute_fast_type2_3d_with_diagnostics(
            self.device(),
            &mut buffers,
            positions,
        )?;
        let output = buffers
            .readback_prefix(positions.len())?
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        Ok((output, diagnostics))
    }
}

#[cfg(test)]
mod tests {
    use super::required_device_limits;

    #[test]
    fn fast_nufft_limit_matches_shader_storage_bindings() {
        let limits = required_device_limits();
        assert_eq!(limits.max_storage_buffers_per_shader_stage, Some(7));
    }
}
