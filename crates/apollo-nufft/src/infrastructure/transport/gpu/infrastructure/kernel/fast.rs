//! Kaiser--Bessel spread/FFT/extract and load/IFFT/interpolate dispatch.

use eunomia::Complex32;
use hephaestus_core::{CommandStream, KernelDevice};
use hephaestus_wgpu::WgpuDevice;

use super::{
    buffers::{ensure_sample_capacity, NufftGpuBuffers1D, NufftGpuBuffers3D},
    descriptors::{FastNufftParams, FastNufftParams3D},
    NufftGpuKernel,
};
use crate::infrastructure::transport::gpu::domain::error::NufftWgpuResult;

use super::fast_support::{
    copy_positions_as_complex, copy_positions_as_pod, copy_real_as_complex, download_prefix, grid,
    one_bindings, product, three_bindings, write_one_type1_buffers, write_three_type1_buffers,
    KaiserBesselOne, KaiserBesselThree,
};

#[cfg(any(test, feature = "diagnostics"))]
use super::fast_support::{snapshot_one, snapshot_three};

#[cfg(any(test, feature = "diagnostics"))]
use super::buffers::NufftType2GridDiagnostics;

impl NufftGpuKernel {
    pub(crate) fn execute_fast_type1_1d(
        device: &WgpuDevice,
        n: usize,
        m: usize,
        configuration: KaiserBesselOne<'_>,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let mut buffers = NufftGpuBuffers1D::new(device, n, m, positions.len())?;
        Self::execute_fast_type1_1d_with_buffers(
            device,
            &mut buffers,
            configuration,
            positions,
            values,
        )
    }

    pub(crate) fn execute_fast_type1_1d_with_buffers(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers1D,
        configuration: KaiserBesselOne<'_>,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_complex(&mut buffers.host_positions, positions);
        copy_real_as_complex(
            &mut buffers.host_deconvolution,
            configuration.deconvolution,
            1.0,
        );
        write_one_type1_buffers(
            device,
            buffers,
            &buffers.host_positions,
            values,
            &buffers.host_deconvolution,
        )?;
        let params =
            FastNufftParams::for_grid(buffers.n, buffers.m, positions.len(), configuration)?;
        let bindings = one_bindings(buffers, &buffers.padding_buffer);
        let mut stream = device.stream()?;
        stream.encode(
            &buffers.kernels.spread,
            &bindings,
            &params,
            grid(buffers.m)?,
        )?;
        buffers.encode_forward(device, &mut stream)?;
        stream.encode(
            &buffers.kernels.extract,
            &bindings,
            &params,
            grid(buffers.n)?,
        )?;
        stream.submit()?;
        download_prefix(device, &buffers.output_buffer, buffers.n)
    }

    pub(crate) fn execute_fast_type2_1d(
        device: &WgpuDevice,
        n: usize,
        m: usize,
        configuration: KaiserBesselOne<'_>,
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let mut buffers = NufftGpuBuffers1D::new(device, n, m, positions.len())?;
        Self::execute_fast_type2_1d_with_buffers(
            device,
            &mut buffers,
            configuration,
            coefficients,
            positions,
        )
    }

    pub(crate) fn execute_fast_type2_1d_with_buffers(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers1D,
        configuration: KaiserBesselOne<'_>,
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_complex(&mut buffers.host_positions, positions);
        copy_real_as_complex(
            &mut buffers.host_deconvolution,
            configuration.deconvolution,
            buffers.m as f32,
        );
        device.write_sub_buffer(&buffers.position_buffer, 0, &buffers.host_positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, &buffers.host_deconvolution)?;
        device.write_sub_buffer(&buffers.coefficient_buffer, 0, coefficients)?;
        let params =
            FastNufftParams::for_grid(buffers.n, buffers.m, positions.len(), configuration)?;
        let load_bindings = one_bindings(buffers, &buffers.coefficient_buffer);
        let interpolate_bindings = one_bindings(buffers, &buffers.coefficient_buffer);
        let mut stream = device.stream()?;
        stream.encode(
            &buffers.kernels.load,
            &load_bindings,
            &params,
            grid(buffers.m)?,
        )?;
        buffers.encode_inverse(device, &mut stream)?;
        stream.encode(
            &buffers.kernels.interpolate,
            &interpolate_bindings,
            &params,
            grid(positions.len())?,
        )?;
        stream.submit()?;
        download_prefix(device, &buffers.output_buffer, positions.len())
    }

    pub(crate) fn execute_fast_type1_3d(
        device: &WgpuDevice,
        shape: (usize, usize, usize),
        oversampled: (usize, usize, usize),
        configuration: KaiserBesselThree<'_>,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let mut buffers = NufftGpuBuffers3D::new(device, shape, oversampled, positions.len())?;
        Self::execute_fast_type1_3d_with_buffers(
            device,
            &mut buffers,
            configuration,
            positions,
            values,
        )
    }

    pub(crate) fn execute_fast_type1_3d_with_buffers(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers3D,
        configuration: KaiserBesselThree<'_>,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_pod(&mut buffers.host_positions, positions);
        write_three_type1_buffers(
            device,
            buffers,
            &buffers.host_positions,
            values,
            configuration.deconvolution,
        )?;
        let params = FastNufftParams3D::for_grid(
            buffers.shape,
            buffers.oversampled,
            positions.len(),
            configuration,
        )?;
        let bindings = three_bindings(buffers, &buffers.padding_buffer);
        let grid_len = product(buffers.oversampled)?;
        let output_len = product(buffers.shape)?;
        let mut stream = device.stream()?;
        stream.encode(&buffers.kernels.spread, &bindings, &params, grid(grid_len)?)?;
        buffers.encode_forward(device, &mut stream)?;
        stream.encode(
            &buffers.kernels.extract,
            &bindings,
            &params,
            grid(output_len)?,
        )?;
        stream.submit()?;
        download_prefix(device, &buffers.output_buffer, output_len)
    }

    pub(crate) fn execute_fast_type2_3d(
        device: &WgpuDevice,
        shape: (usize, usize, usize),
        oversampled: (usize, usize, usize),
        configuration: KaiserBesselThree<'_>,
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let mut buffers = NufftGpuBuffers3D::new(device, shape, oversampled, positions.len())?;
        Self::execute_fast_type2_3d_with_buffers(
            device,
            &mut buffers,
            configuration,
            modes,
            positions,
        )
    }

    pub(crate) fn execute_fast_type2_3d_with_buffers(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers3D,
        configuration: KaiserBesselThree<'_>,
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_pod(&mut buffers.host_positions, positions);
        device.write_sub_buffer(&buffers.position_buffer, 0, &buffers.host_positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, configuration.deconvolution)?;
        device.write_sub_buffer(&buffers.coefficient_buffer, 0, modes)?;
        let params = FastNufftParams3D::for_grid(
            buffers.shape,
            buffers.oversampled,
            positions.len(),
            configuration,
        )?;
        let bindings = three_bindings(buffers, &buffers.coefficient_buffer);
        let grid_len = product(buffers.oversampled)?;
        let mut stream = device.stream()?;
        stream.encode(&buffers.kernels.load, &bindings, &params, grid(grid_len)?)?;
        buffers.encode_inverse(device, &mut stream)?;
        stream.encode(
            &buffers.kernels.interpolate,
            &bindings,
            &params,
            grid(positions.len())?,
        )?;
        stream.submit()?;
        download_prefix(device, &buffers.output_buffer, positions.len())
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn execute_fast_type2_1d_with_diagnostics(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers1D,
        configuration: KaiserBesselOne<'_>,
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<(Vec<Complex32>, NufftType2GridDiagnostics)> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_complex(&mut buffers.host_positions, positions);
        copy_real_as_complex(
            &mut buffers.host_deconvolution,
            configuration.deconvolution,
            buffers.m as f32,
        );
        device.write_sub_buffer(&buffers.position_buffer, 0, &buffers.host_positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, &buffers.host_deconvolution)?;
        device.write_sub_buffer(&buffers.coefficient_buffer, 0, coefficients)?;
        let params =
            FastNufftParams::for_grid(buffers.n, buffers.m, positions.len(), configuration)?;
        let bindings = one_bindings(buffers, &buffers.coefficient_buffer);
        let mut stream = device.stream()?;
        stream.encode(&buffers.kernels.load, &bindings, &params, grid(buffers.m)?)?;
        stream.submit()?;
        let after_load = snapshot_one(device, buffers)?;
        let mut stream = device.stream()?;
        buffers.encode_inverse(device, &mut stream)?;
        stream.submit()?;
        let after_ifft = snapshot_one(device, buffers)?;
        let mut stream = device.stream()?;
        stream.encode(
            &buffers.kernels.interpolate,
            &bindings,
            &params,
            grid(positions.len())?,
        )?;
        stream.submit()?;
        Ok((
            download_prefix(device, &buffers.output_buffer, positions.len())?,
            NufftType2GridDiagnostics {
                after_load,
                after_ifft,
            },
        ))
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn execute_fast_type2_3d_with_diagnostics(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers3D,
        configuration: KaiserBesselThree<'_>,
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<(Vec<Complex32>, NufftType2GridDiagnostics)> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_pod(&mut buffers.host_positions, positions);
        device.write_sub_buffer(&buffers.position_buffer, 0, &buffers.host_positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, configuration.deconvolution)?;
        device.write_sub_buffer(&buffers.coefficient_buffer, 0, modes)?;
        let params = FastNufftParams3D::for_grid(
            buffers.shape,
            buffers.oversampled,
            positions.len(),
            configuration,
        )?;
        let bindings = three_bindings(buffers, &buffers.coefficient_buffer);
        let grid_len = product(buffers.oversampled)?;
        let mut stream = device.stream()?;
        stream.encode(&buffers.kernels.load, &bindings, &params, grid(grid_len)?)?;
        stream.submit()?;
        let after_load = snapshot_three(device, buffers)?;
        let mut stream = device.stream()?;
        buffers.encode_inverse(device, &mut stream)?;
        stream.submit()?;
        let after_ifft = snapshot_three(device, buffers)?;
        let mut stream = device.stream()?;
        stream.encode(
            &buffers.kernels.interpolate,
            &bindings,
            &params,
            grid(positions.len())?,
        )?;
        stream.submit()?;
        Ok((
            download_prefix(device, &buffers.output_buffer, positions.len())?,
            NufftType2GridDiagnostics {
                after_load,
                after_ifft,
            },
        ))
    }
}
