//! Kaiser--Bessel spread/FFT/extract and load/IFFT/interpolate dispatch.

use eunomia::Complex32;
use hephaestus_core::{CommandStream, ComputeDevice, KernelDevice};
use hephaestus_wgpu::WgpuDevice;

use super::{
    buffers::{ensure_sample_capacity, NufftGpuBuffers1D, NufftGpuBuffers3D},
    descriptors::{
        ExtractOne, ExtractThree, FastNufftParams, FastNufftParams3D, FastOneKernel,
        FastThreeKernel, InterpolateOne, InterpolateThree, LoadOne, LoadThree, SpreadOne,
        SpreadThree,
    },
    NufftGpuKernel,
};
use crate::infrastructure::transport::gpu::domain::error::NufftWgpuResult;

use super::fast_support::{
    download_prefix, fft_one, fft_three, grid, one_bindings, positions_to_complex,
    positions_to_pod, product, real_to_complex, three_bindings, write_one_type1_buffers,
    write_three_type1_buffers, KaiserBesselOne, KaiserBesselThree,
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
        let buffers = NufftGpuBuffers1D::new(device, n, m, positions.len())?;
        Self::execute_fast_type1_1d_with_buffers(device, &buffers, configuration, positions, values)
    }

    pub(crate) fn execute_fast_type1_1d_with_buffers(
        device: &WgpuDevice,
        buffers: &NufftGpuBuffers1D,
        configuration: KaiserBesselOne<'_>,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let positions = positions_to_complex(positions);
        let deconv = real_to_complex(configuration.deconvolution, 1.0);
        write_one_type1_buffers(device, buffers, &positions, values, &deconv)?;
        let params =
            FastNufftParams::for_grid(buffers.n, buffers.m, positions.len(), configuration)?;
        let spread = device.prepare(&FastOneKernel::<SpreadOne>::new())?;
        let extract = device.prepare(&FastOneKernel::<ExtractOne>::new())?;
        let bindings = one_bindings(buffers, &buffers.padding_buffer);
        let mut stream = device.stream()?;
        stream.encode(&spread, &bindings, &params, grid(buffers.m)?)?;
        fft_one(device, buffers.m)?.encode_forward_split(
            &mut stream,
            &buffers.real_grid,
            &buffers.imaginary_grid,
        )?;
        stream.encode(&extract, &bindings, &params, grid(buffers.n)?)?;
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
        let buffers = NufftGpuBuffers1D::new(device, n, m, positions.len())?;
        Self::execute_fast_type2_1d_with_buffers(
            device,
            &buffers,
            configuration,
            coefficients,
            positions,
        )
    }

    pub(crate) fn execute_fast_type2_1d_with_buffers(
        device: &WgpuDevice,
        buffers: &NufftGpuBuffers1D,
        configuration: KaiserBesselOne<'_>,
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let positions = positions_to_complex(positions);
        let deconv = real_to_complex(configuration.deconvolution, buffers.m as f32);
        device.write_sub_buffer(&buffers.position_buffer, 0, &positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, &deconv)?;
        let coefficients = device.upload(coefficients)?;
        let params =
            FastNufftParams::for_grid(buffers.n, buffers.m, positions.len(), configuration)?;
        let load = device.prepare(&FastOneKernel::<LoadOne>::new())?;
        let interpolate = device.prepare(&FastOneKernel::<InterpolateOne>::new())?;
        let load_bindings = one_bindings(buffers, &coefficients);
        let interpolate_bindings = one_bindings(buffers, &coefficients);
        let mut stream = device.stream()?;
        stream.encode(&load, &load_bindings, &params, grid(buffers.m)?)?;
        fft_one(device, buffers.m)?.encode_inverse_split(
            &mut stream,
            &buffers.real_grid,
            &buffers.imaginary_grid,
        )?;
        stream.encode(
            &interpolate,
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
        let buffers = NufftGpuBuffers3D::new(device, shape, oversampled, positions.len())?;
        Self::execute_fast_type1_3d_with_buffers(device, &buffers, configuration, positions, values)
    }

    pub(crate) fn execute_fast_type1_3d_with_buffers(
        device: &WgpuDevice,
        buffers: &NufftGpuBuffers3D,
        configuration: KaiserBesselThree<'_>,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let positions = positions_to_pod(positions);
        write_three_type1_buffers(
            device,
            buffers,
            &positions,
            values,
            configuration.deconvolution,
        )?;
        let params = FastNufftParams3D::for_grid(
            buffers.shape,
            buffers.oversampled,
            positions.len(),
            configuration,
        )?;
        let spread = device.prepare(&FastThreeKernel::<SpreadThree>::new())?;
        let extract = device.prepare(&FastThreeKernel::<ExtractThree>::new())?;
        let bindings = three_bindings(buffers, &buffers.padding_buffer);
        let grid_len = product(buffers.oversampled)?;
        let output_len = product(buffers.shape)?;
        let mut stream = device.stream()?;
        stream.encode(&spread, &bindings, &params, grid(grid_len)?)?;
        fft_three(device, buffers.oversampled)?.encode_forward_split(
            &mut stream,
            &buffers.real_grid,
            &buffers.imaginary_grid,
        )?;
        stream.encode(&extract, &bindings, &params, grid(output_len)?)?;
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
        let buffers = NufftGpuBuffers3D::new(device, shape, oversampled, positions.len())?;
        Self::execute_fast_type2_3d_with_buffers(device, &buffers, configuration, modes, positions)
    }

    pub(crate) fn execute_fast_type2_3d_with_buffers(
        device: &WgpuDevice,
        buffers: &NufftGpuBuffers3D,
        configuration: KaiserBesselThree<'_>,
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let positions = positions_to_pod(positions);
        device.write_sub_buffer(&buffers.position_buffer, 0, &positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, configuration.deconvolution)?;
        let coefficients = device.upload(modes)?;
        let params = FastNufftParams3D::for_grid(
            buffers.shape,
            buffers.oversampled,
            positions.len(),
            configuration,
        )?;
        let load = device.prepare(&FastThreeKernel::<LoadThree>::new())?;
        let interpolate = device.prepare(&FastThreeKernel::<InterpolateThree>::new())?;
        let bindings = three_bindings(buffers, &coefficients);
        let grid_len = product(buffers.oversampled)?;
        let mut stream = device.stream()?;
        stream.encode(&load, &bindings, &params, grid(grid_len)?)?;
        fft_three(device, buffers.oversampled)?.encode_inverse_split(
            &mut stream,
            &buffers.real_grid,
            &buffers.imaginary_grid,
        )?;
        stream.encode(&interpolate, &bindings, &params, grid(positions.len())?)?;
        stream.submit()?;
        download_prefix(device, &buffers.output_buffer, positions.len())
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn execute_fast_type2_1d_with_diagnostics(
        device: &WgpuDevice,
        buffers: &NufftGpuBuffers1D,
        configuration: KaiserBesselOne<'_>,
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<(Vec<Complex32>, NufftType2GridDiagnostics)> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let positions = positions_to_complex(positions);
        let deconv = real_to_complex(configuration.deconvolution, buffers.m as f32);
        device.write_sub_buffer(&buffers.position_buffer, 0, &positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, &deconv)?;
        let coefficients = device.upload(coefficients)?;
        let params =
            FastNufftParams::for_grid(buffers.n, buffers.m, positions.len(), configuration)?;
        let load = device.prepare(&FastOneKernel::<LoadOne>::new())?;
        let interpolate = device.prepare(&FastOneKernel::<InterpolateOne>::new())?;
        let bindings = one_bindings(buffers, &coefficients);
        let mut stream = device.stream()?;
        stream.encode(&load, &bindings, &params, grid(buffers.m)?)?;
        stream.submit()?;
        let after_load = snapshot_one(device, buffers)?;
        let mut stream = device.stream()?;
        fft_one(device, buffers.m)?.encode_inverse_split(
            &mut stream,
            &buffers.real_grid,
            &buffers.imaginary_grid,
        )?;
        stream.submit()?;
        let after_ifft = snapshot_one(device, buffers)?;
        let mut stream = device.stream()?;
        stream.encode(&interpolate, &bindings, &params, grid(positions.len())?)?;
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
        buffers: &NufftGpuBuffers3D,
        configuration: KaiserBesselThree<'_>,
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<(Vec<Complex32>, NufftType2GridDiagnostics)> {
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let positions = positions_to_pod(positions);
        device.write_sub_buffer(&buffers.position_buffer, 0, &positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, configuration.deconvolution)?;
        let coefficients = device.upload(modes)?;
        let params = FastNufftParams3D::for_grid(
            buffers.shape,
            buffers.oversampled,
            positions.len(),
            configuration,
        )?;
        let load = device.prepare(&FastThreeKernel::<LoadThree>::new())?;
        let interpolate = device.prepare(&FastThreeKernel::<InterpolateThree>::new())?;
        let bindings = three_bindings(buffers, &coefficients);
        let grid_len = product(buffers.oversampled)?;
        let mut stream = device.stream()?;
        stream.encode(&load, &bindings, &params, grid(grid_len)?)?;
        stream.submit()?;
        let after_load = snapshot_three(device, buffers)?;
        let mut stream = device.stream()?;
        fft_three(device, buffers.oversampled)?.encode_inverse_split(
            &mut stream,
            &buffers.real_grid,
            &buffers.imaginary_grid,
        )?;
        stream.submit()?;
        let after_ifft = snapshot_three(device, buffers)?;
        let mut stream = device.stream()?;
        stream.encode(&interpolate, &bindings, &params, grid(positions.len())?)?;
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
