//! Kaiser--Bessel spread/FFT/extract and load/IFFT/interpolate dispatch.

use eunomia::Complex32;
use hephaestus_core::{GroupedCommandStream, GroupedKernelDevice};
use hephaestus_wgpu::WgpuDevice;

use super::{
    buffers::{ensure_sample_capacity, NufftGpuBuffers1D, NufftGpuBuffers3D},
    descriptors::{FastNufftParams, FastNufftParams3D},
    fast_support::{
        copy_positions_as_complex, copy_positions_as_pod, copy_real_as_complex, download,
        write_one_type1_buffers, write_three_type1_buffers,
    },
    NufftGpuKernel,
};
use crate::infrastructure::transport::gpu::domain::error::NufftWgpuResult;

#[cfg(any(test, feature = "diagnostics"))]
use super::{
    buffers::NufftType2GridDiagnostics,
    fast_support::{snapshot_one, snapshot_three},
};

impl NufftGpuKernel {
    pub(crate) fn execute_fast_type1_1d_with_buffers(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers1D,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<()> {
        buffers.validate_device(device)?;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_complex(&mut buffers.host_positions, positions);
        copy_real_as_complex(
            &mut buffers.host_deconvolution,
            &buffers.configuration.deconvolution,
            1.0,
        );
        write_one_type1_buffers(
            device,
            buffers,
            &buffers.host_positions,
            values,
            &buffers.host_deconvolution,
        )?;
        let params = FastNufftParams::for_grid(
            buffers.n,
            buffers.m,
            positions.len(),
            &buffers.configuration,
        )?;
        buffers.update_type_one(device, &params)?;
        let mut stream = device.grouped_stream()?;
        stream.encode_grouped_sequence("apollo-nufft-fast-type1-1d", |sequence| {
            buffers.encode_spread(sequence)?;
            buffers.encode_forward(sequence)?;
            buffers.encode_extract(sequence)
        })?;
        stream.submit_grouped()?;
        download(device, &buffers.output_buffer, &mut buffers.host_readback)
    }

    pub(crate) fn execute_fast_type2_1d_with_buffers(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers1D,
        positions: &[f32],
    ) -> NufftWgpuResult<()> {
        buffers.validate_device(device)?;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_complex(&mut buffers.host_positions, positions);
        copy_real_as_complex(
            &mut buffers.host_deconvolution,
            &buffers.configuration.deconvolution,
            buffers.m as f32,
        );
        device.write_sub_buffer(&buffers.position_buffer, 0, &buffers.host_positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, &buffers.host_deconvolution)?;
        let params = FastNufftParams::for_grid(
            buffers.n,
            buffers.m,
            positions.len(),
            &buffers.configuration,
        )?;
        buffers.update_type_two(device, &params)?;
        let mut stream = device.grouped_stream()?;
        stream.encode_grouped_sequence("apollo-nufft-fast-type2-1d", |sequence| {
            buffers.encode_load(sequence)?;
            buffers.encode_inverse(sequence)?;
            buffers.encode_interpolate(sequence)
        })?;
        stream.submit_grouped()?;
        download(device, &buffers.output_buffer, &mut buffers.host_readback)
    }

    pub(crate) fn execute_fast_type1_3d_with_buffers(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers3D,
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<()> {
        buffers.validate_device(device)?;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_pod(&mut buffers.host_positions, positions);
        write_three_type1_buffers(
            device,
            buffers,
            &buffers.host_positions,
            values,
            &buffers.configuration.deconvolution,
        )?;
        let params = FastNufftParams3D::for_grid(
            buffers.shape,
            buffers.oversampled,
            positions.len(),
            &buffers.configuration,
        )?;
        buffers.update_type_one(device, &params)?;
        let mut stream = device.grouped_stream()?;
        stream.encode_grouped_sequence("apollo-nufft-fast-type1-3d", |sequence| {
            buffers.encode_spread(sequence)?;
            buffers.encode_forward(sequence)?;
            buffers.encode_extract(sequence)
        })?;
        stream.submit_grouped()?;
        download(device, &buffers.output_buffer, &mut buffers.host_readback)
    }

    pub(crate) fn execute_fast_type2_3d_with_buffers(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers3D,
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<()> {
        buffers.validate_device(device)?;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_pod(&mut buffers.host_positions, positions);
        device.write_sub_buffer(&buffers.position_buffer, 0, &buffers.host_positions)?;
        device.write_sub_buffer(
            &buffers.deconv_buffer,
            0,
            &buffers.configuration.deconvolution,
        )?;
        let params = FastNufftParams3D::for_grid(
            buffers.shape,
            buffers.oversampled,
            positions.len(),
            &buffers.configuration,
        )?;
        buffers.update_type_two(device, &params)?;
        let mut stream = device.grouped_stream()?;
        stream.encode_grouped_sequence("apollo-nufft-fast-type2-3d", |sequence| {
            buffers.encode_load(sequence)?;
            buffers.encode_inverse(sequence)?;
            buffers.encode_interpolate(sequence)
        })?;
        stream.submit_grouped()?;
        download(device, &buffers.output_buffer, &mut buffers.host_readback)
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn execute_fast_type2_1d_with_diagnostics(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers1D,
        positions: &[f32],
    ) -> NufftWgpuResult<NufftType2GridDiagnostics> {
        buffers.validate_device(device)?;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_complex(&mut buffers.host_positions, positions);
        copy_real_as_complex(
            &mut buffers.host_deconvolution,
            &buffers.configuration.deconvolution,
            buffers.m as f32,
        );
        device.write_sub_buffer(&buffers.position_buffer, 0, &buffers.host_positions)?;
        device.write_sub_buffer(&buffers.deconv_buffer, 0, &buffers.host_deconvolution)?;
        let params = FastNufftParams::for_grid(
            buffers.n,
            buffers.m,
            positions.len(),
            &buffers.configuration,
        )?;
        buffers.update_type_two(device, &params)?;
        let (after_load, after_ifft) = {
            let mut stream = device.grouped_stream()?;
            stream.encode_grouped_sequence("apollo-nufft-diagnostics-load-1d", |sequence| {
                buffers.encode_load(sequence)
            })?;
            stream.submit_grouped()?;
            let after_load = snapshot_one(device, buffers)?;
            let mut stream = device.grouped_stream()?;
            stream.encode_grouped_sequence("apollo-nufft-diagnostics-ifft-1d", |sequence| {
                buffers.encode_inverse(sequence)
            })?;
            stream.submit_grouped()?;
            let after_ifft = snapshot_one(device, buffers)?;
            let mut stream = device.grouped_stream()?;
            stream
                .encode_grouped_sequence("apollo-nufft-diagnostics-interpolate-1d", |sequence| {
                    buffers.encode_interpolate(sequence)
                })?;
            stream.submit_grouped()?;
            (after_load, after_ifft)
        };
        download(device, &buffers.output_buffer, &mut buffers.host_readback)?;
        Ok(NufftType2GridDiagnostics {
            after_load,
            after_ifft,
        })
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn execute_fast_type2_3d_with_diagnostics(
        device: &WgpuDevice,
        buffers: &mut NufftGpuBuffers3D,
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<NufftType2GridDiagnostics> {
        buffers.validate_device(device)?;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        copy_positions_as_pod(&mut buffers.host_positions, positions);
        device.write_sub_buffer(&buffers.position_buffer, 0, &buffers.host_positions)?;
        device.write_sub_buffer(
            &buffers.deconv_buffer,
            0,
            &buffers.configuration.deconvolution,
        )?;
        let params = FastNufftParams3D::for_grid(
            buffers.shape,
            buffers.oversampled,
            positions.len(),
            &buffers.configuration,
        )?;
        buffers.update_type_two(device, &params)?;
        let (after_load, after_ifft) = {
            let mut stream = device.grouped_stream()?;
            stream.encode_grouped_sequence("apollo-nufft-diagnostics-load-3d", |sequence| {
                buffers.encode_load(sequence)
            })?;
            stream.submit_grouped()?;
            let after_load = snapshot_three(device, buffers)?;
            let mut stream = device.grouped_stream()?;
            stream.encode_grouped_sequence("apollo-nufft-diagnostics-ifft-3d", |sequence| {
                buffers.encode_inverse(sequence)
            })?;
            stream.submit_grouped()?;
            let after_ifft = snapshot_three(device, buffers)?;
            let mut stream = device.grouped_stream()?;
            stream
                .encode_grouped_sequence("apollo-nufft-diagnostics-interpolate-3d", |sequence| {
                    buffers.encode_interpolate(sequence)
                })?;
            stream.submit_grouped()?;
            (after_load, after_ifft)
        };
        download(device, &buffers.output_buffer, &mut buffers.host_readback)?;
        Ok(NufftType2GridDiagnostics {
            after_load,
            after_ifft,
        })
    }
}
