//! Shared typed-buffer preparation and transfer operations for fast NUFFT dispatch.

use apollo_fft::GpuFft3d;
use eunomia::Complex32;
use hephaestus_core::{Binding, ComputeDevice, DeviceBuffer, DispatchGrid};
use hephaestus_wgpu::{WgpuBuffer, WgpuDevice};

use super::{
    buffers::{NufftGpuBuffers1D, NufftGpuBuffers3D},
    descriptors::{FastNufftParams, FastNufftParams3D, Position3Pod, WORKGROUP_SIZE},
};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};

#[cfg(any(test, feature = "diagnostics"))]
use super::buffers::NufftGridSnapshot;

/// One-dimensional Kaiser--Bessel parameters and deconvolution factors.
#[derive(Clone, Copy)]
pub(crate) struct KaiserBesselOne<'a> {
    pub(crate) kernel_width: usize,
    pub(crate) length: f32,
    pub(crate) beta: f32,
    pub(crate) i0_beta: f32,
    pub(crate) deconvolution: &'a [f32],
}

/// Three-dimensional Kaiser--Bessel parameters and deconvolution factors.
#[derive(Clone, Copy)]
pub(crate) struct KaiserBesselThree<'a> {
    pub(crate) kernel_width: usize,
    pub(crate) lengths: (f32, f32, f32),
    pub(crate) beta: f32,
    pub(crate) i0_beta: f32,
    pub(crate) deconvolution: &'a [f32],
}

impl FastNufftParams {
    pub(super) fn for_grid(
        n: usize,
        m: usize,
        sample_count: usize,
        configuration: KaiserBesselOne<'_>,
    ) -> NufftWgpuResult<Self> {
        Ok(Self {
            n: dimension(n, "mode count")?,
            m: dimension(m, "oversampled grid length")?,
            sample_count: dimension(sample_count, "sample count")?,
            kernel_width: dimension(configuration.kernel_width, "kernel width")?,
            length: configuration.length,
            beta: configuration.beta,
            i0_beta: configuration.i0_beta,
            padding: 0.0,
        })
    }
}

impl FastNufftParams3D {
    pub(super) fn for_grid(
        shape: (usize, usize, usize),
        oversampled: (usize, usize, usize),
        sample_count: usize,
        configuration: KaiserBesselThree<'_>,
    ) -> NufftWgpuResult<Self> {
        Ok(Self {
            nx: dimension(shape.0, "x mode count")?,
            ny: dimension(shape.1, "y mode count")?,
            nz: dimension(shape.2, "z mode count")?,
            mx: dimension(oversampled.0, "x oversampled length")?,
            my: dimension(oversampled.1, "y oversampled length")?,
            mz: dimension(oversampled.2, "z oversampled length")?,
            sample_count: dimension(sample_count, "sample count")?,
            kernel_width: dimension(configuration.kernel_width, "kernel width")?,
            lx: configuration.lengths.0,
            ly: configuration.lengths.1,
            lz: configuration.lengths.2,
            beta: configuration.beta,
            i0_beta: configuration.i0_beta,
            padding: [0.0; 3],
        })
    }
}

pub(super) fn one_bindings<'a>(
    buffers: &'a NufftGpuBuffers1D,
    coefficients: &'a WgpuBuffer<Complex32>,
) -> [Binding<'a, WgpuDevice>; 7] {
    [
        Binding::read(&buffers.position_buffer),
        Binding::read(&buffers.value_buffer),
        Binding::read_write(&buffers.real_grid),
        Binding::read_write(&buffers.imaginary_grid),
        Binding::read(&buffers.deconv_buffer),
        Binding::read_write(&buffers.output_buffer),
        Binding::read(coefficients),
    ]
}

pub(super) fn three_bindings<'a>(
    buffers: &'a NufftGpuBuffers3D,
    coefficients: &'a WgpuBuffer<Complex32>,
) -> [Binding<'a, WgpuDevice>; 7] {
    [
        Binding::read(&buffers.position_buffer),
        Binding::read(&buffers.value_buffer),
        Binding::read_write(&buffers.real_grid),
        Binding::read_write(&buffers.imaginary_grid),
        Binding::read(&buffers.deconv_buffer),
        Binding::read_write(&buffers.output_buffer),
        Binding::read(coefficients),
    ]
}

pub(super) fn write_one_type1_buffers(
    device: &WgpuDevice,
    buffers: &NufftGpuBuffers1D,
    positions: &[Complex32],
    values: &[Complex32],
    deconv: &[Complex32],
) -> NufftWgpuResult<()> {
    device.write_sub_buffer(&buffers.position_buffer, 0, positions)?;
    device.write_sub_buffer(&buffers.value_buffer, 0, values)?;
    device.write_sub_buffer(&buffers.deconv_buffer, 0, deconv)?;
    Ok(())
}

pub(super) fn write_three_type1_buffers(
    device: &WgpuDevice,
    buffers: &NufftGpuBuffers3D,
    positions: &[Position3Pod],
    values: &[Complex32],
    deconv: &[f32],
) -> NufftWgpuResult<()> {
    device.write_sub_buffer(&buffers.position_buffer, 0, positions)?;
    device.write_sub_buffer(&buffers.value_buffer, 0, values)?;
    device.write_sub_buffer(&buffers.deconv_buffer, 0, deconv)?;
    Ok(())
}

pub(super) fn positions_to_complex(positions: &[f32]) -> Vec<Complex32> {
    positions
        .iter()
        .copied()
        .map(|value| Complex32::new(value, 0.0))
        .collect()
}

pub(super) fn positions_to_pod(positions: &[(f32, f32, f32)]) -> Vec<Position3Pod> {
    positions
        .iter()
        .map(|&(x, y, z)| Position3Pod {
            x,
            y,
            z,
            padding: 0.0,
        })
        .collect()
}

pub(super) fn real_to_complex(values: &[f32], scale: f32) -> Vec<Complex32> {
    values
        .iter()
        .copied()
        .map(|value| Complex32::new(value * scale, 0.0))
        .collect()
}

pub(super) fn fft_one(device: &WgpuDevice, m: usize) -> NufftWgpuResult<GpuFft3d> {
    GpuFft3d::new(device.clone(), m, 1, 1).map_err(|_| NufftWgpuError::InvalidPlan {
        message: "oversampled FFT plan is invalid for provider execution",
    })
}

pub(super) fn fft_three(
    device: &WgpuDevice,
    oversampled: (usize, usize, usize),
) -> NufftWgpuResult<GpuFft3d> {
    GpuFft3d::new(device.clone(), oversampled.0, oversampled.1, oversampled.2).map_err(|_| {
        NufftWgpuError::InvalidPlan {
            message: "oversampled 3D FFT plan is invalid for provider execution",
        }
    })
}

pub(super) fn product(shape: (usize, usize, usize)) -> NufftWgpuResult<usize> {
    shape
        .0
        .checked_mul(shape.1)
        .and_then(|value| value.checked_mul(shape.2))
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "3D grid length overflows usize",
        })
}

pub(super) fn grid(elements: usize) -> NufftWgpuResult<DispatchGrid> {
    Ok(DispatchGrid::covering_domain(
        [elements, 1, 1],
        [WORKGROUP_SIZE as usize, 1, 1],
    )?)
}

pub(super) fn download_prefix(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<Complex32>,
    len: usize,
) -> NufftWgpuResult<Vec<Complex32>> {
    if len > buffer.len() {
        return Err(NufftWgpuError::InputLengthMismatch {
            expected: buffer.len(),
            actual: len,
        });
    }
    let mut values = vec![Complex32::new(0.0, 0.0); buffer.len()];
    device.download(buffer, &mut values)?;
    values.truncate(len);
    Ok(values)
}

#[cfg(any(test, feature = "diagnostics"))]
pub(super) fn snapshot_one(
    device: &WgpuDevice,
    buffers: &NufftGpuBuffers1D,
) -> NufftWgpuResult<NufftGridSnapshot> {
    snapshot(
        device,
        &buffers.real_grid,
        &buffers.imaginary_grid,
        buffers.m,
    )
}

#[cfg(any(test, feature = "diagnostics"))]
pub(super) fn snapshot_three(
    device: &WgpuDevice,
    buffers: &NufftGpuBuffers3D,
) -> NufftWgpuResult<NufftGridSnapshot> {
    snapshot(
        device,
        &buffers.real_grid,
        &buffers.imaginary_grid,
        product(buffers.oversampled)?,
    )
}

#[cfg(any(test, feature = "diagnostics"))]
fn snapshot(
    device: &WgpuDevice,
    real: &WgpuBuffer<f32>,
    imaginary: &WgpuBuffer<f32>,
    len: usize,
) -> NufftWgpuResult<NufftGridSnapshot> {
    let mut re = vec![0.0; real.len()];
    let mut im = vec![0.0; imaginary.len()];
    device.download(real, &mut re)?;
    device.download(imaginary, &mut im)?;
    re.truncate(len);
    im.truncate(len);
    Ok(NufftGridSnapshot { re, im })
}

fn dimension(value: usize, name: &'static str) -> NufftWgpuResult<u32> {
    u32::try_from(value).map_err(|_| NufftWgpuError::InvalidPlan { message: name })
}
