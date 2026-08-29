//! Shared typed-buffer preparation and transfer operations for fast NUFFT dispatch.

use eunomia::Complex32;
use hephaestus_core::{ComputeDevice, DeviceBuffer, DispatchGrid};
use hephaestus_wgpu::{WgpuBuffer, WgpuDevice};

use super::{
    buffers::{NufftGpuBuffers1D, NufftGpuBuffers3D},
    configuration::{FastConfiguration1D, FastConfiguration3D},
    descriptors::{FastNufftParams, FastNufftParams3D, Position3Pod, WORKGROUP_SIZE},
};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};

#[cfg(any(test, feature = "diagnostics"))]
use super::buffers::NufftGridSnapshot;

impl FastNufftParams {
    pub(super) fn for_grid(
        n: usize,
        m: usize,
        sample_count: usize,
        configuration: &FastConfiguration1D,
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
        configuration: &FastConfiguration3D,
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

pub(super) fn copy_positions_as_complex(output: &mut Vec<Complex32>, positions: &[f32]) {
    output.clear();
    output.extend(
        positions
            .iter()
            .copied()
            .map(|value| Complex32::new(value, 0.0)),
    );
}

pub(super) fn copy_positions_as_pod(output: &mut Vec<Position3Pod>, positions: &[(f32, f32, f32)]) {
    output.clear();
    output.extend(positions.iter().map(|&(x, y, z)| Position3Pod {
        x,
        y,
        z,
        padding: 0.0,
    }));
}

pub(super) fn copy_real_as_complex(output: &mut Vec<Complex32>, values: &[f32], scale: f32) {
    output.clear();
    output.extend(
        values
            .iter()
            .copied()
            .map(|value| Complex32::new(value * scale, 0.0)),
    );
}

#[cfg(any(test, feature = "diagnostics"))]
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

pub(super) fn download(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<Complex32>,
    output: &mut [Complex32],
) -> NufftWgpuResult<()> {
    if output.len() != buffer.len() {
        return Err(NufftWgpuError::InputLengthMismatch {
            expected: buffer.len(),
            actual: output.len(),
        });
    }
    device.download(buffer, output)?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{copy_positions_as_complex, copy_positions_as_pod, copy_real_as_complex};
    use crate::infrastructure::transport::gpu::verification::count_allocations;
    use eunomia::Complex32;

    #[test]
    fn retained_host_conversions_allocate_nothing() {
        let mut one_positions = Vec::with_capacity(4);
        copy_positions_as_complex(&mut one_positions, &[0.0, 0.25, 0.5, 0.75]);
        let one_pointer = one_positions.as_ptr();
        let one_capacity = one_positions.capacity();

        let mut three_positions = Vec::with_capacity(3);
        copy_positions_as_pod(
            &mut three_positions,
            &[(0.0, 0.25, 0.5), (0.75, 1.0, 1.25), (1.5, 1.75, 2.0)],
        );
        let three_pointer = three_positions.as_ptr();
        let three_capacity = three_positions.capacity();

        let mut deconvolution = Vec::with_capacity(4);
        copy_real_as_complex(&mut deconvolution, &[1.0, 2.0, 3.0, 4.0], 1.0);
        let deconvolution_pointer = deconvolution.as_ptr();
        let deconvolution_capacity = deconvolution.capacity();

        let ((), allocations) = count_allocations(|| {
            copy_positions_as_complex(&mut one_positions, &[0.125, 0.625]);
            copy_positions_as_pod(
                &mut three_positions,
                &[(0.125, 0.375, 0.625), (0.875, 1.125, 1.375)],
            );
            copy_real_as_complex(&mut deconvolution, &[1.5, 2.5, 3.5], 2.0);
        });

        assert_eq!(allocations, 0, "retained host conversion allocated");
        assert_eq!(
            one_positions,
            [Complex32::new(0.125, 0.0), Complex32::new(0.625, 0.0)]
        );
        assert_eq!(three_positions.len(), 2);
        assert_eq!(three_positions[0].x, 0.125);
        assert_eq!(three_positions[0].y, 0.375);
        assert_eq!(three_positions[0].z, 0.625);
        assert_eq!(three_positions[1].x, 0.875);
        assert_eq!(three_positions[1].y, 1.125);
        assert_eq!(three_positions[1].z, 1.375);
        assert_eq!(
            deconvolution,
            [
                Complex32::new(3.0, 0.0),
                Complex32::new(5.0, 0.0),
                Complex32::new(7.0, 0.0)
            ]
        );
        assert_eq!(one_positions.as_ptr(), one_pointer);
        assert_eq!(one_positions.capacity(), one_capacity);
        assert_eq!(three_positions.as_ptr(), three_pointer);
        assert_eq!(three_positions.capacity(), three_capacity);
        assert_eq!(deconvolution.as_ptr(), deconvolution_pointer);
        assert_eq!(deconvolution.capacity(), deconvolution_capacity);
    }
}
