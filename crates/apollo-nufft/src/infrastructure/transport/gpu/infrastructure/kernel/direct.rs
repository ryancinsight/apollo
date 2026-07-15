//! Exact direct-sum NUFFT dispatch through typed Hephaestus buffers.

use eunomia::Complex32;
use hephaestus_core::{Binding, CommandStream, ComputeDevice, DispatchGrid, KernelDevice};
use hephaestus_wgpu::WgpuDevice;

use super::{
    descriptors::{
        DirectKernel, NufftParams, Position3Pod, Type1One, Type1Three, Type2One, Type2Three,
        WORKGROUP_SIZE,
    },
    NufftGpuKernel,
};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};

impl NufftGpuKernel {
    pub(crate) fn execute_type1_1d(
        device: &WgpuDevice,
        n: usize,
        length: f32,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let positions = positions
            .iter()
            .copied()
            .map(|x| Position3Pod {
                x,
                y: 0.0,
                z: 0.0,
                padding: 0.0,
            })
            .collect::<Vec<_>>();
        Self::execute_direct(
            device,
            &DirectKernel::<Type1One>::new(),
            &positions,
            values,
            n,
            NufftParams::one(n, positions.len(), length)?,
        )
    }

    pub(crate) fn execute_type2_1d(
        device: &WgpuDevice,
        n: usize,
        length: f32,
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let positions = positions
            .iter()
            .copied()
            .map(|x| Position3Pod {
                x,
                y: 0.0,
                z: 0.0,
                padding: 0.0,
            })
            .collect::<Vec<_>>();
        Self::execute_direct(
            device,
            &DirectKernel::<Type2One>::new(),
            &positions,
            coefficients,
            positions.len(),
            NufftParams::one(n, positions.len(), length)?,
        )
    }

    pub(crate) fn execute_type1_3d(
        device: &WgpuDevice,
        shape: (usize, usize, usize),
        lengths: (f32, f32, f32),
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let positions = positions
            .iter()
            .map(|&(x, y, z)| Position3Pod {
                x,
                y,
                z,
                padding: 0.0,
            })
            .collect::<Vec<_>>();
        let output_len = shape
            .0
            .checked_mul(shape.1)
            .and_then(|value| value.checked_mul(shape.2))
            .ok_or(NufftWgpuError::InvalidPlan {
                message: "3D direct Type-1 output length overflows usize",
            })?;
        Self::execute_direct(
            device,
            &DirectKernel::<Type1Three>::new(),
            &positions,
            values,
            output_len,
            NufftParams::three(shape, positions.len(), lengths)?,
        )
    }

    pub(crate) fn execute_type2_3d(
        device: &WgpuDevice,
        shape: (usize, usize, usize),
        lengths: (f32, f32, f32),
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let positions = positions
            .iter()
            .map(|&(x, y, z)| Position3Pod {
                x,
                y,
                z,
                padding: 0.0,
            })
            .collect::<Vec<_>>();
        Self::execute_direct(
            device,
            &DirectKernel::<Type2Three>::new(),
            &positions,
            modes,
            positions.len(),
            NufftParams::three(shape, positions.len(), lengths)?,
        )
    }

    fn execute_direct<K>(
        device: &WgpuDevice,
        kernel: &K,
        positions: &[Position3Pod],
        values: &[Complex32],
        output_len: usize,
        params: NufftParams,
    ) -> NufftWgpuResult<Vec<Complex32>>
    where
        K: hephaestus_core::KernelInterface<Params = NufftParams>
            + hephaestus_core::KernelSource<hephaestus_core::Wgsl>,
    {
        let positions = device.upload(positions)?;
        let values = device.upload(values)?;
        let output = device.alloc_zeroed(output_len)?;
        let prepared = device.prepare(kernel)?;
        let bindings = [
            Binding::read(&positions),
            Binding::read(&values),
            Binding::read_write(&output),
        ];
        let mut stream = device.stream()?;
        stream.encode(&prepared, &bindings, &params, grid(output_len)?)?;
        stream.submit()?;
        let mut result = vec![Complex32::new(0.0, 0.0); output_len];
        device.download(&output, &mut result)?;
        Ok(result)
    }
}

impl NufftParams {
    fn one(n: usize, sample_count: usize, length: f32) -> NufftWgpuResult<Self> {
        Ok(Self {
            n0: dimension(n, "mode count")?,
            n1: 1,
            n2: 1,
            sample_count: dimension(sample_count, "sample count")?,
            l0: length,
            l1: 1.0,
            l2: 1.0,
            padding: 0.0,
        })
    }

    fn three(
        shape: (usize, usize, usize),
        sample_count: usize,
        lengths: (f32, f32, f32),
    ) -> NufftWgpuResult<Self> {
        Ok(Self {
            n0: dimension(shape.0, "x mode count")?,
            n1: dimension(shape.1, "y mode count")?,
            n2: dimension(shape.2, "z mode count")?,
            sample_count: dimension(sample_count, "sample count")?,
            l0: lengths.0,
            l1: lengths.1,
            l2: lengths.2,
            padding: 0.0,
        })
    }
}

fn dimension(value: usize, name: &'static str) -> NufftWgpuResult<u32> {
    u32::try_from(value).map_err(|_| NufftWgpuError::InvalidPlan { message: name })
}

fn grid(elements: usize) -> NufftWgpuResult<DispatchGrid> {
    Ok(DispatchGrid::covering_domain(
        [elements, 1, 1],
        [WORKGROUP_SIZE as usize, 1, 1],
    )?)
}
