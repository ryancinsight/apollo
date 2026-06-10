use super::{NufftGpuKernel, NufftParams, Position3Pod};
use crate::domain::error::NufftWgpuResult;
use num_complex::Complex32;

impl NufftGpuKernel {
    /// Execute exact direct Type-1 1D NUFFT.
    pub fn execute_type1_1d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
        length: f32,
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|x| Position3Pod {
                x: *x,
                y: 0.0,
                z: 0.0,
                _pad: 0.0,
            })
            .collect();
        let params = NufftParams {
            n0: n as u32,
            n1: 1,
            n2: 1,
            sample_count: positions.len() as u32,
            l0: length,
            l1: 1.0,
            l2: 1.0,
            _pad: 0.0,
        };
        self.execute(
            device,
            queue,
            &position_data,
            values,
            n,
            params,
            &self.type1_1d_pipeline,
        )
    }

    /// Execute exact direct Type-1 3D NUFFT.
    pub fn execute_type1_3d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: (usize, usize, usize),
        lengths: (f32, f32, f32),
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();
        let output_len = shape.0 * shape.1 * shape.2;
        let params = NufftParams {
            n0: shape.0 as u32,
            n1: shape.1 as u32,
            n2: shape.2 as u32,
            sample_count: positions.len() as u32,
            l0: lengths.0,
            l1: lengths.1,
            l2: lengths.2,
            _pad: 0.0,
        };
        self.execute(
            device,
            queue,
            &position_data,
            values,
            output_len,
            params,
            &self.type1_3d_pipeline,
        )
    }
}
