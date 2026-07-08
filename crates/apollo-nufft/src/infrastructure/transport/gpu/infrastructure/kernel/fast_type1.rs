use super::{
    binding, dispatch_count, positions_to_complex_1d, read_complex_buffer,
    read_complex_buffer_with_staging, real_to_complex, split_grid_buffers, storage_buffer,
    FastNufftParams, FastNufftParams3D, NufftGpuKernel, Position3Pod,
};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::buffers::ensure_sample_capacity;
use crate::infrastructure::transport::gpu::infrastructure::kernel::{
    NufftGpuBuffers1D, NufftGpuBuffers3D,
};
use apollo_fft::GpuFft3d;
use eunomia::Complex32;
use std::sync::Arc;
use wgpu::util::DeviceExt;

impl NufftGpuKernel {
    /// Execute fast gridded Type-1 1D NUFFT with GPU spreading, FFT, and deconvolution.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type1_1d(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        n: usize,
        oversampled_len: usize,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let position_data = positions_to_complex_1d(positions);
        let deconv_data = real_to_complex(deconv);
        let position_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast positions", &position_data);
        let value_buffer = storage_buffer(device, "apollo-nufft-wgpu fast values", values);
        let deconv_buffer = storage_buffer(device, "apollo-nufft-wgpu fast deconv", &deconv_data);
        let (re_buffer, im_buffer) = split_grid_buffers(device, oversampled_len);
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 output"),
            size: (n * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = FastNufftParams {
            n: n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        let fast_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu fast params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let spread_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 spread bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &value_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &self.layout_padding_buffer),
                binding(7, &fast_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 spread encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 spread pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_spread_1d_pipeline);
            pass.set_bind_group(0, &spread_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }

        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_forward_split(&mut encoder, &re_buffer, &im_buffer);

        let extract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 extract bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &value_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &self.layout_padding_buffer),
                binding(7, &fast_params_buffer),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 extract pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_extract_1d_pipeline);
            pass.set_bind_group(0, &extract_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, n)
    }

    /// Execute fast gridded Type-1 3D NUFFT with GPU separable spreading, 3D FFT, and deconvolution.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type1_3d(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        shape: (usize, usize, usize),
        oversampled: (usize, usize, usize),
        kernel_width: usize,
        lengths: (f32, f32, f32),
        beta: f32,
        i0_beta: f32,
        deconv_xyz: &[f32],
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let (nx, ny, nz) = shape;
        let (mx, my, mz) = oversampled;
        let (lx, ly, lz) = lengths;
        let grid_len = mx * my * mz;
        let output_len = nx * ny * nz;

        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();

        let position_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast3d positions", &position_data);
        let value_buffer = storage_buffer(device, "apollo-nufft-wgpu fast3d values", values);
        let deconv_buffer = storage_buffer(device, "apollo-nufft-wgpu fast3d deconv", deconv_xyz);
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 output"),
            size: (output_len * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (re_buffer, im_buffer) = split_grid_buffers(device, grid_len);

        let params = FastNufftParams3D {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            mx: mx as u32,
            my: my as u32,
            mz: mz as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            lx,
            ly,
            lz,
            beta,
            i0_beta,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let fast_3d_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu fast 3d params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let spread_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 spread bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &value_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &self.layout_padding_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type1 spread pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_spread_3d_pipeline);
            pass.set_bind_group(0, &spread_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }

        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D FFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_forward_split(&mut encoder, &re_buffer, &im_buffer);

        let extract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 extract bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &value_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &self.layout_padding_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type1 extract pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_extract_3d_pipeline);
            pass.set_bind_group(0, &extract_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(output_len as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, output_len)
    }

    /// Execute fast gridded Type-1 1D NUFFT with pre-allocated buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type1_1d_with_buffers(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers1D,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        positions: &[f32],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let n = buffers.n;
        let oversampled_len = buffers.m;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;

        let position_data = positions_to_complex_1d(positions);
        let deconv_data = real_to_complex(deconv);

        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(&buffers.value_buffer, 0, bytemuck::cast_slice(values));
        queue.write_buffer(
            &buffers.deconv_buffer,
            0,
            bytemuck::cast_slice(&deconv_data),
        );

        let params = FastNufftParams {
            n: n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        let fast_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu fast params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let spread_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 spread bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &buffers.value_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &buffers.output_buffer),
                binding(6, &self.layout_padding_buffer),
                binding(7, &fast_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 spread encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 spread pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_spread_1d_pipeline);
            pass.set_bind_group(0, &spread_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }

        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_forward_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);

        let extract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type1 extract bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &buffers.value_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &buffers.output_buffer),
                binding(6, &self.layout_padding_buffer),
                binding(7, &fast_params_buffer),
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 extract pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_extract_1d_pipeline);
            pass.set_bind_group(0, &extract_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer_with_staging(
            device,
            queue,
            &buffers.output_buffer,
            &buffers.staging_buffer,
            n,
        )
    }

    /// Execute fast gridded Type-1 3D NUFFT with pre-allocated buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type1_3d_with_buffers(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers3D,
        kernel_width: usize,
        lengths: (f32, f32, f32),
        beta: f32,
        i0_beta: f32,
        deconv_xyz: &[f32],
        positions: &[(f32, f32, f32)],
        values: &[Complex32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let (nx, ny, nz) = buffers.shape;
        let (mx, my, mz) = buffers.oversampled;
        let (lx, ly, lz) = lengths;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let grid_len = mx * my * mz;
        let output_len = nx * ny * nz;

        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();

        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(&buffers.value_buffer, 0, bytemuck::cast_slice(values));
        queue.write_buffer(&buffers.deconv_buffer, 0, bytemuck::cast_slice(deconv_xyz));

        let params = FastNufftParams3D {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            mx: mx as u32,
            my: my as u32,
            mz: mz as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            lx,
            ly,
            lz,
            beta,
            i0_beta,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let fast_3d_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu fast 3d params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let spread_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 spread bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &buffers.value_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &buffers.output_buffer),
                binding(6, &self.layout_padding_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type1 spread pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_spread_3d_pipeline);
            pass.set_bind_group(0, &spread_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }

        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D FFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_forward_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);

        let extract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type1 extract bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &buffers.value_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &buffers.output_buffer),
                binding(6, &self.layout_padding_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type1 extract pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type1_extract_3d_pipeline);
            pass.set_bind_group(0, &extract_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(output_len as u32), 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer_with_staging(
            device,
            queue,
            &buffers.output_buffer,
            &buffers.staging_buffer,
            output_len,
        )
    }
}
