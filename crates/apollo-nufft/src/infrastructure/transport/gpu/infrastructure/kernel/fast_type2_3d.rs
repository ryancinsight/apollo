use super::{
    binding, dispatch_count, read_complex_buffer, split_grid_buffers, storage_buffer,
    FastNufftParams3D, NufftGpuKernel, Position3Pod,
};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};
use wgpu::util::DeviceExt;
use crate::infrastructure::transport::gpu::infrastructure::kernel::buffers::ensure_sample_capacity;
use crate::infrastructure::transport::gpu::infrastructure::kernel::NufftGpuBuffers3D;
use apollo_fft::GpuFft3d;
use num_complex::Complex32;
use std::sync::Arc;

#[cfg(any(test, feature = "diagnostics"))]
use super::{read_split_grid_snapshot, NufftType2GridDiagnostics};

impl NufftGpuKernel {
    /// Execute fast gridded Type-2 3D NUFFT with GPU separable IFFT, deconvolution, and interpolation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_3d(
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
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let (nx, ny, nz) = shape;
        let (mx, my, mz) = oversampled;
        let (lx, ly, lz) = lengths;
        let grid_len = mx * my * mz;
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();

        let coeff_buffer = storage_buffer(device, "apollo-nufft-wgpu fast3d type2 coeffs", modes);
        let position_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu fast3d type2 positions",
            &position_data,
        );
        let deconv_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast3d type2 deconv", deconv_xyz);
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 output"),
            size: (positions.len() * std::mem::size_of::<Complex32>()) as u64,
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

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 load bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &self.layout_padding_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_3d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }

        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D IFFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_inverse_split(&mut encoder, &re_buffer, &im_buffer);

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 interpolate bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &self.layout_padding_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_3d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, positions.len())
    }

    /// Execute fast gridded Type-2 3D NUFFT with pre-allocated buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_3d_with_buffers(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers3D,
        kernel_width: usize,
        lengths: (f32, f32, f32),
        beta: f32,
        i0_beta: f32,
        deconv_xyz: &[f32],
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let (nx, ny, nz) = buffers.shape;
        let (mx, my, mz) = buffers.oversampled;
        let (lx, ly, lz) = lengths;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let grid_len = mx * my * mz;
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();

        let coeff_buffer = storage_buffer(device, "apollo-nufft-wgpu fast3d type2 coeffs", modes);

        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(&buffers.deconv_buffer, 0, bytemuck::cast_slice(deconv_xyz));

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 output"),
            size: (positions.len() * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 load bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &self.layout_padding_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_3d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }

        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D IFFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_inverse_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast3d type2 interpolate bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &self.layout_padding_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast3d type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_3d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, positions.len())
    }

    /// Execute fast gridded Type-2 3D NUFFT and return diagnostic grid snapshots.
    #[cfg(any(test, feature = "diagnostics"))]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_3d_with_diagnostics(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers3D,
        kernel_width: usize,
        lengths: (f32, f32, f32),
        beta: f32,
        i0_beta: f32,
        deconv_xyz: &[f32],
        modes: &[Complex32],
        positions: &[(f32, f32, f32)],
    ) -> NufftWgpuResult<(Vec<Complex32>, NufftType2GridDiagnostics)> {
        let (nx, ny, nz) = buffers.shape;
        let (mx, my, mz) = buffers.oversampled;
        let (lx, ly, lz) = lengths;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let grid_len = mx * my * mz;
        let position_data: Vec<Position3Pod> = positions
            .iter()
            .map(|(x, y, z)| Position3Pod {
                x: *x,
                y: *y,
                z: *z,
                _pad: 0.0,
            })
            .collect();
        let coeff_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu diagnostic fast3d type2 coeffs",
            modes,
        );
        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(&buffers.deconv_buffer, 0, bytemuck::cast_slice(deconv_xyz));

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 output"),
            size: (positions.len() * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 load bg"),
            layout: &self.fast_3d_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &self.layout_padding_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 load encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu diagnostic fast3d type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_3d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(grid_len as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let after_load = read_split_grid_snapshot(
            device,
            queue,
            &buffers.re_buffer,
            &buffers.im_buffer,
            grid_len,
        )?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 ifft encoder"),
        });
        let fft =
            GpuFft3d::new(Arc::clone(device), Arc::clone(queue), mx, my, mz).map_err(|_| {
                NufftWgpuError::InvalidPlan {
                    message: "oversampled 3D IFFT plan is invalid for WGPU execution",
                }
            })?;
        fft.encode_inverse_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);
        queue.submit(std::iter::once(encoder.finish()));
        let after_ifft = read_split_grid_snapshot(
            device,
            queue,
            &buffers.re_buffer,
            &buffers.im_buffer,
            grid_len,
        )?;

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 interpolate bg"),
            layout: &self.fast_3d_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &self.layout_padding_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coeff_buffer),
                binding(7, &fast_3d_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast3d type2 interpolate encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu diagnostic fast3d type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_3d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let output = read_complex_buffer(device, queue, &output_buffer, positions.len())?;
        Ok((
            output,
            NufftType2GridDiagnostics {
                after_load,
                after_ifft,
            },
        ))
    }
}
